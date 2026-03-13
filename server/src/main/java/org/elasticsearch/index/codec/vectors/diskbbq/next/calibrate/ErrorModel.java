/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq.next.calibrate;

import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.elasticsearch.index.codec.vectors.OptimizedScalarQuantizer;
import org.elasticsearch.index.codec.vectors.cluster.HierarchicalKMeans;
import org.elasticsearch.index.codec.vectors.cluster.KMeansFloatVectorValues;
import org.elasticsearch.index.codec.vectors.cluster.KMeansResult;
import org.elasticsearch.logging.LogManager;
import org.elasticsearch.logging.Logger;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Error model for representation (quantization) error in scalar quantization.
 * Estimates the standard deviation of the error in distance/similarity after quantizing
 * queries and documents. Used by calibration to predict recall.
 * <p>
 * Corpus vectors are accessed lazily via {@link FloatVectorValues} and ordinal arrays
 * to avoid materializing a large {@code float[][]}. Queries are kept materialized as
 * they are small (~128 vectors).
 * <p>
 * Fits two OLS regressions:
 * <ul>
 *   <li><b>Scaling model</b>: sweeps (nDocsPerCluster, sampleSize) pairs to fit
 *       {@code log(error_std) ~ beta0 + beta1 * (log(L) - log(N))}.</li>
 *   <li><b>Magnitude model</b>: for a specific (qbits, dbits), sweeps cluster sizes at
 *       a fixed sample size, fitting with a plug-in regression that reuses the slope
 *       from the scaling model.</li>
 * </ul>
 */
public final class ErrorModel {

    private static final Logger logger = LogManager.getLogger(ErrorModel.class);

    static final int N_QUERY_CLUSTERS = 32;

    static final int[] N_DOCS_PER_CLUSTER_SCALING = { 256, 240, 224, 216, 200, 184, 176, 160, 144, 136, 120, 104, 96, 80, 64 };
    static final int[] SAMPLE_SIZES_SCALING = {
        8192,
        8872,
        9552,
        9892,
        10572,
        11252,
        11592,
        12272,
        12952,
        13292,
        13972,
        14652,
        14992,
        15672,
        16352 };

    static final int[] N_DOCS_PER_CLUSTER_SCALING_FAST = { 256, 216, 176, 136, 104, 80, 64 };
    static final int[] SAMPLE_SIZES_SCALING_FAST = { 8192, 9892, 11592, 13292, 14652, 15672, 16352 };

    static final int[] N_DOCS_PER_CLUSTER_MAGNITUDE = { 64, 72, 80, 88, 96, 104, 112, 120, 128 };
    static final int[] N_DOCS_PER_CLUSTER_MAGNITUDE_FAST = { 64, 80, 96, 112, 128 };
    static final int SAMPLE_SIZE_MAGNITUDE = 4096;

    private ErrorModel() {}

    /**
     * Exact similarity between two vectors, consistent with the metric convention
     * where higher values indicate greater similarity.
     */
    static double simExact(VectorSimilarityFunction sim, int dim, float[] y, float[] x) {
        if (sim == VectorSimilarityFunction.EUCLIDEAN) {
            return 2.0 * CalibrationUtils.dot(dim, y, x) - CalibrationUtils.dot(dim, x, x);
        }
        return CalibrationUtils.dot(dim, y, x);
    }

    static long dotInt(int dim, int[] x, int xOff, int[] y, int yOff) {
        long sum = 0;
        for (int i = 0; i < dim; i++) {
            sum += (long) x[xOff + i] * y[yOff + i];
        }
        return sum;
    }

    /**
     * Centroid representation error standard deviation. For each query, finds the top 5%
     * of clusters by similarity and measures the error between the exact similarity to
     * each document and the similarity to its assigned centroid.
     *
     * @param fvv the vector values source for lazy access
     * @param corpusOrdinals ordinal indices into {@code fvv}
     * @param cosine if true, normalize corpus vectors on-the-fly
     */
    static double centroidRepErrorStd(
        VectorSimilarityFunction sim,
        int dim,
        float[][] queries,
        FloatVectorValues fvv,
        int[] corpusOrdinals,
        boolean cosine,
        int[][] perClusterAssignments,
        float[][] centroids
    ) throws IOException {
        int k = perClusterAssignments.length;
        int visit = Math.max(1, (5 * k + 99) / 100);
        OnlineMeanAndVariance moments = new OnlineMeanAndVariance();

        Integer[] order = new Integer[k];
        for (int i = 0; i < k; i++) {
            order[i] = i;
        }
        float[] scratch = cosine ? new float[dim] : null;

        for (float[] query : queries) {
            Arrays.sort(order, (a, b) -> Double.compare(simExact(sim, dim, query, centroids[b]), simExact(sim, dim, query, centroids[a])));
            for (int idx = 0; idx < Math.min(visit, k); idx++) {
                int ci = order[idx];
                float[] cent = centroids[ci];
                for (int j : perClusterAssignments[ci]) {
                    float[] doc = fvv.vectorValue(corpusOrdinals[j]);
                    if (cosine) {
                        doc = CalibrationUtils.copyAndNormalize(doc, scratch);
                    }
                    double err = simExact(sim, dim, query, doc) - simExact(sim, dim, query, cent);
                    moments.add(err);
                }
            }
        }
        return Math.sqrt(moments.var());
    }

    /**
     * Quantized representation error standard deviation. Quantizes doc residuals and
     * query residuals using OSQ, estimates dot products, and compares to exact
     * similarities for the top-5k ranked documents per query.
     *
     * @param fvv the vector values source for lazy access
     * @param corpusOrdinals ordinal indices into {@code fvv}
     * @param cosine if true, normalize corpus vectors on-the-fly
     */
    static double quantizedRepErrorStd(
        VectorSimilarityFunction sim,
        int dim,
        float[][] queries,
        FloatVectorValues fvv,
        int[] corpusOrdinals,
        boolean cosine,
        int[] docAssignments,
        float[][] docCentroids,
        int nQueryClusters,
        int nDocsPerCluster,
        int qbits,
        int dbits,
        int k
    ) throws IOException {
        int nDocs = docAssignments.length;
        int nDocClusters = docCentroids.length;
        if (nDocClusters == 0 || nDocs == 0) {
            return 1.0;
        }

        int effectiveQueryClusters = Math.min(nQueryClusters, nDocClusters);
        float[][] queryCentroids;
        int[] docCentroidAssignments;
        if (effectiveQueryClusters <= 1) {
            queryCentroids = new float[][] { docCentroids[0].clone() };
            docCentroidAssignments = new int[nDocClusters];
        } else {
            int targetSize = Math.max(1, nDocClusters / effectiveQueryClusters);
            KMeansFloatVectorValues centroidVectors = KMeansFloatVectorValues.build(Arrays.asList(docCentroids), null, dim);
            KMeansResult queryClustering = HierarchicalKMeans.ofSerial(dim).cluster(centroidVectors, targetSize);
            queryCentroids = queryClustering.centroids();
            docCentroidAssignments = queryClustering.assignments();
        }
        int actualQueryClusters = queryCentroids.length;

        double[] centroidDotCentroid = new double[nDocClusters];
        for (int i = 0; i < nDocClusters; i++) {
            centroidDotCentroid[i] = CalibrationUtils.dot(dim, queryCentroids[docCentroidAssignments[i]], docCentroids[i]);
        }

        OptimizedScalarQuantizer quantizer = new OptimizedScalarQuantizer(VectorSimilarityFunction.EUCLIDEAN);
        float[] residualScratch = new float[dim];
        float[] normScratch = cosine ? new float[dim] : null;

        float[] docLower = new float[nDocs];
        float[] docUpper = new float[nDocs];
        int[] docL1 = new int[nDocs];
        int[][] docQuantized = new int[nDocs][dim];
        double[] corpusDotCentroid = new double[nDocs];
        double[] docDotDoc = sim == VectorSimilarityFunction.EUCLIDEAN ? new double[nDocs] : null;

        for (int i = 0; i < nDocs; i++) {
            float[] doc = fvv.vectorValue(corpusOrdinals[i]);
            if (cosine) {
                doc = CalibrationUtils.copyAndNormalize(doc, normScratch);
            }
            int qc = docCentroidAssignments[docAssignments[i]];
            corpusDotCentroid[i] = CalibrationUtils.dot(dim, queryCentroids[qc], doc);
            var qr = quantizer.scalarQuantize(doc, residualScratch, docQuantized[i], (byte) dbits, docCentroids[docAssignments[i]]);
            docLower[i] = qr.lowerInterval();
            docUpper[i] = qr.upperInterval();
            docL1[i] = qr.quantizedComponentSum();
            if (docDotDoc != null) {
                docDotDoc[i] = CalibrationUtils.dot(dim, doc, doc);
            }
        }

        OnlineMeanAndVariance moments = new OnlineMeanAndVariance();
        double dScale = 1.0 / ((1 << dbits) - 1);
        double qScale = 1.0 / ((1 << qbits) - 1);

        float[] queryLower = new float[actualQueryClusters];
        float[] queryUpper = new float[actualQueryClusters];
        int[] queryL1 = new int[actualQueryClusters];
        int[][] queryQuantized = new int[actualQueryClusters][dim];

        for (float[] query : queries) {
            for (int qc = 0; qc < actualQueryClusters; qc++) {
                var qr = quantizer.scalarQuantize(query, residualScratch, queryQuantized[qc], (byte) qbits, queryCentroids[qc]);
                queryLower[qc] = qr.lowerInterval();
                queryUpper[qc] = qr.upperInterval();
                queryL1[qc] = qr.quantizedComponentSum();
            }

            double[] queryDotCentroid = new double[nDocClusters];
            for (int i = 0; i < nDocClusters; i++) {
                queryDotCentroid[i] = CalibrationUtils.dot(dim, query, docCentroids[i]);
            }

            double[] simOsq = new double[nDocs];
            for (int i = 0; i < nDocs; i++) {
                int dc = docAssignments[i];
                int qc = docCentroidAssignments[dc];

                double ad = docLower[i];
                double ld = dScale * (docUpper[i] - docLower[i]);
                double aq = queryLower[qc];
                double lq = qScale * (queryUpper[qc] - queryLower[qc]);

                long intDot = dotInt(dim, docQuantized[i], 0, queryQuantized[qc], 0);
                double dotEst = ad * aq * dim + aq * ld * docL1[i] + ad * lq * queryL1[qc] + ld * lq * intDot;

                dotEst += corpusDotCentroid[i] + queryDotCentroid[dc] - centroidDotCentroid[dc];

                if (sim == VectorSimilarityFunction.EUCLIDEAN) {
                    dotEst = 2.0 * dotEst - docDotDoc[i];
                }
                simOsq[i] = dotEst;
            }

            Integer[] order = new Integer[nDocs];
            for (int i = 0; i < nDocs; i++) {
                order[i] = i;
            }
            Arrays.sort(order, (a, b) -> Double.compare(simOsq[b], simOsq[a]));

            int topN = Math.min(5 * k, nDocs);
            for (int i = 0; i < topN; i++) {
                int docIdx = order[i];
                float[] doc = fvv.vectorValue(corpusOrdinals[docIdx]);
                if (cosine) {
                    doc = CalibrationUtils.copyAndNormalize(doc, normScratch);
                }
                double exact = simExact(sim, dim, query, doc);
                moments.add(exact - simOsq[docIdx]);
            }
        }

        return Math.sqrt(3.0 * moments.var());
    }

    /**
     * Clusters the corpus and measures both centroid and quantized representation error
     * standard deviations for the given configuration.
     *
     * @param fvv the vector values source for lazy access
     * @param corpusOrdinals ordinal indices into {@code fvv}
     * @param cosine if true, normalize corpus vectors on-the-fly
     * @return {@code double[]{centroidStd, quantizedStd}}
     */
    static double[] repErrorStds(
        VectorSimilarityFunction sim,
        int dim,
        float[][] queries,
        FloatVectorValues fvv,
        int[] corpusOrdinals,
        boolean cosine,
        int nQueryClusters,
        int nDocsPerCluster,
        int qbits,
        int dbits,
        int k
    ) throws IOException {
        KMeansFloatVectorValues corpusVectors = KMeansFloatVectorValues.wrap(fvv, corpusOrdinals);
        KMeansResult docClusters = HierarchicalKMeans.ofSerial(dim).cluster(corpusVectors, nDocsPerCluster);

        float[][] centroids = docClusters.centroids();
        int[] flatAssignments = docClusters.assignments();
        if (centroids.length == 0) {
            return new double[] { 1.0, 1.0 };
        }

        int nClusters = centroids.length;
        List<List<Integer>> perClusterLists = new ArrayList<>(nClusters);
        for (int i = 0; i < nClusters; i++) {
            perClusterLists.add(new ArrayList<>());
        }
        for (int i = 0; i < flatAssignments.length; i++) {
            perClusterLists.get(flatAssignments[i]).add(i);
        }
        int[][] perClusterAssignments = new int[nClusters][];
        for (int i = 0; i < nClusters; i++) {
            List<Integer> list = perClusterLists.get(i);
            perClusterAssignments[i] = list.stream().mapToInt(Integer::intValue).toArray();
        }

        double cStd = centroidRepErrorStd(sim, dim, queries, fvv, corpusOrdinals, cosine, perClusterAssignments, centroids);
        double qStd = quantizedRepErrorStd(
            sim,
            dim,
            queries,
            fvv,
            corpusOrdinals,
            cosine,
            flatAssignments,
            centroids,
            nQueryClusters,
            nDocsPerCluster,
            qbits,
            dbits,
            k
        );

        return new double[] { cStd, qStd };
    }

    /**
     * Plug-in regression that reuses the slope from the scaling model and only fits the
     * intercept. Used by the magnitude model to avoid overfitting with few data points.
     */
    static Regression.OLSResult fitRepErrorStdPlugin(
        RepErrorStdModel scalingModel,
        double[] logNDocsPerCluster,
        double[] logSampleSizes,
        double[] logErrorStd
    ) {
        int m = logErrorStd.length;
        double beta1 = scalingModel.qparams().beta1();
        double var1 = scalingModel.qparams().var1();

        double sumRes = 0.0;
        double sumX = 0.0;
        for (int i = 0; i < m; i++) {
            double x = logNDocsPerCluster[i] - logSampleSizes[i];
            sumRes += logErrorStd[i] - beta1 * x;
            sumX += x;
        }
        double beta0 = sumRes / m;
        double xBar = sumX / m;

        double rss = 0.0;
        for (int i = 0; i < m; i++) {
            double x = logNDocsPerCluster[i] - logSampleSizes[i];
            double err = logErrorStd[i] - (beta0 + beta1 * x);
            rss += err * err;
        }
        double sigmaSq = m > 1 ? rss / (m - 1) : 0.0;

        return new Regression.OLSResult(beta0, beta1, (sigmaSq / m) + (xBar * xBar * var1), var1, -xBar * var1, sigmaSq);
    }

    /**
     * Estimate the scaling of representation error using default (full) sweep parameters.
     */
    public static RepErrorStdModel estimateRepErrorStdScalingParameter(
        VectorSimilarityFunction similarityFunction,
        int dim,
        float[][] queries,
        FloatVectorValues fvv,
        int[] corpusOrdinals,
        boolean cosine,
        int k
    ) {
        return estimateRepErrorStdScalingParameter(
            similarityFunction,
            dim,
            queries,
            fvv,
            corpusOrdinals,
            cosine,
            k,
            N_DOCS_PER_CLUSTER_SCALING,
            SAMPLE_SIZES_SCALING
        );
    }

    /**
     * Estimate the scaling of representation error using reduced sweep parameters for faster execution.
     */
    public static RepErrorStdModel estimateRepErrorStdScalingParameterFast(
        VectorSimilarityFunction similarityFunction,
        int dim,
        float[][] queries,
        FloatVectorValues fvv,
        int[] corpusOrdinals,
        boolean cosine,
        int k
    ) {
        return estimateRepErrorStdScalingParameter(
            similarityFunction,
            dim,
            queries,
            fvv,
            corpusOrdinals,
            cosine,
            k,
            N_DOCS_PER_CLUSTER_SCALING_FAST,
            SAMPLE_SIZES_SCALING_FAST
        );
    }

    /**
     * Estimate the scaling of representation error by sweeping (nDocsPerCluster, sampleSize) pairs,
     * clustering the corpus at each, measuring centroid and quantized error, and fitting OLS on
     * {@code log(error_std) ~ beta0 + beta1 * (log(L) - log(N))}.
     */
    static RepErrorStdModel estimateRepErrorStdScalingParameter(
        VectorSimilarityFunction similarityFunction,
        int dim,
        float[][] queries,
        FloatVectorValues fvv,
        int[] corpusOrdinals,
        boolean cosine,
        int k,
        int[] nDocsPerClusterArray,
        int[] sampleSizesArray
    ) {
        int m = nDocsPerClusterArray.length;
        int nDocsTotal = corpusOrdinals.length;

        List<Double> logCentroidStds = new ArrayList<>();
        List<Double> logQuantizedStds = new ArrayList<>();

        for (int i = 0; i < m; i++) {
            int ss = sampleSizesArray[i];
            if (ss > nDocsTotal) {
                break;
            }
            if (ss < 2) {
                continue;
            }
            int[] subOrdinals = Arrays.copyOf(corpusOrdinals, ss);
            try {
                double[] stds = repErrorStds(
                    similarityFunction,
                    dim,
                    queries,
                    fvv,
                    subOrdinals,
                    cosine,
                    N_QUERY_CLUSTERS,
                    nDocsPerClusterArray[i],
                    4,
                    1,
                    k
                );
                logCentroidStds.add(Math.log(Math.max(stds[0], 1e-38)));
                logQuantizedStds.add(Math.log(Math.max(stds[1], 1e-38)));
            } catch (IOException e) {
                logger.warn("failed to compute rep error stds for sample size [{}]", ss, e);
            }
        }

        int mActual = logCentroidStds.size();
        if (mActual < 2) {
            return new RepErrorStdModel(Regression.OLSResult.ZERO, Regression.OLSResult.ZERO);
        }

        double[] x = new double[mActual];
        double[] logCStd = new double[mActual];
        double[] logQStd = new double[mActual];
        for (int i = 0; i < mActual; i++) {
            x[i] = Math.log(nDocsPerClusterArray[i]) - Math.log(sampleSizesArray[i]);
            logCStd[i] = logCentroidStds.get(i);
            logQStd[i] = logQuantizedStds.get(i);
        }

        Regression.OLSResult cparams = Regression.fitOls(x, logCStd);
        Regression.OLSResult qparams = Regression.fitOls(x, logQStd);

        if (logger.isDebugEnabled()) {
            logger.debug(
                "Fit error scaling models: centroid error {} (L/N)^{}, quantization error (L/N)^{}",
                Math.exp(cparams.beta0()),
                cparams.beta1(),
                qparams.beta1()
            );
        }

        return new RepErrorStdModel(cparams, qparams);
    }

    /**
     * Estimate the magnitude of representation error using default (full) sweep parameters.
     */
    public static RepErrorStdModel estimateRepErrorStdMagnitudeParameter(
        RepErrorStdModel scalingModel,
        VectorSimilarityFunction similarityFunction,
        int dim,
        float[][] queries,
        FloatVectorValues fvv,
        int[] corpusOrdinals,
        boolean cosine,
        int k,
        int qbits,
        int dbits
    ) {
        return estimateRepErrorStdMagnitudeParameter(
            scalingModel,
            similarityFunction,
            dim,
            queries,
            fvv,
            corpusOrdinals,
            cosine,
            k,
            qbits,
            dbits,
            N_DOCS_PER_CLUSTER_MAGNITUDE
        );
    }

    /**
     * Estimate the magnitude of representation error using reduced sweep parameters for faster execution.
     */
    public static RepErrorStdModel estimateRepErrorStdMagnitudeParameterFast(
        RepErrorStdModel scalingModel,
        VectorSimilarityFunction similarityFunction,
        int dim,
        float[][] queries,
        FloatVectorValues fvv,
        int[] corpusOrdinals,
        boolean cosine,
        int k,
        int qbits,
        int dbits
    ) {
        return estimateRepErrorStdMagnitudeParameter(
            scalingModel,
            similarityFunction,
            dim,
            queries,
            fvv,
            corpusOrdinals,
            cosine,
            k,
            qbits,
            dbits,
            N_DOCS_PER_CLUSTER_MAGNITUDE_FAST
        );
    }

    /**
     * Estimate the magnitude of representation error for a specific (qbits, dbits) pair.
     * Sweeps cluster sizes at a fixed sample size and fits a plug-in regression that
     * reuses the slope from the scaling model.
     */
    static RepErrorStdModel estimateRepErrorStdMagnitudeParameter(
        RepErrorStdModel scalingModel,
        VectorSimilarityFunction similarityFunction,
        int dim,
        float[][] queries,
        FloatVectorValues fvv,
        int[] corpusOrdinals,
        boolean cosine,
        int k,
        int qbits,
        int dbits,
        int[] nDocsPerClusterArray
    ) {
        int m = nDocsPerClusterArray.length;
        int sampleSize = Math.min(SAMPLE_SIZE_MAGNITUDE, corpusOrdinals.length);
        int[] subOrdinals = Arrays.copyOf(corpusOrdinals, sampleSize);

        List<Double> logQuantizedStds = new ArrayList<>();
        for (int i = 0; i < m; i++) {
            try {
                double[] stds = repErrorStds(
                    similarityFunction,
                    dim,
                    queries,
                    fvv,
                    subOrdinals,
                    cosine,
                    N_QUERY_CLUSTERS,
                    nDocsPerClusterArray[i],
                    qbits,
                    dbits,
                    k
                );
                logQuantizedStds.add(Math.log(Math.max(stds[1], 1e-38)));
            } catch (IOException e) {
                logger.warn("failed to compute rep error stds for magnitude iteration [{}]", i, e);
            }
        }

        int mActual = logQuantizedStds.size();
        if (mActual < 2) {
            return scalingModel;
        }

        double[] logNDocs = new double[mActual];
        double[] logSizes = new double[mActual];
        double[] logQStd = new double[mActual];
        for (int i = 0; i < mActual; i++) {
            logNDocs[i] = Math.log(nDocsPerClusterArray[i]);
            logSizes[i] = Math.log(SAMPLE_SIZE_MAGNITUDE);
            logQStd[i] = logQuantizedStds.get(i);
        }

        Regression.OLSResult qparams = fitRepErrorStdPlugin(scalingModel, logNDocs, logSizes, logQStd);

        if (logger.isDebugEnabled()) {
            logger.debug("Fit error magnitude model: quantization error {} (L/N)^{}", Math.exp(qparams.beta0()), qparams.beta1());
        }

        return new RepErrorStdModel(scalingModel.cparams(), qparams);
    }
}
