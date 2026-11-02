/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq.calibrate;

import org.elasticsearch.index.codec.vectors.cluster.CentroidOps;
import org.elasticsearch.index.codec.vectors.cluster.HierarchicalKMeans;
import org.elasticsearch.index.codec.vectors.cluster.KMeansFloatVectorValues;
import org.elasticsearch.logging.LogManager;
import org.elasticsearch.logging.Logger;

import java.io.IOException;
import java.util.Arrays;

import static org.elasticsearch.core.Strings.format;

/**
 * Error model for error in scalar quantization.
 * Estimates the standard deviation of the error in distance/similarity after quantizing
 * queries and documents. Used by calibration to predict recall.
 * Fitting is two-stage: {@link #estimateErrorScalingFit} sweeps sample sizes at fixed
 * 4q/1d encoding to fit the slope of {@code log(error_std)} vs {@code log(L/N)}; then
 * {@link #estimateMagnitudeModel} sweeps at a target candidate pair and fits magnitude
 * while reusing that slope.
 */
public final class ErrorModel {

    private static final Logger logger = LogManager.getLogger(ErrorModel.class);

    static final int N_QUERY_CLUSTERS = 32;

    static final int[] SAMPLE_SIZES_SCALING = { 4096, 5120, 6144, 7168, 8192, 9216, 10240, 11264, 12288, 13312, 14336, 15360 };

    static final int[] SAMPLE_SIZES_MAGNITUDE = { 2048, 3072, 4096 };

    private ErrorModel() {}

    /**
     * Quantized representation error standard deviation. Builds the query clustering for the doc clustering it is
     * given, then asks {@code quantizer} to measure the error of {@code candidate} against that shared clustering.
     */
    static QuantizedQueryErrorResult quantizedRepErrorStd(
        CalibrationSource source,
        boolean usePreconditioned,
        int nDocs,
        int[] docAssignments,
        float[][] docCentroids,
        CalibrationQuantizer quantizer,
        CalibrationQuantizer.Candidate candidate,
        HierarchicalKMeans<float[]> kmeans,
        float[][] warmStartQueryCentroids,
        CalibrationQuantizer.Scratch scratch
    ) throws IOException {
        int dimWork = source.workingDim();

        int nDocClusters = docCentroids.length;
        if (nDocClusters == 0 || nDocs == 0) {
            return new QuantizedQueryErrorResult(1.0, docCentroids.length > 0 ? new float[][] { docCentroids[0].clone() } : new float[0][]);
        }

        int effectiveQueryClusters = Math.min(N_QUERY_CLUSTERS, nDocClusters);
        float[][] queryCentroids;
        int[] docCentroidAssignments;
        if (effectiveQueryClusters <= 1) {
            queryCentroids = new float[][] { docCentroids[0].clone() };
            docCentroidAssignments = new int[nDocClusters];
        } else {
            int targetSize = Math.max(1, nDocClusters / effectiveQueryClusters);
            KMeansFloatVectorValues centroidVectors = KMeansFloatVectorValues.build(Arrays.asList(docCentroids), null, dimWork);
            var queryClustering = kmeans.cluster(centroidVectors, targetSize, warmStartQueryCentroids);
            queryCentroids = queryClustering.centroids();
            docCentroidAssignments = queryClustering.assignments();
        }

        CalibrationQuantizer.ResidualSample sample = new CalibrationQuantizer.ResidualSample(
            source,
            usePreconditioned,
            nDocs,
            docAssignments,
            docCentroids,
            queryCentroids,
            docCentroidAssignments
        );
        return new QuantizedQueryErrorResult(quantizer.measureError(candidate, sample, scratch), queryCentroids);
    }

    private record QuantizedQueryErrorResult(double std, float[][] queryCentroids) {}

    /**
     * Clusters the corpus prefix and measures quantized representation error standard deviation.
     */
    static QuantizedErrorComputeResult quantizedRepErrorStdWithCentroids(
        CalibrationSource source,
        boolean usePreconditioned,
        int nDocs,
        int nDocsPerCluster,
        CalibrationQuantizer quantizer,
        CalibrationQuantizer.Candidate candidate,
        HierarchicalKMeans<float[]> kmeans,
        float[][] warmStartDocCentroids,
        float[][] warmStartQueryCentroids,
        CalibrationQuantizer.Scratch scratch
    ) throws IOException {
        KMeansFloatVectorValues corpusVectors = KMeansFloatVectorValues.wrap(source.vectors(), source.corpusOrdinals(), nDocs);
        var docClusters = kmeans.cluster(corpusVectors, nDocsPerCluster, warmStartDocCentroids);

        float[][] docCentroids = docClusters.centroids();
        int[] flatAssignments = docClusters.assignments();
        if (docCentroids.length == 0) {
            return new QuantizedErrorComputeResult(1.0, docCentroids, warmStartQueryCentroids);
        }

        // K-means distances are preserved under orthogonal rotation, so assignments are the same in both spaces.
        // Rotating each centroid gives the exact centroid of the preconditioned cluster, so we can cluster in
        // original space and then rotate the resulting centroids rather than re-clustering preconditioned vectors.
        if (usePreconditioned && source.preconditioner() != null) {
            float[][] preconditionedCentroids = new float[docCentroids.length][source.workingDim()];
            for (int i = 0; i < docCentroids.length; i++) {
                source.preconditioner().applyTransform(docCentroids[i], preconditionedCentroids[i]);
            }
            docCentroids = preconditionedCentroids;
        }

        QuantizedQueryErrorResult queryError = quantizedRepErrorStd(
            source,
            usePreconditioned,
            nDocs,
            flatAssignments,
            docCentroids,
            quantizer,
            candidate,
            kmeans,
            warmStartQueryCentroids,
            scratch
        );

        return new QuantizedErrorComputeResult(queryError.std(), docCentroids, queryError.queryCentroids());
    }

    record QuantizedErrorComputeResult(double std, float[][] docCentroids, float[][] queryCentroids) {}

    public static ErrorScalingFit estimateErrorScalingFit(CalibrationSource source, int nDocsPerCluster, CalibrationQuantizer quantizer) {
        return estimateErrorScalingFit(
            source,
            nDocsPerCluster,
            quantizer,
            HierarchicalKMeans.ofSerial(CentroidOps.FLOAT, source.workingDim())
        );
    }

    private static ErrorScalingFit estimateErrorScalingFit(
        CalibrationSource source,
        int nDocsPerCluster,
        CalibrationQuantizer quantizer,
        HierarchicalKMeans<float[]> kmeans
    ) {
        logger.debug("Fitting error scaling model");
        long scalingStartNanos = System.nanoTime();

        double logNDocsPerCluster = Math.log(nDocsPerCluster);
        Regression.OLSAccumulator state = new Regression.OLSAccumulator();
        float[][] warmStartDocCentroids = null;
        float[][] warmStartQueryCentroids = null;
        int corpusLength;

        int maxNDocs = Math.min(SAMPLE_SIZES_SCALING[SAMPLE_SIZES_SCALING.length - 1], source.corpusOrdinals().length);
        CalibrationQuantizer.Scratch scratch = quantizer.newScratch(source, maxNDocs);
        for (int i = 0; i < SAMPLE_SIZES_SCALING.length; i++) {
            int sampleSize = SAMPLE_SIZES_SCALING[i];
            if (sampleSize > source.corpusOrdinals().length) {
                break;
            }
            corpusLength = sampleSize;
            try {
                QuantizedErrorComputeResult computed = quantizedRepErrorStdWithCentroids(
                    source,
                    true,
                    corpusLength,
                    nDocsPerCluster,
                    quantizer,
                    quantizer.scalingCandidate(),
                    kmeans,
                    warmStartDocCentroids,
                    warmStartQueryCentroids,
                    scratch
                );
                warmStartDocCentroids = computed.docCentroids();
                warmStartQueryCentroids = computed.queryCentroids();
                double x = logNDocsPerCluster - Math.log(sampleSize);
                double y = Math.log(Math.max(computed.std(), 1e-38));
                state.update(x, y);
                if ((i + 1) % 4 == 0) {
                    logger.debug("Processed {}/{} scaling samples", i + 1, SAMPLE_SIZES_SCALING.length);
                }
            } catch (IOException e) {
                logger.warn("failed to compute quantization error std for sample size [{}]", sampleSize, e);
            }
        }

        Regression.OLSResult rawParams = state.fit();
        if (rawParams == Regression.OLSResult.ZERO) {
            return new ErrorScalingFit(
                new QuantizationErrorStdModel(Regression.OLSResult.ZERO),
                warmStartDocCentroids,
                warmStartQueryCentroids
            );
        }

        // Clamp to 0 so the error model degenerates to a constant (no corpus-size extrapolation) rather than
        // producing unbounded estimates.
        // This can happen when the k-means warm-start introduces non-monotone error steps
        // over the narrow x-range of the sweep (~1.3 log-units for typical nDocsPerCluster).
        Regression.OLSResult params;
        if (rawParams.beta1() < 0) {
            logger.debug(
                () -> format(
                    "Error scaling fit produced negative slope [%.4f] (R²=[%.4f]); "
                        + "clamping to 0 (constant error model, no corpus-size extrapolation).",
                    rawParams.beta1(),
                    state.r2(rawParams)
                )
            );
            params = new Regression.OLSResult(rawParams.beta0(), 0.0, 0, 0, 0, rawParams.sigmaSq());
        } else {
            params = rawParams;
        }

        double scalingSeconds = (System.nanoTime() - scalingStartNanos) / 1_000_000_000.0;
        logger.debug(
            () -> format(
                "Fit error scaling model in %.5fs\nquantization error ∝ (L/N)^%.4f (R² = %.4f)",
                scalingSeconds,
                params.beta1(),
                state.r2(params)
            )
        );

        return new ErrorScalingFit(new QuantizationErrorStdModel(params), warmStartDocCentroids, warmStartQueryCentroids);
    }

    /**
     * Estimate the magnitude of quantization error for a specific candidate quantization.
     * Sweeps sample sizes at fixed {@code nDocsPerCluster} and fits a plug-in regression
     * that reuses the slope from {@link ErrorScalingFit#scalingModel()}.
     *
     * @param quantizer the quantizer whose error is being modeled
     * @param candidate the quantizer candidate (e.g. bit widths) to measure
     */
    public static QuantizationErrorStdModel estimateMagnitudeModel(
        ErrorScalingFit scalingFit,
        CalibrationSource source,
        boolean usePreconditionedQueries,
        CalibrationQuantizer quantizer,
        CalibrationQuantizer.Candidate candidate,
        int nDocsPerCluster
    ) {
        return estimateMagnitudeModel(
            scalingFit,
            source,
            usePreconditionedQueries,
            quantizer,
            candidate,
            nDocsPerCluster,
            HierarchicalKMeans.ofSerial(CentroidOps.FLOAT, source.workingDim())
        );
    }

    private static QuantizationErrorStdModel estimateMagnitudeModel(
        ErrorScalingFit scalingFit,
        CalibrationSource source,
        boolean usePreconditionedQueries,
        CalibrationQuantizer quantizer,
        CalibrationQuantizer.Candidate candidate,
        int nDocsPerCluster,
        HierarchicalKMeans<float[]> kmeans
    ) {
        QuantizationErrorStdModel scalingModel = scalingFit.scalingModel();
        long magnitudeStartNanos = System.nanoTime();

        double logNDocsPerCluster = Math.log(nDocsPerCluster);
        Regression.OLSAccumulator state = new Regression.OLSAccumulator();
        float[][] docWarmStart = scalingFit.lastDocCentroids();
        float[][] queryWarmStart = scalingFit.lastQueryCentroids();

        int maxNDocs = Math.min(SAMPLE_SIZES_MAGNITUDE[SAMPLE_SIZES_MAGNITUDE.length - 1], source.corpusOrdinals().length);
        CalibrationQuantizer.Scratch scratch = quantizer.newScratch(source, maxNDocs);

        for (int sampleSize : SAMPLE_SIZES_MAGNITUDE) {
            if (sampleSize > source.corpusOrdinals().length) {
                break;
            }
            try {
                QuantizedErrorComputeResult computed = quantizedRepErrorStdWithCentroids(
                    source,
                    usePreconditionedQueries,
                    sampleSize,
                    nDocsPerCluster,
                    quantizer,
                    candidate,
                    kmeans,
                    docWarmStart,
                    queryWarmStart,
                    scratch
                );
                docWarmStart = computed.docCentroids();
                queryWarmStart = computed.queryCentroids();
                double x = logNDocsPerCluster - Math.log(sampleSize);
                double y = Math.log(Math.max(computed.std(), 1e-38));
                state.update(x, y);
            } catch (IOException e) {
                logger.warn("failed to compute quantization error std for magnitude sample size [{}]", sampleSize, e);
            }
        }

        Regression.OLSResult params = state.fitPlugin(scalingModel.params());
        if (params == Regression.OLSResult.ZERO) {
            return scalingModel;
        }

        double magnitudeSeconds = (System.nanoTime() - magnitudeStartNanos) / 1_000_000_000.0;
        logger.debug(
            () -> format(
                "Fit error magnitude model in %.2fs\nquantization error %.4f (L/N)^%.4f (R² = %.4f)",
                magnitudeSeconds,
                Math.exp(params.beta0()),
                params.beta1(),
                state.r2(params)
            )
        );

        return new QuantizationErrorStdModel(params);
    }

    /** Corpus sample size for the single-shot real-residual magnitude measurement. */
    static final int REAL_RESIDUAL_SAMPLE = 2048;

    /**
     * Opaque per-sweep state for the real-residual magnitude path: reusable quantizer scratch, a serial k-means
     * instance, and the (encoding-independent) clustering from the first candidate, reused as a warm start
     * for subsequent candidates so k-means is not recomputed from scratch per encoding. Construct once per
     * calibration and thread through every candidate.
     */
    public static final class RealResidualState {
        private final CalibrationQuantizer.Scratch scratch;
        private final HierarchicalKMeans<float[]> kmeans;
        private final int nDocs;
        private QuantizedErrorComputeResult shared;
        private boolean sharedPreconditioned; // whether {@link #shared} was computed with {@code usePreconditioned=true}

        public RealResidualState(CalibrationSource source, CalibrationQuantizer quantizer) {
            this.nDocs = Math.min(REAL_RESIDUAL_SAMPLE, source.corpusOrdinals().length);
            this.kmeans = HierarchicalKMeans.ofSerial(CentroidOps.FLOAT, source.workingDim());
            this.scratch = quantizer.newScratch(source, nDocs);
        }

        public int ndocs() {
            return nDocs;
        }
    }

    /**
     * Estimates the quantization error-std model for {@code candidate} from <em>real</em> corpus
     * residuals. The clustering warm start is reused across candidates via {@code state} so k-means is
     * not recomputed per encoding.
     * <p>
     * Measures OSQ error once at {@link #REAL_RESIDUAL_SAMPLE} and anchors the intercept at that sample
     * size. The manifold slope is used as the scaling exponent, so evaluating at the real corpus size {@code N}
     * extrapolates as {@code errorStd = measuredStd × (REAL_RESIDUAL_SAMPLE / N)^invDim}.
     */
    public static QuantizationErrorStdModel estimateMagnitudeFromRealResiduals(
        double invDim,
        CalibrationSource source,
        boolean usePreconditionedQueries,
        CalibrationQuantizer quantizer,
        CalibrationQuantizer.Candidate candidate,
        int nDocsPerCluster,
        RealResidualState state
    ) throws IOException {
        // doc warm start: reuse only when shared.docCentroids() is in original space (sharedPreconditioned=false).
        float[][] warmDoc = (state.shared != null && !state.sharedPreconditioned) ? state.shared.docCentroids() : null;

        // query warm start: reuse only when the space of shared.queryCentroids() matches the current call.
        float[][] warmQuery = (state.shared != null && state.sharedPreconditioned == usePreconditionedQueries)
            ? state.shared.queryCentroids()
            : null;
        QuantizedErrorComputeResult r = quantizedRepErrorStdWithCentroids(
            source,
            usePreconditionedQueries,
            state.nDocs,
            nDocsPerCluster,
            quantizer,
            candidate,
            state.kmeans,
            warmDoc,
            warmQuery,
            state.scratch
        );
        if (state.shared == null) {
            state.shared = r;
            state.sharedPreconditioned = usePreconditionedQueries;
        }
        double measured = r.std();
        // The manifold is fit in distance units for every metric (see ManifoldModel), so invDim is positive
        // throughout and needs no sign correction for dot-like similarities any more.
        // single measurement anchored at state.nDocs, so evaluating at N gives measuredStd × (state.nDocs / N)^invDim
        double beta0 = Math.log(Math.max(measured, 1e-38)) - invDim * (Math.log(nDocsPerCluster) - Math.log(state.nDocs));
        return new QuantizationErrorStdModel(new Regression.OLSResult(beta0, invDim, 0, 0, 0, 0));
    }
}
