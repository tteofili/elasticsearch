/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq.calibrate;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.elasticsearch.core.WelfordVariance;
import org.elasticsearch.index.codec.vectors.OptimizedScalarQuantizer;
import org.elasticsearch.index.codec.vectors.diskbbq.IvfSegmentConfig;
import org.elasticsearch.index.codec.vectors.diskbbq.QuantEncoding;
import org.elasticsearch.simdvec.ESVectorUtil;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * {@link CalibrationQuantizer} for OSQ (optimal scalar quantization), the quantizer used by the DiskBBQ BBQ
 * writers. Candidates are the {@link QuantEncoding}s that have a distinct (query bits, doc bits) pair, and error is
 * measured by quantizing document and query residuals against their centroids exactly as the OSQ scorer does.
 */
public final class OsqCalibrationQuantizer implements CalibrationQuantizer {

    /** Stateless, so a single instance is shared. */
    public static final OsqCalibrationQuantizer INSTANCE = new OsqCalibrationQuantizer();

    /**
     * Weight applied to doc bits in the calibration cost model
     * ({@code DOC_BITS_WEIGHT * dbits + RERANK_COST_WEIGHT * rerankDepth}).
     * Doc bits represent a permanent per-segment storage and memory cost, so they are weighted
     * more heavily than rerank depth (a per-query compute cost). The value must satisfy
     * {@code DOC_BITS_WEIGHT > RERANK_COST_WEIGHT * (maxRerankDepth - minRerankDepth)} to guarantee
     * that no entry with higher doc bits ever sorts before an entry with lower doc bits — i.e. the
     * cost ordering naturally tiers by doc-bit level without any explicit phase logic.
     * With a rerank weight of 1.3 and rerank depths in [1.25, 3.0] the minimum is
     * {@code 1.3 * 1.75 ≈ 2.28}; 3.0 provides a comfortable margin.
     */
    public static final double DOC_BITS_WEIGHT = 3.0;

    /**
     * A candidate OSQ encoding with the query and doc bit widths used when measuring its error.
     */
    public record OsqCandidate(QuantEncoding encoding, int qbits, int dbits) implements Candidate {
        @Override
        public String id() {
            return "osq-q" + qbits + "d" + dbits;
        }

        @Override
        public IvfSegmentConfig.QuantConfig quantConfig() {
            if (encoding == null) {
                throw new IllegalStateException("measurement-only candidate [" + id() + "] has no encoding to write");
            }
            return new IvfSegmentConfig.OsqConfig(encoding);
        }

        @Override
        public double storageCost() {
            return DOC_BITS_WEIGHT * dbits;
        }

        @Override
        public double queryCost() {
            return qbits;
        }
    }

    /**
     * Candidate encodings paired with their (qbits, dbits) for the calibration model. Each entry encodes the ES
     * {@link QuantEncoding} and the query/doc bit widths used during recall estimation.
     */
    private static final List<OsqCandidate> CANDIDATES = List.of(
        new OsqCandidate(QuantEncoding.ONE_BIT_1BIT_QUERY, 1, 1),
        new OsqCandidate(QuantEncoding.ONE_BIT_4BIT_QUERY, 4, 1),
        new OsqCandidate(QuantEncoding.TWO_BIT_4BIT_QUERY, 4, 2),
        new OsqCandidate(QuantEncoding.FOUR_BIT_SYMMETRIC, 4, 4),
        new OsqCandidate(QuantEncoding.SEVEN_BIT_SYMMETRIC, 7, 7)
    );

    /** The scaling fit always uses (qbits=4, dbits=1), matching {@code ONE_BIT_4BIT_QUERY}. */
    private static final OsqCandidate SCALING_CANDIDATE = new OsqCandidate(QuantEncoding.ONE_BIT_4BIT_QUERY, 4, 1);

    private OsqCalibrationQuantizer() {}

    /**
     * The distinct candidate quantization encodings evaluated during calibration. Exposed so tests can assert
     * that a calibrated segment picked one of these without duplicating (and drifting from) the list here.
     */
    public static Set<QuantEncoding> candidateEncodings() {
        return CANDIDATES.stream().map(OsqCandidate::encoding).collect(Collectors.toUnmodifiableSet());
    }

    /**
     * Returns the candidate with the given bit widths. Bit-width pairs that no {@link QuantEncoding} supports yield a
     * measurement-only candidate (null encoding): its error can be measured, e.g. to study how error varies with
     * bits, but it can never be selected, so {@link OsqCandidate#quantConfig()} throws for it.
     */
    public static OsqCandidate candidate(int qbits, int dbits) {
        return CANDIDATES.stream()
            .filter(c -> c.qbits() == qbits && c.dbits() == dbits)
            .findFirst()
            .orElseGet(() -> new OsqCandidate(null, qbits, dbits));
    }

    @Override
    public List<Candidate> candidates(int maxDocBits) {
        return CANDIDATES.stream().filter(c -> c.dbits() <= maxDocBits).map(c -> (Candidate) c).toList();
    }

    @Override
    public Candidate defaultCandidate() {
        return candidate(4, 1);
    }

    @Override
    public Candidate scalingCandidate() {
        return SCALING_CANDIDATE;
    }

    @Override
    public Scratch newScratch(CalibrationSource source, int maxNDocs) {
        return new OsqScratch(
            maxNDocs,
            source.workingDim(),
            source.cosine(),
            source.similarityFunction() == VectorSimilarityFunction.EUCLIDEAN,
            source.preconditioner() != null
        );
    }

    /**
     * Quantizes doc residuals and query residuals using OSQ, estimates dot products, and compares to exact
     * similarities for the top-5k ranked documents per query.
     */
    @Override
    public double measureError(Candidate candidate, ResidualSample sample, Scratch scratchBuffers) throws IOException {
        OsqCandidate osq = (OsqCandidate) candidate;
        OsqScratch scratch = (OsqScratch) scratchBuffers;
        CalibrationSource source = sample.source();
        boolean usePreconditioned = sample.usePreconditioned();
        int nDocs = sample.nDocs();
        int[] docAssignments = sample.docAssignments();
        float[][] docCentroids = sample.docCentroids();
        float[][] queryCentroids = sample.queryCentroids();
        int[] docCentroidAssignments = sample.docCentroidToQueryCentroid();
        int qbits = osq.qbits();
        int dbits = osq.dbits();
        VectorSimilarityFunction sim = source.similarityFunction();
        int dimWork = source.workingDim();
        boolean cosine = source.cosine();
        int nDocClusters = docCentroids.length;
        int actualQueryClusters = queryCentroids.length;

        // Cluster-count scratch: sized to the actual cluster counts (~nDocs / vectorsPerCluster), not maxNDocs.
        double[] centroidDotCentroid = new double[nDocClusters];
        for (int i = 0; i < nDocClusters; i++) {
            centroidDotCentroid[i] = ESVectorUtil.dotProduct(queryCentroids[docCentroidAssignments[i]], docCentroids[i]);
        }

        OptimizedScalarQuantizer quantizer = new OptimizedScalarQuantizer(VectorSimilarityFunction.EUCLIDEAN);
        float[] residualScratch = scratch.residualScratch;
        float[] normScratch = scratch.normScratch;
        int[] quantizeScratch = scratch.quantizeScratch;
        float[] preconditionScratch = scratch.preconditionScratch;

        float[] docLower = scratch.docLower;
        float[] docUpper = scratch.docUpper;
        int[] docL1 = scratch.docL1;
        byte[][] docQuantized = scratch.docQuantized;
        double[] corpusDotCentroid = scratch.corpusDotCentroid;
        double[] docDotDoc = scratch.docDotDoc;

        for (int i = 0; i < nDocs; i++) {
            float[] doc = source.vectors().vectorValue(source.corpusOrdinals()[i]);
            if (cosine) {
                doc = CalibrationUtils.copyAndNormalize(doc, normScratch);
            }
            if (usePreconditioned && source.preconditioner() != null) {
                source.preconditioner().applyTransform(doc, preconditionScratch);
                doc = preconditionScratch;
            }
            int qc = docCentroidAssignments[docAssignments[i]];
            corpusDotCentroid[i] = ESVectorUtil.dotProduct(queryCentroids[qc], doc);
            var qr = quantizer.scalarQuantize(doc, residualScratch, quantizeScratch, (byte) dbits, docCentroids[docAssignments[i]]);
            ESVectorUtil.packAsBytes(quantizeScratch, docQuantized[i], dimWork);
            docLower[i] = qr.lowerInterval();
            docUpper[i] = qr.upperInterval();
            docL1[i] = qr.quantizedComponentSum();
            if (docDotDoc != null) {
                docDotDoc[i] = ESVectorUtil.dotProduct(doc, doc);
            }
        }

        WelfordVariance moments = new WelfordVariance();
        double dScale = 1.0 / ((1 << dbits) - 1);
        double qScale = 1.0 / ((1 << qbits) - 1);

        float[] queryLower = new float[actualQueryClusters];
        float[] queryUpper = new float[actualQueryClusters];
        int[] queryL1 = new int[actualQueryClusters];
        byte[][] queryQuantized = new byte[actualQueryClusters][dimWork];

        float[] queryScratch = scratch.queryScratch;

        double[] queryDotCentroid = new double[nDocClusters];
        double[] simOsq = scratch.simOsq;
        int[] order = scratch.order;

        // Group doc indices by their query cluster once (assignments are query-independent), so the
        // packed-byte dot product can be evaluated four docs at a time against the shared quantized
        // query operand for that cluster. bucketedDocs holds doc indices sorted by cluster;
        // bucketStart is the CSR offset array with cluster c occupying [bucketStart[c], bucketStart[c+1]).
        int[] bucketStart = scratch.bucketStart;
        int[] bucketedDocs = scratch.bucketedDocs;
        int[] bucketCursor = scratch.bucketCursor;
        int[] intDots = scratch.intDots;
        float[] bulkDistances = scratch.bulkDistances;
        Arrays.fill(bucketStart, 0, actualQueryClusters + 1, 0);
        for (int i = 0; i < nDocs; i++) {
            bucketStart[docCentroidAssignments[docAssignments[i]] + 1]++;
        }
        for (int c = 0; c < actualQueryClusters; c++) {
            bucketStart[c + 1] += bucketStart[c];
        }
        System.arraycopy(bucketStart, 0, bucketCursor, 0, actualQueryClusters);
        for (int i = 0; i < nDocs; i++) {
            int qc = docCentroidAssignments[docAssignments[i]];
            bucketedDocs[bucketCursor[qc]++] = i;
        }

        for (int queryOrdinal : source.queryOrdinals()) {
            CalibrationUtils.materializeCalibrationQuery(
                source.vectors(),
                queryOrdinal,
                source.baseDim(),
                dimWork,
                cosine,
                source.neyshabur(),
                source.preconditioner(),
                usePreconditioned,
                queryScratch,
                preconditionScratch
            );
            for (int qc = 0; qc < actualQueryClusters; qc++) {
                var qr = quantizer.scalarQuantize(queryScratch, residualScratch, quantizeScratch, (byte) qbits, queryCentroids[qc]);
                ESVectorUtil.packAsBytes(quantizeScratch, queryQuantized[qc], dimWork);
                queryLower[qc] = qr.lowerInterval();
                queryUpper[qc] = qr.upperInterval();
                queryL1[qc] = qr.quantizedComponentSum();
            }

            for (int i = 0; i < nDocClusters; i++) {
                queryDotCentroid[i] = ESVectorUtil.dotProduct(queryScratch, docCentroids[i]);
            }

            // Bulk-evaluate the packed-byte doc·query dot products four docs at a time, one query
            // cluster at a time so the shared quantized query operand is reused across the group.
            for (int c = 0; c < actualQueryClusters; c++) {
                byte[] qq = queryQuantized[c];
                int p = bucketStart[c];
                int end = bucketStart[c + 1];
                int bulkLimit = end - 3;
                for (; p < bulkLimit; p += 4) {
                    int d0 = bucketedDocs[p], d1 = bucketedDocs[p + 1], d2 = bucketedDocs[p + 2], d3 = bucketedDocs[p + 3];
                    ESVectorUtil.dotProductBulk(
                        qq,
                        docQuantized[d0],
                        docQuantized[d1],
                        docQuantized[d2],
                        docQuantized[d3],
                        0,
                        bulkDistances
                    );
                    intDots[d0] = Math.round(bulkDistances[0]);
                    intDots[d1] = Math.round(bulkDistances[1]);
                    intDots[d2] = Math.round(bulkDistances[2]);
                    intDots[d3] = Math.round(bulkDistances[3]);
                }
                for (; p < end; p++) {
                    int d = bucketedDocs[p];
                    intDots[d] = Math.round(ESVectorUtil.dotProduct(qq, docQuantized[d]));
                }
            }

            for (int i = 0; i < nDocs; i++) {
                int dc = docAssignments[i];
                int qc = docCentroidAssignments[dc];

                double ad = docLower[i];
                double ld = dScale * (docUpper[i] - docLower[i]);
                double aq = queryLower[qc];
                double lq = qScale * (queryUpper[qc] - queryLower[qc]);

                double dotEst = ad * aq * dimWork + aq * ld * docL1[i] + ad * lq * queryL1[qc] + ld * lq * intDots[i];

                dotEst += corpusDotCentroid[i] + queryDotCentroid[dc] - centroidDotCentroid[dc];

                if (sim == VectorSimilarityFunction.EUCLIDEAN) {
                    dotEst = 2.0 * dotEst;
                    if (docDotDoc != null) {
                        dotEst -= docDotDoc[i];
                    }
                }
                simOsq[i] = dotEst;
            }

            int topN = Math.min(5 * source.k(), nDocs);
            CalibrationUtils.selectTopNDescending(simOsq, order, nDocs, topN);
            for (int i = 0; i < topN; i++) {
                int docIdx = order[i];
                float[] doc = source.vectors().vectorValue(source.corpusOrdinals()[docIdx]);
                if (cosine) {
                    doc = CalibrationUtils.copyAndNormalize(doc, normScratch);
                }
                if (usePreconditioned && source.preconditioner() != null) {
                    source.preconditioner().applyTransform(doc, preconditionScratch);
                    doc = preconditionScratch;
                }
                double exact;
                if (sim == VectorSimilarityFunction.EUCLIDEAN) {
                    assert docDotDoc != null;
                    exact = 2.0 * ESVectorUtil.dotProduct(queryScratch, doc) - docDotDoc[docIdx];
                } else {
                    exact = ESVectorUtil.dotProduct(queryScratch, doc);
                }
                moments.add(exact - simOsq[docIdx]);
            }
        }

        return Math.sqrt(moments.sampleVariance());
    }

    /**
     * Reusable scratch buffers for {@link #measureError}, sized for the maximum
     * sample size of a sweep. Allocated once per sweep via {@link #newScratch} and threaded through each sample-size
     * iteration to avoid repeated large array allocations.
     * Every buffer is fully overwritten before it is read on each call (per the logical
     * lengths {@code nDocs}, {@code actualQueryClusters}, {@code nDocClusters}), so no
     * clearing is needed between invocations and correctness is independent of call order.
     */
    static final class OsqScratch implements Scratch {
        // per-dim
        final float[] residualScratch;
        final float[] queryScratch;
        final int[] quantizeScratch;
        /** null unless cosine normalisation is active */
        final float[] normScratch;
        /** null unless a preconditioner is provided */
        final float[] preconditionScratch;

        // per-doc (sized to maxNDocs)
        final float[] docLower;
        final float[] docUpper;
        final int[] docL1;
        final byte[][] docQuantized;
        final double[] corpusDotCentroid;
        /** null unless the similarity function is EUCLIDEAN */
        final double[] docDotDoc;
        /** per-query-loop: OSQ estimated similarities, indexed [0..nDocs) */
        final double[] simOsq;
        /** per-query-loop: sort permutation over simOsq, indexed [0..nDocs) */
        final int[] order;
        /** per-query-loop: raw packed-byte doc·query dot products, indexed [0..nDocs) */
        final int[] intDots;
        /** doc indices grouped by query cluster, so bulk scoring reuses one query operand */
        final int[] bucketedDocs;
        /** CSR-style start offset of each query cluster's slice in {@link #bucketedDocs} */
        final int[] bucketStart;
        /** fill cursors used while populating {@link #bucketedDocs} */
        final int[] bucketCursor;
        /** 4-lane scratch for {@link ESVectorUtil#dotProductBulk(byte[], byte[], byte[], byte[], byte[], int, float[])} */
        final float[] bulkDistances;

        // Cluster-count arrays (queryLower/Upper/L1/queryQuantized, centroidDotCentroid, queryDotCentroid) are
        // indexed by cluster (nDocClusters / actualQueryClusters ~= nDocs / vectorsPerCluster).
        // they are allocated per call in measureError at the exact cluster count, which is known once k-means has run.

        OsqScratch(int maxNDocs, int dim, boolean cosine, boolean euclidean, boolean hasPreconditioner) {
            residualScratch = new float[dim];
            queryScratch = new float[dim];
            quantizeScratch = new int[dim];
            normScratch = cosine ? new float[dim] : null;
            preconditionScratch = hasPreconditioner ? new float[dim] : null;

            docLower = new float[maxNDocs];
            docUpper = new float[maxNDocs];
            docL1 = new int[maxNDocs];
            docQuantized = new byte[maxNDocs][dim];
            corpusDotCentroid = new double[maxNDocs];
            docDotDoc = euclidean ? new double[maxNDocs] : null;
            simOsq = new double[maxNDocs];
            order = new int[maxNDocs];
            intDots = new int[maxNDocs];
            bucketedDocs = new int[maxNDocs];
            bucketStart = new int[maxNDocs + 1];
            bucketCursor = new int[maxNDocs];
            bulkDistances = new float[4];
        }
    }

}
