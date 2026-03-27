/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq.next;

import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.elasticsearch.index.codec.vectors.diskbbq.CentroidSupplier;
import org.elasticsearch.index.codec.vectors.diskbbq.Preconditioner;
import org.elasticsearch.index.codec.vectors.diskbbq.next.calibrate.CalibrationQueries;
import org.elasticsearch.index.codec.vectors.diskbbq.next.calibrate.CalibrationUtils;
import org.elasticsearch.index.codec.vectors.diskbbq.next.calibrate.ErrorModel;
import org.elasticsearch.index.codec.vectors.diskbbq.next.calibrate.ExpectedRecall;
import org.elasticsearch.index.codec.vectors.diskbbq.next.calibrate.ManifoldModel;
import org.elasticsearch.index.codec.vectors.diskbbq.next.calibrate.RepErrorStdModel;
import org.elasticsearch.logging.LogManager;
import org.elasticsearch.logging.Logger;
import org.elasticsearch.simdvec.ESVectorUtil;

import java.io.IOException;
import java.util.EnumMap;
import java.util.Map;

/**
 * Selects a {@link ESNextDiskBBQVectorsFormat.QuantEncoding} at segment write time by running
 * calibration on a sample of the segment's vectors. The calibration fits a manifold model and
 * an error model, then sweeps candidate encodings in ascending bit-cost order and returns the
 * first one that meets the target recall.
 */
public class CalibratingAutoQuantizationSelector implements AutoQuantizationSelector {

    private static final Logger logger = LogManager.getLogger(CalibratingAutoQuantizationSelector.class);

    static final double DEFAULT_TARGET_RECALL = 0.97;
    static final int DEFAULT_K = 100;
    static final int MIN_VECTORS_FOR_CALIBRATION = 4096;

    /**
     * If the merged segment is more than this factor larger than the largest input segment,
     * re-run calibration because the OLS models may not extrapolate well.
     */
    static final double RECALIBRATE_GROWTH_RATIO = 4.0;

    /**
     * Maximum per-dimension squared centroid shift before triggering re-calibration.
     * squareDistance(inputGlobalCentroid, mergedGlobalCentroid) / dim must stay below this.
     */
    static final float RECALIBRATE_DRIFT_THRESHOLD = 0.1f;

    /**
     * Minimum fraction of total docs that must agree on a single encoding to skip re-calibration
     * when input segments disagree.
     */
    static final double ENCODING_AGREEMENT_THRESHOLD = 0.8;

    /**
     * Candidate encodings in ascending bit-cost order paired with their (qbits, dbits) for
     * the calibration model. Each entry encodes the ES QuantEncoding and the actual query/doc
     * bits used during calibration recall estimation.
     */
    private static final CandidateEncoding[] CANDIDATES = {
        new CandidateEncoding(ESNextDiskBBQVectorsFormat.QuantEncoding.ONE_BIT_4BIT_QUERY, 4, 1),
        new CandidateEncoding(ESNextDiskBBQVectorsFormat.QuantEncoding.TWO_BIT_4BIT_QUERY, 4, 2),
        new CandidateEncoding(ESNextDiskBBQVectorsFormat.QuantEncoding.FOUR_BIT_SYMMETRIC, 4, 4),
        new CandidateEncoding(ESNextDiskBBQVectorsFormat.QuantEncoding.SEVEN_BIT_SYMMETRIC, 7, 7), };

    private static final int[][] RERANK_RATIOS = { { 15, 10 }, { 2, 1 }, { 3, 1 } };

    private final int vectorsPerCluster;
    private final int blockDimension;
    private final double targetRecall;
    private final int k;

    public CalibratingAutoQuantizationSelector(int vectorsPerCluster) {
        this(vectorsPerCluster, ESNextDiskBBQVectorsFormat.DEFAULT_PRECONDITIONING_BLOCK_DIMENSION);
    }

    public CalibratingAutoQuantizationSelector(int vectorsPerCluster, int blockDimension) {
        this(vectorsPerCluster, blockDimension, DEFAULT_TARGET_RECALL, DEFAULT_K);
    }

    public CalibratingAutoQuantizationSelector(int vectorsPerCluster, int blockDimension, double targetRecall, int k) {
        this.vectorsPerCluster = vectorsPerCluster;
        this.blockDimension = blockDimension;
        this.targetRecall = targetRecall;
        this.k = k;
    }

    /**
     * On flush ({@code mergeState == null}), runs full {@link #calibrate} when the segment is large enough.
     * <p>
     * On merge, attempts to reuse quantization metadata from input segments via {@link #selectFromMergeState},
     * except for <em>bounded</em> (force-merge) merges: those run {@link #runFastCalibration} first so calibration
     * is not skipped after major segment consolidation; if the fast path does not meet {@link #targetRecall},
     * full {@link #calibrate} runs once on the same vectors (fast-then-full fallback) when the fast path does not
     * meet the configured target recall. Bounded merges are detected
     * from the merged segment's Lucene diagnostics key {@code mergeMaxNumSegments} ({@code >= 1}).
     */
    @Override
    public CalibrationResult select(
        FieldInfo fieldInfo,
        FloatVectorValues floatVectorValues,
        CentroidSupplier centroidSupplier,
        int[] assignments,
        int[] overspillAssignments,
        MergeState mergeState
    ) {
        int dim = fieldInfo.getVectorDimension();
        VectorSimilarityFunction similarityFunction = fieldInfo.getVectorSimilarityFunction();
        int N = floatVectorValues.size();

        if (N < MIN_VECTORS_FOR_CALIBRATION) {
            return new CalibrationResult(ESNextDiskBBQVectorsFormat.QuantEncoding.FOUR_BIT_SYMMETRIC, DEFAULT_CALIBRATED_OVERSAMPLE, false);
        }

        if (mergeState != null) {
            MergeCalibrationContext mergeCtx = MergeCalibrationContext.from(mergeState);
            if (mergeCtx.boundedForceMerge()) {
                logger.info(
                    "Merge calibration: bounded force merge (mergeMaxNumSegments=[{}], inputSegments=[{}]), skipping metadata reuse; running fast calibration",
                    mergeCtx.mergeMaxNumSegmentsForLog(),
                    mergeCtx.inputSegments()
                );
                try {
                    FastCalibrationOutcome fastOutcome = runFastCalibration(floatVectorValues, dim, similarityFunction, N, mergeCtx);
                    if (fastOutcome.metTargetRecall()) {
                        return fastOutcome.result();
                    }
                    logger.info(
                        "Merge calibration: fast path did not meet target recall [{}], running full calibration [inputSegments={} mergeKind={} mergeMaxNumSegments={}]",
                        targetRecall,
                        mergeCtx.inputSegments(),
                        mergeCtx.mergeKind(),
                        mergeCtx.mergeMaxNumSegmentsForLog()
                    );
                    try {
                        return calibrate(floatVectorValues, dim, similarityFunction, N);
                    } catch (IOException e) {
                        logger.warn("full calibration failed after bounded-merge fast miss, falling back to ONE_BIT_4BIT_QUERY", e);
                        return new CalibrationResult(
                            ESNextDiskBBQVectorsFormat.QuantEncoding.ONE_BIT_4BIT_QUERY,
                            NO_CALIBRATED_OVERSAMPLE,
                            false
                        );
                    }
                } catch (IOException e) {
                    logger.warn("fast calibration failed, falling back to ONE_BIT_4BIT_QUERY", e);
                    return new CalibrationResult(
                        ESNextDiskBBQVectorsFormat.QuantEncoding.ONE_BIT_4BIT_QUERY,
                        NO_CALIBRATED_OVERSAMPLE,
                        false
                    );
                }
            }
            CalibrationResult reused = selectFromMergeState(fieldInfo, floatVectorValues, centroidSupplier, mergeState, mergeCtx);
            if (reused != null) {
                return reused;
            }
            logger.debug("Merge calibration reuse not possible, running fast calibration");
            try {
                return calibrateFast(floatVectorValues, dim, similarityFunction, N, mergeCtx);
            } catch (IOException e) {
                logger.warn("fast calibration failed, falling back to ONE_BIT_4BIT_QUERY", e);
                return new CalibrationResult(ESNextDiskBBQVectorsFormat.QuantEncoding.ONE_BIT_4BIT_QUERY, NO_CALIBRATED_OVERSAMPLE, false);
            }
        }

        try {
            return calibrate(floatVectorValues, dim, similarityFunction, N);
        } catch (IOException e) {
            logger.warn("calibration failed, falling back to ONE_BIT_4BIT_QUERY", e);
            return new CalibrationResult(ESNextDiskBBQVectorsFormat.QuantEncoding.ONE_BIT_4BIT_QUERY, NO_CALIBRATED_OVERSAMPLE, false);
        }
    }

    /**
     * Attempts to reuse calibration results from the input segments being merged.
     * Returns a merged {@link CalibrationResult} if the data has not changed significantly,
     * or {@code null} if merge-time fast calibration should be performed.
     * Not used for bounded (force-merge) merges; those use {@link #runFastCalibration} with a full
     * {@link #calibrate} fallback when the fast path does not meet the target recall.
     */
    CalibrationResult selectFromMergeState(
        FieldInfo fieldInfo,
        FloatVectorValues mergedVectors,
        CentroidSupplier centroidSupplier,
        MergeState mergeState,
        MergeCalibrationContext mergeCtx
    ) {
        int dim = fieldInfo.getVectorDimension();
        Map<ESNextDiskBBQVectorsFormat.QuantEncoding, Long> encodingDocCounts = new EnumMap<>(
            ESNextDiskBBQVectorsFormat.QuantEncoding.class
        );
        double oversampleWeightedSum = 0;
        long totalDocs = 0;
        long largestSegmentDocs = 0;
        long preconditionTrueDocs = 0;
        long preconditionFalseDocs = 0;
        int calibratedSegments = 0;

        float[] mergedGlobalCentroid = null;
        try {
            if (centroidSupplier != null && centroidSupplier.size() > 0) {
                mergedGlobalCentroid = centroidSupplier.centroid(0);
            }
        } catch (IOException e) {
            logger.debug("could not read merged global centroid for drift detection", e);
        }

        for (int i = 0; i < mergeState.knnVectorsReaders.length; i++) {
            KnnVectorsReader reader = mergeState.knnVectorsReaders[i];
            if (reader instanceof CalibrationAwareReader car) {
                ESNextDiskBBQVectorsFormat.QuantEncoding enc = car.getQuantEncoding(fieldInfo);
                if (enc == null) {
                    continue;
                }
                long docs = mergeState.maxDocs[i];
                calibratedSegments++;
                encodingDocCounts.merge(enc, docs, Long::sum);
                float oversample = car.getOversampleFactor(fieldInfo);
                oversampleWeightedSum += oversample * docs;
                totalDocs += docs;
                largestSegmentDocs = Math.max(largestSegmentDocs, docs);

                if (car.shouldPrecondition(fieldInfo)) {
                    preconditionTrueDocs += docs;
                } else {
                    preconditionFalseDocs += docs;
                }

                if (mergedGlobalCentroid != null) {
                    float[] segmentCentroid = car.getGlobalCentroid(fieldInfo);
                    if (segmentCentroid != null) {
                        float drift = ESVectorUtil.squareDistance(segmentCentroid, mergedGlobalCentroid) / dim;
                        if (drift > RECALIBRATE_DRIFT_THRESHOLD) {
                            logger.info(
                                "Merge calibration: centroid drift [{}] exceeds threshold [{}] for segment [{}], re-calibrating [inputSegments={} mergeKind={} mergeMaxNumSegments={}]",
                                drift,
                                RECALIBRATE_DRIFT_THRESHOLD,
                                i,
                                mergeCtx.inputSegments(),
                                mergeCtx.mergeKind(),
                                mergeCtx.mergeMaxNumSegmentsForLog()
                            );
                            return null;
                        }
                    }
                }
            }
        }

        if (calibratedSegments == 0) {
            return null;
        }

        long mergedSize = mergedVectors.size();
        if (mergedSize > RECALIBRATE_GROWTH_RATIO * largestSegmentDocs) {
            logger.info(
                "Merge calibration: growth ratio [{}] exceeds threshold [{}], re-calibrating [inputSegments={} mergeKind={} mergeMaxNumSegments={}]",
                (double) mergedSize / largestSegmentDocs,
                RECALIBRATE_GROWTH_RATIO,
                mergeCtx.inputSegments(),
                mergeCtx.mergeKind(),
                mergeCtx.mergeMaxNumSegmentsForLog()
            );
            return null;
        }

        if (encodingDocCounts.size() > 1) {
            long maxEncDocs = encodingDocCounts.values().stream().mapToLong(Long::longValue).max().orElse(0);
            if (maxEncDocs < ENCODING_AGREEMENT_THRESHOLD * totalDocs) {
                logger.info(
                    "Merge calibration: encoding disagreement (max encoding covers [{}]% of docs), re-calibrating [inputSegments={} mergeKind={} mergeMaxNumSegments={}]",
                    (100.0 * maxEncDocs / totalDocs),
                    mergeCtx.inputSegments(),
                    mergeCtx.mergeKind(),
                    mergeCtx.mergeMaxNumSegmentsForLog()
                );
                return null;
            }
        }

        ESNextDiskBBQVectorsFormat.QuantEncoding bestEncoding = encodingDocCounts.entrySet()
            .stream()
            .max(Map.Entry.comparingByValue())
            .get()
            .getKey();
        float avgOversample = (float) (oversampleWeightedSum / totalDocs);
        boolean doPreconditionResult = preconditionTrueDocs > preconditionFalseDocs;

        logger.info(
            "Merge calibration: reusing encoding [{}] (oversample={}, precondition={}) from [{}] input segments [inputSegments={} mergeKind={} mergeMaxNumSegments={}]",
            bestEncoding,
            avgOversample,
            doPreconditionResult,
            calibratedSegments,
            mergeCtx.inputSegments(),
            mergeCtx.mergeKind(),
            mergeCtx.mergeMaxNumSegmentsForLog()
        );
        return new CalibrationResult(bestEncoding, avgOversample, doPreconditionResult);
    }

    CalibrationResult calibrate(FloatVectorValues floatVectorValues, int dim, VectorSimilarityFunction similarityFunction, int N)
        throws IOException {
        CalibrationUtils.SampledData sampled = CalibrationUtils.sampleData(floatVectorValues, dim);
        int[] queryOrdinals = sampled.queryOrdinals();
        int[] corpusOrdinals = sampled.corpusOrdinals();

        boolean cosine = similarityFunction == VectorSimilarityFunction.COSINE;
        boolean neyshabur = CalibrationUtils.needsNeyshaburSrebroLift(similarityFunction);

        int dimWork = dim;
        FloatVectorValues fvvForCalibration = floatVectorValues;
        if (neyshabur) {
            double maxNormSq = CalibrationUtils.maxSquaredNormOverCorpusSample(floatVectorValues, corpusOrdinals, dim);
            fvvForCalibration = new CalibrationUtils.NeyshaburCorpusFloatVectorValues(floatVectorValues, dim, maxNormSq);
            dimWork = dim + 1;
        }

        Preconditioner calibrationPreconditioner = Preconditioner.createPreconditioner(dimWork, blockDimension);
        CalibrationQueries calibrationQueries = new CalibrationQueries(
            floatVectorValues,
            queryOrdinals,
            dim,
            cosine,
            neyshabur,
            calibrationPreconditioner,
            dimWork
        );

        // Manifold model uses original (un-preconditioned) data; after optional Neyshabur lift for dot/MIP.
        double[] manifold = ManifoldModel.estimateManifoldParameters(
            similarityFunction,
            dimWork,
            calibrationQueries,
            fvvForCalibration,
            corpusOrdinals,
            cosine,
            k
        );
        double alpha = manifold[0];
        double invDim = manifold[1];

        // Reference auto_osq: always fit the error scaling model on random orthogonal transforms,
        // independent of whether the field enables preconditioning at index time.
        FloatVectorValues fvvOrth = preconditionFvv(fvvForCalibration, calibrationPreconditioner);

        RepErrorStdModel errorScalingModel = ErrorModel.estimateRepErrorStdScalingParameter(
            similarityFunction,
            dimWork,
            calibrationQueries,
            fvvOrth,
            corpusOrdinals,
            cosine,
            k
        );

        double maxRecall = -1;
        ESNextDiskBBQVectorsFormat.QuantEncoding bestEncoding = ESNextDiskBBQVectorsFormat.QuantEncoding.ONE_BIT_4BIT_QUERY;
        float bestOversample = AutoQuantizationSelector.NO_CALIBRATED_OVERSAMPLE;
        boolean bestPrecondition = false;

        boolean[] preconditionValues = new boolean[] { false, true };

        for (CandidateEncoding candidate : CANDIDATES) {
            for (boolean precondition : preconditionValues) {
                FloatVectorValues magnitudeFvv = precondition ? fvvOrth : fvvForCalibration;

                RepErrorStdModel errorModel = ErrorModel.estimateRepErrorStdMagnitudeParameter(
                    errorScalingModel,
                    similarityFunction,
                    dimWork,
                    calibrationQueries,
                    precondition,
                    magnitudeFvv,
                    corpusOrdinals,
                    cosine,
                    k,
                    candidate.qbits(),
                    candidate.dbits()
                );

                for (int[] rerankRatio : RERANK_RATIOS) {
                    int rerankVal = ExpectedRecall.rerankN(k, rerankRatio[0], rerankRatio[1]);
                    float oversample = (float) rerankRatio[0] / rerankRatio[1];
                    double errorStd = errorModel.quantizeRepErrorStd(vectorsPerCluster, N);
                    double expected = ExpectedRecall.expectedRecallAtK(similarityFunction, N, alpha, invDim, errorStd, k, rerankVal);

                    if (logger.isDebugEnabled()) {
                        logger.debug(
                            "Calibration: encoding [{}] precondition [{}] rerank [{}] oversample [{}] -> expected recall [{}]",
                            candidate.encoding(),
                            precondition,
                            rerankVal,
                            oversample,
                            expected
                        );
                    }

                    if (expected >= targetRecall) {
                        logger.info(
                            "Calibration selected encoding [{}] (precondition={}, rerank={}, oversample={}) with expected recall [{}]",
                            candidate.encoding(),
                            precondition,
                            rerankVal,
                            oversample,
                            expected
                        );
                        return new CalibrationResult(candidate.encoding(), oversample, precondition);
                    }
                    if (expected > maxRecall) {
                        maxRecall = expected;
                        bestEncoding = candidate.encoding();
                        bestOversample = oversample;
                        bestPrecondition = precondition;
                    }
                }
            }
        }

        logger.info(
            "Calibration: no encoding met target recall [{}], selecting best [{}] with oversample [{}] precondition [{}] and recall [{}]",
            targetRecall,
            bestEncoding,
            bestOversample,
            bestPrecondition,
            maxRecall
        );
        return new CalibrationResult(bestEncoding, bestOversample, bestPrecondition);
    }

    /**
     * Outcome of {@link #runFastCalibration}: {@code metTargetRecall} is true when an encoding met the
     * target recall before the best-effort path.
     */
    protected record FastCalibrationOutcome(CalibrationResult result, boolean metTargetRecall) {}

    /**
     * Runs calibration with reduced sample sizes, fewer sweep iterations, and fewer
     * manifold model data points for faster execution during merge re-calibration.
     */
    CalibrationResult calibrateFast(
        FloatVectorValues floatVectorValues,
        int dim,
        VectorSimilarityFunction similarityFunction,
        int N,
        MergeCalibrationContext mergeCtx
    ) throws IOException {
        return runFastCalibration(floatVectorValues, dim, similarityFunction, N, mergeCtx).result();
    }

    /**
     * Same work as {@link #calibrateFast} but exposes whether the target recall was reached (for bounded-merge fallback).
     */
    protected FastCalibrationOutcome runFastCalibration(
        FloatVectorValues floatVectorValues,
        int dim,
        VectorSimilarityFunction similarityFunction,
        int N,
        MergeCalibrationContext mergeCtx
    ) throws IOException {
        CalibrationUtils.SampledData sampled = CalibrationUtils.sampleDataFast(floatVectorValues, dim);
        int[] queryOrdinals = sampled.queryOrdinals();
        int[] corpusOrdinals = sampled.corpusOrdinals();

        boolean cosine = similarityFunction == VectorSimilarityFunction.COSINE;
        boolean neyshabur = CalibrationUtils.needsNeyshaburSrebroLift(similarityFunction);

        int dimWork = dim;
        FloatVectorValues fvvForCalibration = floatVectorValues;
        if (neyshabur) {
            double maxNormSq = CalibrationUtils.maxSquaredNormOverCorpusSample(floatVectorValues, corpusOrdinals, dim);
            fvvForCalibration = new CalibrationUtils.NeyshaburCorpusFloatVectorValues(floatVectorValues, dim, maxNormSq);
            dimWork = dim + 1;
        }

        Preconditioner calibrationPreconditioner = Preconditioner.createPreconditioner(dimWork, blockDimension);
        CalibrationQueries calibrationQueries = new CalibrationQueries(
            floatVectorValues,
            queryOrdinals,
            dim,
            cosine,
            neyshabur,
            calibrationPreconditioner,
            dimWork
        );

        double[] manifold = ManifoldModel.estimateManifoldParametersFast(
            similarityFunction,
            dimWork,
            calibrationQueries,
            fvvForCalibration,
            corpusOrdinals,
            cosine,
            k
        );
        double alpha = manifold[0];
        double invDim = manifold[1];

        FloatVectorValues fvvOrth = preconditionFvv(fvvForCalibration, calibrationPreconditioner);

        RepErrorStdModel errorScalingModel = ErrorModel.estimateRepErrorStdScalingParameterFast(
            similarityFunction,
            dimWork,
            calibrationQueries,
            fvvOrth,
            corpusOrdinals,
            cosine,
            k
        );

        double maxRecall = -1;
        ESNextDiskBBQVectorsFormat.QuantEncoding bestEncoding = ESNextDiskBBQVectorsFormat.QuantEncoding.ONE_BIT_4BIT_QUERY;
        float bestOversample = AutoQuantizationSelector.NO_CALIBRATED_OVERSAMPLE;
        boolean bestPrecondition = false;

        boolean[] preconditionValues = new boolean[] { false, true };

        for (CandidateEncoding candidate : CANDIDATES) {
            for (boolean precondition : preconditionValues) {
                FloatVectorValues magnitudeFvv = precondition ? fvvOrth : fvvForCalibration;

                RepErrorStdModel errorModel = ErrorModel.estimateRepErrorStdMagnitudeParameterFast(
                    errorScalingModel,
                    similarityFunction,
                    dimWork,
                    calibrationQueries,
                    precondition,
                    magnitudeFvv,
                    corpusOrdinals,
                    cosine,
                    k,
                    candidate.qbits(),
                    candidate.dbits()
                );

                for (int[] rerankRatio : RERANK_RATIOS) {
                    int rerankVal = ExpectedRecall.rerankN(k, rerankRatio[0], rerankRatio[1]);
                    float oversample = (float) rerankRatio[0] / rerankRatio[1];
                    double errorStd = errorModel.quantizeRepErrorStd(vectorsPerCluster, N);
                    double expected = ExpectedRecall.expectedRecallAtK(similarityFunction, N, alpha, invDim, errorStd, k, rerankVal);

                    if (logger.isDebugEnabled()) {
                        logger.debug(
                            "Fast calibration: encoding [{}] precondition [{}] rerank [{}] oversample [{}] -> expected recall [{}]",
                            candidate.encoding(),
                            precondition,
                            rerankVal,
                            oversample,
                            expected
                        );
                    }

                    if (expected >= targetRecall) {
                        if (mergeCtx != null) {
                            logger.info(
                                "Fast calibration selected encoding [{}] (precondition={}, rerank={}, oversample={}) with expected recall [{}] [inputSegments={} mergeKind={} mergeMaxNumSegments={}]",
                                candidate.encoding(),
                                precondition,
                                rerankVal,
                                oversample,
                                expected,
                                mergeCtx.inputSegments(),
                                mergeCtx.mergeKind(),
                                mergeCtx.mergeMaxNumSegmentsForLog()
                            );
                        } else {
                            logger.info(
                                "Fast calibration selected encoding [{}] (precondition={}, rerank={}, oversample={}) with expected recall [{}]",
                                candidate.encoding(),
                                precondition,
                                rerankVal,
                                oversample,
                                expected
                            );
                        }
                        return new FastCalibrationOutcome(new CalibrationResult(candidate.encoding(), oversample, precondition), true);
                    }
                    if (expected > maxRecall) {
                        maxRecall = expected;
                        bestEncoding = candidate.encoding();
                        bestOversample = oversample;
                        bestPrecondition = precondition;
                    }
                }
            }
        }

        if (mergeCtx != null) {
            logger.info(
                "Fast calibration: no encoding met target recall [{}], selecting best [{}] oversample [{}] precondition [{}] recall [{}] [inputSegments={} mergeKind={} mergeMaxNumSegments={}]",
                targetRecall,
                bestEncoding,
                bestOversample,
                bestPrecondition,
                maxRecall,
                mergeCtx.inputSegments(),
                mergeCtx.mergeKind(),
                mergeCtx.mergeMaxNumSegmentsForLog()
            );
        } else {
            logger.info(
                "Fast calibration: no encoding met target recall [{}], selecting best [{}] oversample [{}] precondition [{}] recall [{}]",
                targetRecall,
                bestEncoding,
                bestOversample,
                bestPrecondition,
                maxRecall
            );
        }
        return new FastCalibrationOutcome(new CalibrationResult(bestEncoding, bestOversample, bestPrecondition), false);
    }

    private static FloatVectorValues preconditionFvv(FloatVectorValues fvv, Preconditioner preconditioner) {
        return new FloatVectorValues() {
            final float[] preconditioned = new float[fvv.dimension()];
            int cachedOrd = -1;

            @Override
            public float[] vectorValue(int ord) throws IOException {
                if (ord != cachedOrd) {
                    float[] raw = fvv.vectorValue(ord);
                    preconditioner.applyTransform(raw, preconditioned);
                    cachedOrd = ord;
                }
                return preconditioned;
            }

            @Override
            public FloatVectorValues copy() throws IOException {
                return fvv.copy();
            }

            @Override
            public int dimension() {
                return fvv.dimension();
            }

            @Override
            public int size() {
                return fvv.size();
            }

            @Override
            public DocIndexIterator iterator() {
                return fvv.iterator();
            }
        };
    }

    private record CandidateEncoding(ESNextDiskBBQVectorsFormat.QuantEncoding encoding, int qbits, int dbits) {}
}
