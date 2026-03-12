/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq.next;

import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.elasticsearch.index.codec.vectors.diskbbq.CentroidSupplier;
import org.elasticsearch.index.codec.vectors.diskbbq.next.calibrate.CalibrationUtils;
import org.elasticsearch.index.codec.vectors.diskbbq.next.calibrate.ErrorModel;
import org.elasticsearch.index.codec.vectors.diskbbq.next.calibrate.ExpectedRecall;
import org.elasticsearch.index.codec.vectors.diskbbq.next.calibrate.ManifoldModel;
import org.elasticsearch.index.codec.vectors.diskbbq.next.calibrate.RepErrorStdModel;
import org.elasticsearch.logging.LogManager;
import org.elasticsearch.logging.Logger;

import java.io.IOException;

/**
 * Selects a {@link ESNextDiskBBQVectorsFormat.QuantEncoding} at segment write time by running
 * calibration on a sample of the segment's vectors. The calibration fits a manifold model and
 * an error model, then sweeps candidate encodings in ascending bit-cost order and returns the
 * first one that meets the target recall.
 */
public final class CalibratingAutoQuantizationSelector implements AutoQuantizationSelector {

    private static final Logger logger = LogManager.getLogger(CalibratingAutoQuantizationSelector.class);

    static final double DEFAULT_TARGET_RECALL = 0.9;
    static final int DEFAULT_K = 10;
    static final int MIN_VECTORS_FOR_CALIBRATION = 4096;

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
    private final boolean doPrecondition;
    private final double targetRecall;
    private final int k;

    public CalibratingAutoQuantizationSelector(int vectorsPerCluster, boolean doPrecondition) {
        this(vectorsPerCluster, doPrecondition, DEFAULT_TARGET_RECALL, DEFAULT_K);
    }

    public CalibratingAutoQuantizationSelector(int vectorsPerCluster, boolean doPrecondition, double targetRecall, int k) {
        this.vectorsPerCluster = vectorsPerCluster;
        this.doPrecondition = doPrecondition;
        this.targetRecall = targetRecall;
        this.k = k;
    }

    @Override
    public ESNextDiskBBQVectorsFormat.QuantEncoding select(
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
            return ESNextDiskBBQVectorsFormat.QuantEncoding.FOUR_BIT_SYMMETRIC;
        }

        try {
            return calibrate(floatVectorValues, dim, similarityFunction, N);
        } catch (IOException e) {
            logger.warn("calibration failed, falling back to ONE_BIT_4BIT_QUERY", e);
            return ESNextDiskBBQVectorsFormat.QuantEncoding.ONE_BIT_4BIT_QUERY;
        }
    }

    ESNextDiskBBQVectorsFormat.QuantEncoding calibrate(
        FloatVectorValues floatVectorValues,
        int dim,
        VectorSimilarityFunction similarityFunction,
        int N
    ) throws IOException {
        CalibrationUtils.SampledData sampled = CalibrationUtils.sampleData(floatVectorValues, dim);
        float[][] queries = sampled.queries();
        int[] corpusOrdinals = sampled.corpusOrdinals();

        boolean cosine = similarityFunction == VectorSimilarityFunction.COSINE;
        if (cosine) {
            CalibrationUtils.normalize(queries);
        }

        double[] manifold = ManifoldModel.estimateManifoldParameters(
            similarityFunction,
            dim,
            queries,
            floatVectorValues,
            corpusOrdinals,
            cosine,
            k
        );
        double alpha = manifold[0];
        double invDim = manifold[1];

        RepErrorStdModel errorScalingModel = ErrorModel.estimateRepErrorStdScalingParameter(
            similarityFunction,
            dim,
            queries,
            floatVectorValues,
            corpusOrdinals,
            cosine,
            k
        );

        double maxRecall = -1;
        ESNextDiskBBQVectorsFormat.QuantEncoding bestEncoding = ESNextDiskBBQVectorsFormat.QuantEncoding.ONE_BIT_4BIT_QUERY;

        boolean[] preconditionValues = doPrecondition ? new boolean[] { false, true } : new boolean[] { false };

        for (CandidateEncoding candidate : CANDIDATES) {
            for (boolean precondition : preconditionValues) {
                RepErrorStdModel errorModel = ErrorModel.estimateRepErrorStdMagnitudeParameter(
                    errorScalingModel,
                    similarityFunction,
                    dim,
                    queries,
                    floatVectorValues,
                    corpusOrdinals,
                    cosine,
                    k,
                    candidate.qbits(),
                    candidate.dbits()
                );

                for (int[] rerankRatio : RERANK_RATIOS) {
                    int rerankVal = ExpectedRecall.rerankN(k, rerankRatio[0], rerankRatio[1]);
                    double errorStd = errorModel.quantizeRepErrorStd(vectorsPerCluster, N);
                    double expected = ExpectedRecall.expectedRecallAtK(similarityFunction, N, alpha, invDim, errorStd, k, rerankVal);

                    if (logger.isDebugEnabled()) {
                        logger.debug(
                            "Calibration: encoding [{}] precondition [{}] rerank [{}] -> expected recall [{}]",
                            candidate.encoding(),
                            precondition,
                            rerankVal,
                            expected
                        );
                    }

                    if (expected >= targetRecall) {
                        logger.info(
                            "Calibration selected encoding [{}] (precondition={}, rerank={}) with expected recall [{}]",
                            candidate.encoding(),
                            precondition,
                            rerankVal,
                            expected
                        );
                        return candidate.encoding();
                    }
                    if (expected > maxRecall) {
                        maxRecall = expected;
                        bestEncoding = candidate.encoding();
                    }
                }
            }
        }

        logger.info(
            "Calibration: no encoding met target recall [{}], selecting best [{}] with recall [{}]",
            targetRecall,
            bestEncoding,
            maxRecall
        );
        return bestEncoding;
    }

    private record CandidateEncoding(ESNextDiskBBQVectorsFormat.QuantEncoding encoding, int qbits, int dbits) {}
}
