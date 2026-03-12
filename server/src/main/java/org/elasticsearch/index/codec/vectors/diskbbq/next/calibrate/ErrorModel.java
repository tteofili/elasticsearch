/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq.next.calibrate;

import org.apache.lucene.index.VectorSimilarityFunction;

/**
 * Error model for representation (quantization) error in scalar quantization.
 * Estimates the standard deviation of the error in distance/similarity after quantizing
 * queries and documents. Used by calibration to predict recall.
 * <p>
 * This is a stub implementation returning constant-error models; a full implementation
 * would fit scaling/magnitude from actual quantized vs exact distances.
 */
public final class ErrorModel {

    static final int N_QUERY_CLUSTERS = 32;

    private ErrorModel() {}

    /**
     * Estimate scaling of representation error (stub: returns constant-error model).
     */
    public static RepErrorStdModel estimateRepErrorStdScalingParameter(
        VectorSimilarityFunction similarityFunction,
        int dim,
        float[][] queries,
        float[][] corpus,
        int k
    ) {
        Regression.OLSResult c = new Regression.OLSResult(0, 0, 0.1, 0.1, 0, 0.01);
        Regression.OLSResult q = new Regression.OLSResult(0, 0, 0.1, 0.1, 0, 0.01);
        return new RepErrorStdModel(c, q);
    }

    /**
     * Estimate magnitude of representation error for given qbits/dbits (stub: returns scaling model as-is).
     */
    public static RepErrorStdModel estimateRepErrorStdMagnitudeParameter(
        RepErrorStdModel scalingModel,
        VectorSimilarityFunction similarityFunction,
        int dim,
        float[][] queries,
        float[][] corpus,
        int k,
        int qbits,
        int dbits
    ) {
        return scalingModel;
    }
}
