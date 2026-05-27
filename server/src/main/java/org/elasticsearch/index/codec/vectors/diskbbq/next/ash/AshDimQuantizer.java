/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq.next.ash;

/**
 * Interface for quantizers used in the ASH projected (latent) space.
 */
public sealed interface AshDimQuantizer permits AshBinaryQuantizer, AshSphericalScalarQuantizer {

    /**
     * Number of bits used per projected dimension.
     */
    int bitsPerDimension();

    /**
     * Encodes a batch of projected vectors.
     *
     * @param x matrix of shape (n, nDims) in the latent space
     * @return centered codes and their norms
     */
    QuantizeResult encode(float[][] x);

    /**
     * Result of quantization.
     *
     * @param centeredCodes codes centered around zero, shape (n, nDims)
     * @param codeNorms L2 norm of each code vector, length n
     */
    record QuantizeResult(float[][] centeredCodes, float[] codeNorms) {}
}
