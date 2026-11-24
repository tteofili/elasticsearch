/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.ash;

import org.elasticsearch.simdvec.ESVectorUtil;

import java.io.IOException;

/**
 * Exposes, to tests in other packages, the angular error of ASH's cosine quantizer behind a random rotation with
 * {@code d = D}: the codebook of the paper's distortion-rate law.
 */
public final class AsymmetricHashingQuantizerExperimentAccess {
    private final int dim;
    private final AsymmetricHashingQuantizer quantizer;

    public AsymmetricHashingQuantizerExperimentAccess(int dim, int bits) {
        this.dim = dim;
        this.quantizer = new AsymmetricHashingQuantizer(
            1.0f,
            Integer.MAX_VALUE,
            bits,
            AsymmetricHashingQuantizer.Method.RANDOM,
            0,
            1,
            42L,
            null
        );
    }

    /** Mean of {@code 2 - 2 cos(latent, code)} over the vectors' residuals against {@code centroid}. */
    public double meanAngularError(float[][] vecs, float[] centroid) throws IOException {
        var w = quantizer.train(ord -> new float[dim], 1, dim, i -> new float[dim]);
        int nDims = quantizer.nDims(dim);
        var precomputed = AsymmetricHashingQuantizer.precomputeCentroid(centroid, w.wT());
        double sum = 0;
        try (var encoder = quantizer.newBlockEncoder(w.w(), dim, vecs.length)) {
            for (float[] v : vecs) {
                encoder.add(v, centroid);
            }
            encoder.encode(precomputed);
            float[] residual = new float[dim];
            for (int i = 0; i < vecs.length; i++) {
                for (int k = 0; k < dim; k++) {
                    residual[k] = vecs[i][k] - centroid[k];
                }
                float[] latent = ESVectorUtil.matrixVectorMultiply(w.wT(), nDims, dim, residual);
                float[] code = encoder.code(i);
                double dot = ESVectorUtil.dotProduct(latent, code, nDims);
                double nl = Math.sqrt(ESVectorUtil.dotProduct(latent, latent, nDims));
                double nc = Math.sqrt(ESVectorUtil.dotProduct(code, code, nDims));
                sum += 2 - 2 * dot / (nl * nc);
            }
        }
        return sum / vecs.length;
    }
}
