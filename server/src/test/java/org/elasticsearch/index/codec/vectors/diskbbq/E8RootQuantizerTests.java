/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.VectorUtil;
import org.elasticsearch.index.codec.vectors.OptimizedScalarQuantizer;
import org.elasticsearch.simdvec.E8Roots;
import org.elasticsearch.test.ESTestCase;

public class E8RootQuantizerTests extends ESTestCase {

    public void testCorrectionsMatchOsqSemantics() {
        int dims = randomIntBetween(8, 768);
        VectorSimilarityFunction sim = randomFrom(VectorSimilarityFunction.EUCLIDEAN, VectorSimilarityFunction.DOT_PRODUCT);
        float[] x = randomVector(dims), c = randomVector(dims);
        QuantEncoding encoding = QuantEncoding.E8_ROOTS;
        int[] codes = new int[encoding.discretizedDimensions(dims)];
        float[] residual = new float[dims];
        var e8 = encoding.newQuantizer(sim).scalarQuantize(x, residual, codes, encoding.bits(), c);
        var osq = new OptimizedScalarQuantizer(sim).scalarQuantize(x, new float[dims], new int[dims], (byte) 1, c);
        assertEquals(osq.additionalCorrection(), e8.additionalCorrection(), 1e-3f * Math.max(1f, Math.abs(osq.additionalCorrection())));
        assertEquals(0f, e8.lowerInterval(), 0f);
        // the ratio scale makes the estimate exact along the residual itself
        float[] reconstruction = new float[E8Roots.blocks(dims) * E8Roots.BLOCK_DIMS];
        for (int b = 0; b < E8Roots.blocks(dims); b++) {
            E8Roots.decode(codes[b], reconstruction, b * E8Roots.BLOCK_DIMS);
        }
        float alongResidual = 0f;
        for (int i = 0; i < dims; i++) {
            alongResidual += residual[i] * reconstruction[i];
        }
        float norm2 = VectorUtil.dotProduct(residual, residual);
        assertEquals(norm2, e8.upperInterval() * alongResidual, 1e-3f * norm2);
    }

    public void testEstimateIsCloseToTheInnerProduct() {
        int dims = randomIntBetween(256, 1024);
        float[] c = randomVector(dims), cq = randomVector(dims), q = randomVector(dims);
        QuantEncoding encoding = QuantEncoding.E8_ROOTS;
        E8RootQuantizer quantizer = (E8RootQuantizer) encoding.newQuantizer(VectorSimilarityFunction.DOT_PRODUCT);
        byte[] table = new byte[encoding.getQueryPackedLength(dims)];
        var qc = encoding.quantizeQuery(quantizer, q, new float[dims], null, cq, table);
        double sqErr = 0, sqDot = 0;
        int n = 200;
        for (int iter = 0; iter < n; iter++) {
            // documents scattered around their centroid, as in an IVF posting list
            float[] x = randomVector(dims);
            for (int i = 0; i < dims; i++) {
                x[i] += c[i];
            }
            int[] codes = new int[encoding.discretizedDimensions(dims)];
            float[] residual = new float[dims];
            var dc = quantizer.scalarQuantize(x, residual, codes, encoding.bits(), c);
            byte[] packed = new byte[encoding.getDocPackedLength(dims)];
            encoding.pack(codes, packed);
            float[] raw = new float[1];
            E8Roots.scoreBulk(packed, 0, table, packed.length, 1, raw);
            // the scorers' formula with zero lower intervals: (upper / 255) * alpha * raw
            float estimate = qc.upperInterval() / 255f * dc.upperInterval() * raw[0];
            float exact = 0f;
            float unquantizedTable = 0f;
            float[] reconstruction = new float[packed.length * E8Roots.BLOCK_DIMS];
            for (int b = 0; b < packed.length; b++) {
                E8Roots.decode(packed[b] & 0xFF, reconstruction, b * E8Roots.BLOCK_DIMS);
            }
            for (int i = 0; i < dims; i++) {
                exact += (q[i] - cq[i]) * residual[i];
                unquantizedTable += (q[i] - cq[i]) * reconstruction[i];
            }
            // the int8 table only adds rounding error, at most half a table step per block
            float tableScale = qc.upperInterval() / 255f;
            assertEquals(dc.upperInterval() * unquantizedTable, estimate, dc.upperInterval() * tableScale * 0.5f * packed.length + 1e-3f);
            sqErr += (estimate - exact) * (estimate - exact);
            sqDot += exact * exact;
        }
        // independent Gaussian vectors are the worst case for a one-bit code (about 0.45 relative squared error)
        assertTrue("relative squared error " + sqErr / sqDot, sqErr < 0.8 * sqDot);
    }

    private static float[] randomVector(int dims) {
        float[] v = new float[dims];
        for (int i = 0; i < dims; i++) {
            v[i] = (float) random().nextGaussian();
        }
        return v;
    }
}
