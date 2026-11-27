/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.ash;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.elasticsearch.index.codec.vectors.diskbbq.quantize.DocEncoder;
import org.elasticsearch.simdvec.ESVectorUtil;
import org.elasticsearch.test.ESTestCase;

import java.io.IOException;

import static org.hamcrest.Matchers.equalTo;

/**
 * What the generic {@link DocEncoder} hands to a layout writer, the packed code and the corrections, is what the ASH list
 * writer computes from the encoder it uses today: the same bytes, whatever the bits, the dimension and the similarity.
 */
public class AshDocEncoderLayoutTests extends ESTestCase {

    private static final int BLOCK = 32;

    public void testPackedCodesAndLanesAreWhatTheListWriterComputes() throws IOException {
        for (int dim : new int[] { 64, 100 }) {
            for (int bits : new int[] { 1, 2, 4 }) {
                for (VectorSimilarityFunction similarity : new VectorSimilarityFunction[] {
                    VectorSimilarityFunction.DOT_PRODUCT,
                    VectorSimilarityFunction.EUCLIDEAN }) {
                    check(dim, bits, similarity);
                }
            }
        }
    }

    private void check(int dim, int bits, VectorSimilarityFunction similarity) throws IOException {
        String label = "dim=" + dim + " bits=" + bits + " " + similarity;
        AsymmetricHashingQuantizer quantizer = new AsymmetricHashingQuantizer(
            1.0f,
            Integer.MAX_VALUE,
            bits,
            AsymmetricHashingQuantizer.Method.RANDOM,
            0,
            1,
            randomLong(),
            null
        );
        AshProjectionMatrix matrix = quantizer.train(ord -> new float[dim], 1, dim, ord -> new float[dim]).matrix();
        int nDims = matrix.nDims();
        float[] w = ESVectorUtil.transposeMatrix(matrix.wT(), nDims, dim);
        float[][] vectors = new float[BLOCK][dim];
        for (float[] vector : vectors) {
            for (int d = 0; d < dim; d++) {
                vector[d] = (float) random().nextGaussian();
            }
        }
        float[] centroid = new float[dim];
        for (int d = 0; d < dim; d++) {
            centroid[d] = (float) random().nextGaussian() * 0.2f;
        }
        boolean euclidean = similarity == VectorSimilarityFunction.EUCLIDEAN;

        try (
            DocEncoder generic = new AshDocEncoder(matrix, quantizer, w, similarity, bits, BLOCK);
            AsymmetricHashingQuantizer.BlockEncoder production = quantizer.newBlockEncoder(w, dim, BLOCK)
        ) {
            var precomputed = AsymmetricHashingQuantizer.precomputeCentroid(centroid, matrix.wT());
            generic.reset(centroid);
            production.reset();
            for (float[] vector : vectors) {
                generic.add(vector);
                production.add(vector, centroid);
            }
            generic.encode();
            production.encode(precomputed);

            assertThat(label, generic.packedLength(), equalTo(bits * ((nDims + 7) >>> 3)));
            assertThat(label, generic.nativeLaneCount(), equalTo(5));
            float centerOffset = ((1 << bits) - 1) / 2.0f;
            byte[] packed = new byte[generic.packedLength()];
            int[] lanes = new int[generic.nativeLaneCount()];
            for (int j = 0; j < BLOCK; j++) {
                // what AshPostingsListWriter writes for the vector
                float[] code = production.code(j);
                byte[] expectedPacked = ESVectorUtil.ashPack(code, bits);
                int docSum = 0;
                for (int d = 0; d < nDims; d++) {
                    docSum += Math.round(code[d] + centerOffset);
                }

                generic.pack(j, packed, 0);
                assertArrayEquals(label + " code " + j, expectedPacked, packed);
                generic.nativeLanes(j, lanes);
                assertSameFloat(label + " scale " + j, production.scale(j), lanes[0]);
                assertSameFloat(label + " offset " + j, production.offset(j), lanes[1]);
                assertThat(label + " sum " + j, lanes[2], equalTo(docSum));
                assertSameFloat(label + " vecCentroidDot " + j, euclidean ? production.vecCentroidDot(j) : 0f, lanes[3]);
                assertSameFloat(
                    label + " vecCentroidSqDist " + j,
                    euclidean ? ESVectorUtil.squareDistance(vectors[j], centroid) : 0f,
                    lanes[4]
                );
            }
        }
    }

    /**
     * Two quantizations of the same vector at different times can differ in the last bit of a float: the vectorized
     * reductions sum in another order once the JIT has compiled them. Codes are compared exactly, floats to a few ulps.
     */
    private static void assertSameFloat(String label, float expected, int actualBits) {
        float actual = Float.intBitsToFloat(actualBits);
        assertEquals(label, expected, actual, 1e-5f * Math.max(1f, Math.abs(expected)));
    }
}
