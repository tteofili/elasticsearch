/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq.quantize;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.elasticsearch.index.codec.vectors.OptimizedScalarQuantizer;
import org.elasticsearch.index.codec.vectors.diskbbq.QuantEncoding;
import org.elasticsearch.test.ESTestCase;

import static org.hamcrest.Matchers.equalTo;

/**
 * What the generic {@link DocEncoder} hands to a layout writer, the packed code and the corrections, is what the OSQ
 * segment writer computes with the quantizer and the packer of its encoding: the same bytes, whatever the bits, the
 * dimension (a dimension that needs padding included) and the similarity.
 */
public class OsqDocEncoderLayoutTests extends ESTestCase {

    private static final int BLOCK = 24;

    public void testPackedCodesAndLanesAreWhatTheSegmentWriterComputes() {
        for (int dim : new int[] { 64, 100 }) {
            for (int bits : new int[] { 1, 2, 4, 7 }) {
                for (VectorSimilarityFunction similarity : new VectorSimilarityFunction[] {
                    VectorSimilarityFunction.DOT_PRODUCT,
                    VectorSimilarityFunction.EUCLIDEAN }) {
                    check(dim, bits, similarity);
                }
            }
        }
    }

    private void check(int dim, int bits, VectorSimilarityFunction similarity) {
        String label = "dim=" + dim + " bits=" + bits + " " + similarity;
        QuantEncoding encoding = QuantEncoding.fromBits((byte) bits);
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

        OptimizedScalarQuantizer reference = new OptimizedScalarQuantizer(similarity);
        int[] quantized = new int[encoding.discretizedDimensions(dim)];
        byte[] expectedPacked = new byte[encoding.getDocPackedLength(dim)];
        float[] scratch = new float[dim];
        try (DocEncoder generic = OsqDocEncoder.forField(similarity, dim, bits, BLOCK)) {
            generic.reset(centroid);
            for (float[] vector : vectors) {
                generic.add(vector);
            }
            generic.encode();

            assertThat(label, generic.packedLength(), equalTo(expectedPacked.length));
            assertThat(label, generic.nativeLaneCount(), equalTo(4));
            byte[] packed = new byte[generic.packedLength() + 3];
            int[] lanes = new int[generic.nativeLaneCount()];
            for (int j = 0; j < BLOCK; j++) {
                // what the segment writer does for the vector
                OptimizedScalarQuantizer.QuantizationResult result = reference.scalarQuantize(
                    vectors[j],
                    scratch,
                    quantized,
                    (byte) bits,
                    centroid
                );
                encoding.pack(quantized, expectedPacked);

                // at an offset, as a layout writer puts the codes of a block one after the other
                generic.pack(j, packed, 3);
                for (int b = 0; b < expectedPacked.length; b++) {
                    assertThat(label + " code " + j + " byte " + b, packed[3 + b], equalTo(expectedPacked[b]));
                }
                generic.nativeLanes(j, lanes);
                assertSameFloat(label + " lower " + j, result.lowerInterval(), lanes[0]);
                assertSameFloat(label + " upper " + j, result.upperInterval(), lanes[1]);
                assertSameFloat(label + " additional " + j, result.additionalCorrection(), lanes[2]);
                assertThat(label + " sum " + j, lanes[3], equalTo(result.quantizedComponentSum()));
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
