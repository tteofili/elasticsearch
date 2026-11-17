/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.simdvec;

import org.elasticsearch.test.ESTestCase;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

public class E8RootsTests extends ESTestCase {

    public void testCodebookIsTheE8Roots() {
        Set<String> codewords = new HashSet<>();
        float[] cw = new float[E8Roots.BLOCK_DIMS];
        for (int code = 0; code < 256; code++) {
            E8Roots.decode(code, cw, 0);
            float norm2 = 0f, sum = 0f;
            int nonZero = 0;
            for (float v : cw) {
                norm2 += v * v;
                sum += v;
                nonZero += v != 0f ? 1 : 0;
            }
            if ((code & 0x7F) >= 120) {
                assertEquals(0f, norm2, 0f);
                continue;
            }
            // twice an E8 root: squared norm 8, all even or all odd coordinates, coordinate sum divisible by 4
            assertEquals(8f, norm2, 0f);
            assertTrue(nonZero == 2 || nonZero == 8);
            assertEquals(0f, sum % 4f, 0f);
            assertTrue(codewords.add(Arrays.toString(cw)));
        }
        assertEquals(240, codewords.size());
    }

    public void testEncodeFindsNearestRoot() {
        float[] cw = new float[E8Roots.BLOCK_DIMS];
        for (int iter = 0; iter < 200; iter++) {
            int dims = randomIntBetween(1, 200);
            float[] x = new float[dims];
            for (int i = 0; i < dims; i++) {
                x[i] = randomBoolean() ? (float) random().nextGaussian() : randomIntBetween(-2, 2); // ties and zeros
            }
            int[] codes = new int[E8Roots.blocks(dims)];
            float dot = E8Roots.encode(x, codes);
            float expectedDot = 0f;
            for (int b = 0; b < codes.length; b++) {
                float best = Float.NEGATIVE_INFINITY;
                for (int code = 0; code < 256; code++) {
                    if ((code & 0x7F) < 120) {
                        E8Roots.decode(code, cw, 0);
                        best = Math.max(best, blockDot(x, b, cw));
                    }
                }
                E8Roots.decode(codes[b], cw, 0);
                float actual = blockDot(x, b, cw);
                assertEquals(best, actual, 1e-4f * (1f + Math.abs(best)));
                expectedDot += actual;
            }
            assertEquals(expectedDot, dot, 1e-3f * (1f + Math.abs(expectedDot)));
        }
    }

    /** The branch free score used to scale query tables must agree with the score of the encoder's nearest codeword. */
    public void testBestScoreMatchesEncoder() {
        for (int iter = 0; iter < 2000; iter++) {
            int len = randomIntBetween(1, E8Roots.BLOCK_DIMS);
            float[] x = new float[E8Roots.BLOCK_DIMS];
            for (int i = 0; i < len; i++) {
                // zeros, negative zeros and repeated magnitudes exercise ties and the parity correction
                x[i] = switch (randomIntBetween(0, 3)) {
                    case 0 -> 0f;
                    case 1 -> -0f;
                    case 2 -> randomBoolean() ? 1f : -1f;
                    case 3 -> (float) random().nextGaussian();
                    default -> throw new AssertionError();
                };
            }
            float expected = Float.intBitsToFloat((int) (E8Roots.encodeBlock(x, 0, len) >>> 32));
            assertEquals(expected, E8Roots.bestScore(x, 0, len), 1e-5f);
        }
    }

    /** The vectorized table agrees with the scalar one up to the rounding of ties, which the two break differently. */
    public void testVectorizedQueryTableMatchesScalar() {
        for (int iter = 0; iter < 50; iter++) {
            int dims = randomIntBetween(1, 1100);
            float[] y = new float[dims];
            for (int i = 0; i < dims; i++) {
                y[i] = (float) random().nextGaussian() * randomFloat();
            }
            int size = E8Roots.blocks(dims) * E8Roots.TABLE_SIZE;
            byte[] expected = new byte[size];
            byte[] actual = new byte[size];
            float expectedScale = E8Roots.buildQueryTable(y, expected);
            float actualScale = ESVectorUtil.buildE8QueryTable(y, actual);
            assertEquals(expectedScale, actualScale, 0f);
            for (int i = 0; i < size; i++) {
                assertEquals("entry " + i, expected[i], actual[i], 1);
            }
        }
        // a zero query gives a zero table
        byte[] table = new byte[E8Roots.TABLE_SIZE];
        Arrays.fill(table, (byte) 7);
        assertEquals(0f, ESVectorUtil.buildE8QueryTable(new float[5], table), 0f);
        for (byte b : table) {
            assertEquals(0, b);
        }
    }

    public void testQueryTable() {
        int dims = randomIntBetween(1, 1000);
        float[] y = new float[dims];
        for (int i = 0; i < dims; i++) {
            y[i] = (float) random().nextGaussian();
        }
        int blocks = E8Roots.blocks(dims);
        byte[] table = new byte[blocks * E8Roots.TABLE_SIZE];
        float scale = E8Roots.buildQueryTable(y, table);
        float[] cw = new float[E8Roots.BLOCK_DIMS];
        for (int b = 0; b < blocks; b++) {
            for (int code = 0; code < E8Roots.TABLE_SIZE; code++) {
                E8Roots.decode(code, cw, 0);
                assertEquals(blockDot(y, b, cw), scale * table[b * E8Roots.TABLE_SIZE + code], 0.5f * scale + 1e-5f);
            }
        }
        // a zero query gives a zero table
        assertEquals(0f, E8Roots.buildQueryTable(new float[dims], table), 0f);
    }

    public void testScoreBulkTransposed() {
        int dims = randomIntBetween(1, 1000);
        int count = randomIntBetween(1, 64);
        int blocks = E8Roots.blocks(dims);
        byte[] table = new byte[blocks * E8Roots.TABLE_SIZE];
        random().nextBytes(table);
        int offset = randomIntBetween(0, 10);
        byte[] codes = new byte[offset + blocks * count];
        random().nextBytes(codes);
        float[] scores = new float[count];
        E8Roots.scoreBulk(codes, offset, table, blocks, count, scores);
        for (int d = 0; d < count; d++) {
            int expected = 0;
            for (int b = 0; b < blocks; b++) {
                int c = codes[offset + b * count + d];
                int v = table[b * E8Roots.TABLE_SIZE + (c & 0x7F)];
                expected += c < 0 ? -v : v;
            }
            assertEquals(expected, scores[d], 0f);
        }
    }

    private static float blockDot(float[] x, int block, float[] cw) {
        float s = 0f;
        for (int k = 0; k < E8Roots.BLOCK_DIMS; k++) {
            int i = block * E8Roots.BLOCK_DIMS + k;
            s += i < x.length ? x[i] * cw[k] : 0f;
        }
        return s;
    }
}
