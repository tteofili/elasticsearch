/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.simdvec.internal.vectorization;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.store.NIOFSDirectory;
import org.apache.lucene.util.VectorUtil;
import org.elasticsearch.simdvec.BaseVectorizationTests;
import org.elasticsearch.simdvec.BulkMask;
import org.elasticsearch.simdvec.E8Lattice;
import org.elasticsearch.simdvec.E8VectorsScorer;

import java.io.IOException;

public class E8VectorsScorerTests extends BaseVectorizationTests {

    /** The 240 roots as float vectors, enumerated from the code bytes. */
    private static float[][] allRoots() {
        float[][] roots = new float[240][8];
        int r = 0;
        for (int sign = 0; sign < 2; sign++) {
            for (int pair = 0; pair < E8Lattice.PAIRS; pair++) {
                for (int t = 0; t < 8; t++) {
                    roots[r][t] = (sign == 1 ? -1 : 1) * E8Lattice.representative(pair, t);
                }
                r++;
            }
        }
        return roots;
    }

    public void testRootsAreTheE8Roots() {
        float[][] roots = allRoots();
        for (float[] root : roots) {
            float norm2 = 0, sum = 0;
            boolean allInt = true, allHalf = true;
            for (float v : root) {
                norm2 += v * v;
                sum += v;
                allInt &= v == Math.rint(v);
                allHalf &= Math.abs(v) == 0.5f;
            }
            assertEquals(2f, norm2, 0f);
            assertTrue(allInt || allHalf);
            assertEquals(0, ((int) Math.rint(sum)) & 1);
        }
        for (int i = 0; i < roots.length; i++) {
            for (int j = i + 1; j < roots.length; j++) {
                assertFalse("duplicate root", java.util.Arrays.equals(roots[i], roots[j]));
            }
        }
    }

    public void testEncodeIsNearestRoot() {
        float[][] roots = allRoots();
        float[] x = new float[8];
        for (int iter = 0; iter < 2000; iter++) {
            for (int t = 0; t < 8; t++) {
                x[t] = (float) random().nextGaussian();
            }
            byte code = E8Lattice.encodeBlock(x, 0, 8);
            int c = code & 0xFF;
            int expected = -1;
            float best = Float.NEGATIVE_INFINITY;
            for (int r = 0; r < roots.length; r++) {
                float d = VectorUtil.dotProduct(x, roots[r]);
                if (d > best) {
                    best = d;
                    expected = r;
                }
            }
            int got = ((c & 0x80) != 0 ? E8Lattice.PAIRS : 0) + (c & 0x7F);
            assertEquals(best, E8Lattice.dotRoot(x, 0, 8, code), 1e-5f);
            assertEquals(expected, got);
        }
    }

    public void testTableMatchesDotProducts() {
        int dims = randomIntBetween(1, 300);
        float[] q = new float[dims];
        for (int i = 0; i < dims; i++) {
            q[i] = (float) random().nextGaussian();
        }
        int blocks = E8Lattice.blocks(dims);
        float[] table = new float[blocks * E8Lattice.TABLE_STRIDE];
        E8Lattice.buildFloatTable(q, dims, table);
        for (int b = 0; b < blocks; b++) {
            int len = Math.min(8, dims - 8 * b);
            for (int pair = 0; pair < E8Lattice.PAIRS; pair++) {
                assertEquals(E8Lattice.dotRoot(q, 8 * b, len, (byte) pair), table[b * E8Lattice.TABLE_STRIDE + pair], 1e-5f);
            }
            for (int pair = E8Lattice.PAIRS; pair < E8Lattice.TABLE_STRIDE; pair++) {
                assertEquals(0f, table[b * E8Lattice.TABLE_STRIDE + pair], 0f);
            }
        }
    }

    /** All kernels must give bit-identical scores, for any bulk size (tails) and block count (incl. the 16-bit flush). */
    public void testKernelsAgree() throws IOException {
        int blocks = randomFrom(1, 7, 48, 96, 128, 192, randomIntBetween(257, 520));
        int count = randomIntBetween(1, 32);
        int numBulks = randomIntBetween(1, 4);
        VectorSimilarityFunction sim = randomFrom(VectorSimilarityFunction.values());
        byte[][] codes = new byte[numBulks * count][blocks];
        float[] alpha = new float[numBulks * count], add = new float[numBulks * count];
        for (int i = 0; i < codes.length; i++) {
            for (int b = 0; b < blocks; b++) {
                codes[i][b] = (byte) ((random().nextBoolean() ? 0x80 : 0) | random().nextInt(E8Lattice.PAIRS));
            }
            alpha[i] = 0.5f + random().nextFloat();
            add[i] = random().nextFloat();
        }
        byte[] table = new byte[blocks * E8Lattice.TABLE_STRIDE];
        for (int i = 0; i < table.length; i++) {
            table[i] = (i % E8Lattice.TABLE_STRIDE) < E8Lattice.PAIRS ? (byte) randomIntBetween(-127, 127) : 0;
        }
        float scale = random().nextFloat() * 0.01f;
        float qAdd = random().nextFloat();
        int mask = random().nextBoolean() ? BulkMask.all(count) : random().nextInt() & BulkMask.all(count);
        try (Directory dir = random().nextBoolean() ? new MMapDirectory(createTempDir()) : new NIOFSDirectory(createTempDir())) {
            try (IndexOutput out = dir.createOutput("e8", IOContext.DEFAULT)) {
                for (int k = 0; k < numBulks; k++) {
                    for (int b = 0; b < blocks; b++) {
                        for (int j = 0; j < count; j++) {
                            out.writeByte(codes[k * count + j][b]);
                        }
                    }
                    for (int j = 0; j < count; j++) {
                        out.writeInt(Float.floatToIntBits(alpha[k * count + j]));
                    }
                    for (int j = 0; j < count; j++) {
                        out.writeInt(0);
                    }
                    for (int j = 0; j < count; j++) {
                        out.writeInt(0);
                    }
                    for (int j = 0; j < count; j++) {
                        out.writeInt(Float.floatToIntBits(add[k * count + j]));
                    }
                }
            }
            float[][] results = new float[E8VectorsScorer.Kernel.values().length][numBulks * count];
            for (E8VectorsScorer.Kernel kernel : E8VectorsScorer.Kernel.values()) {
                try (IndexInput in = dir.openInput("e8", IOContext.DEFAULT)) {
                    E8VectorsScorer scorer = E8VectorsScorer.create(in, blocks, 32, kernel);
                    float[] scores = new float[32];
                    for (int k = 0; k < numBulks; k++) {
                        float max = scorer.scoreBulk(table, scale, qAdd, sim, mask, scores, count);
                        float expectedMax = Float.NEGATIVE_INFINITY;
                        for (int j = 0; j < count; j++) {
                            int i = k * count + j;
                            if (BulkMask.isSelected(mask, j)) {
                                int s = 0;
                                for (int b = 0; b < blocks; b++) {
                                    int c = codes[i][b] & 0xFF;
                                    int x = table[b * E8Lattice.TABLE_STRIDE + (c & 0x7F)];
                                    s += (c & 0x80) != 0 ? -x : x;
                                }
                                float expected = E8VectorsScorer.applyCorrection(alpha[i] * scale * s, qAdd, add[i], sim);
                                assertEquals(kernel + " doc " + i, expected, scores[j], 1e-6f * Math.max(1f, Math.abs(expected)));
                                expectedMax = Math.max(expectedMax, scores[j]);
                            } else {
                                assertEquals(0f, scores[j], 0f);
                            }
                            results[kernel.ordinal()][i] = scores[j];
                        }
                        assertEquals(expectedMax, max, 0f);
                    }
                    assertEquals(in.length(), in.getFilePointer());
                }
            }
        }
    }
}
