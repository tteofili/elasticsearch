/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.ash;

import org.elasticsearch.index.codec.vectors.diskbbq.IvfSegmentConfig;
import org.elasticsearch.test.ESTestCase;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import static org.hamcrest.Matchers.closeTo;
import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.greaterThanOrEqualTo;

public class E8CodebookTests extends ESTestCase {

    /** All 240 codes decode to distinct E8 lattice points of squared norm 2. */
    public void testCodesCoverTheRoots() {
        List<float[]> roots = allRoots();
        assertThat(roots.size(), equalTo(240));
        Set<String> distinct = new HashSet<>();
        for (float[] r : roots) {
            double normSq = 0, sum = 0;
            boolean half = Math.abs(r[0]) == 0.5f;
            for (float v : r) {
                normSq += v * v;
                sum += v;
                assertThat("all coordinates integer or all half-integer", Math.abs(v) == 0.5f, equalTo(half));
            }
            assertThat(normSq, closeTo(2.0, 1e-9));
            assertThat("coordinate sum is even", Math.abs(sum % 2), closeTo(0.0, 1e-9));
            assertTrue(distinct.add(Arrays.toString(r)));
        }
    }

    /** The O(8) encoder finds the root with the largest inner product, as exhaustive search does. */
    public void testEncodeBlockIsNearest() {
        List<float[]> roots = allRoots();
        float[] x = new float[8];
        float[] out = new float[8];
        float[] decoded = new float[8];
        for (int it = 0; it < 2000; it++) {
            for (int t = 0; t < 8; t++) {
                x[t] = (float) random().nextGaussian();
            }
            if (rarely()) {
                x[random().nextInt(8)] *= 8; // a spiky block
            }
            byte code = E8Codebook.encodeBlock(x, 0, out, 0);
            double best = Double.NEGATIVE_INFINITY;
            for (float[] r : roots) {
                best = Math.max(best, dot(x, r));
            }
            assertThat(dot(x, out), greaterThanOrEqualTo(best - 1e-5));
            E8Codebook.decodeBlock(code, decoded, 0);
            assertArrayEquals(out, decoded, 0f);
        }
    }

    /** Table scoring equals the inner product with the reconstruction, for one and two stages. */
    public void testTableScoreMatchesReconstruction() {
        int nDims = 8 * randomIntBetween(1, 64);
        for (int stages = 1; stages <= 2; stages++) {
            float[] x = randomGaussian(nDims);
            float[] q = randomGaussian(nDims);
            float[] rec = new float[nDims];
            byte[] codes = new byte[E8Codebook.codeBytes(nDims, stages)];
            float norm = E8Codebook.encode(x, 0, nDims, stages, rec, 0, codes);
            assertThat((double) norm, closeTo(Math.sqrt(dot(rec, rec)), 1e-3 * norm));

            // the reconstruction is r1 + beta * r2, rebuilt from the stored codes
            int nBlocks = nDims / 8;
            float[] rebuilt = new float[nDims];
            float[] block = new float[8];
            for (int s = 0; s < stages; s++) {
                float weight = s == 0 ? 1f : E8Codebook.TWO_STAGE_BETA;
                for (int b = 0; b < nBlocks; b++) {
                    E8Codebook.decodeBlock(codes[s * nBlocks + b], block, 0);
                    for (int t = 0; t < 8; t++) {
                        rebuilt[8 * b + t] += weight * block[t];
                    }
                }
            }
            assertArrayEquals(rec, rebuilt, 1e-5f);

            float[] table = E8Codebook.buildTable(q, nDims);
            float score = E8Codebook.score(table, codes, 0, nBlocks, stages);
            double expected = dot(q, rec);
            assertThat((double) score, closeTo(expected, 1e-3 * Math.max(1, Math.abs(expected))));
        }
    }

    /** The mean cosine on random directions matches the codebook constants measured offline (c ≈ 0.353 and 0.106). */
    public void testCodebookConstants() {
        int nDims = 768;
        int n = 500;
        double[] expectedC = { 0.353, 0.106 };
        for (int stages = 1; stages <= 2; stages++) {
            double cosSum = 0;
            for (int i = 0; i < n; i++) {
                float[] x = randomGaussian(nDims);
                float[] rec = new float[nDims];
                float norm = E8Codebook.encode(x, 0, nDims, stages, rec, 0, new byte[E8Codebook.codeBytes(nDims, stages)]);
                cosSum += dot(x, rec) / (Math.sqrt(dot(x, x)) * norm);
            }
            double c = 2 * (1 - cosSum / n);
            assertThat("stages=" + stages, c, closeTo(expectedC[stages - 1], 0.01));
        }
    }

    public void testStoredBitsPerDimRoundTrip() {
        for (int bits : new int[] { 1, 2 }) {
            var e8 = IvfSegmentConfig.AshConfig.of(bits, 4, 0.5f, IvfSegmentConfig.AshConfig.DEFAULT_MAX_PROJECTED_DIMS).withCodebook(IvfSegmentConfig.AshConfig.CODEBOOK_E8);
            assertThat(IvfSegmentConfig.AshConfig.bitsFromStored(e8.storedBitsPerDim()), equalTo(bits));
            int storedCodebook = IvfSegmentConfig.AshConfig.codebookFromStored(e8.storedBitsPerDim());
            assertThat(storedCodebook, equalTo(IvfSegmentConfig.AshConfig.CODEBOOK_E8));
        }
        for (int bits : new int[] { 1, 2, 3, 4, 8 }) {
            var grid = IvfSegmentConfig.AshConfig.of(bits, 4, 0.5f, IvfSegmentConfig.AshConfig.DEFAULT_MAX_PROJECTED_DIMS);
            // segments written with the grid codebook keep their old metadata value
            assertThat(grid.storedBitsPerDim(), equalTo(bits));
            assertThat(IvfSegmentConfig.AshConfig.codebookFromStored(bits), equalTo(IvfSegmentConfig.AshConfig.CODEBOOK_GRID));
        }
        expectThrows(
            IllegalArgumentException.class,
            () -> IvfSegmentConfig.AshConfig.of(4, 4, 0.5f, IvfSegmentConfig.AshConfig.DEFAULT_MAX_PROJECTED_DIMS).withCodebook(IvfSegmentConfig.AshConfig.CODEBOOK_E8)
        );
        expectThrows(IllegalArgumentException.class, () -> E8Codebook.validateDims(12));
    }

    private static List<float[]> allRoots() {
        List<float[]> roots = new ArrayList<>();
        for (int c = 0; c < 256; c++) {
            if ((c & 0x7F) >= E8Codebook.PAIRS) {
                continue;
            }
            float[] r = new float[8];
            E8Codebook.decodeBlock((byte) c, r, 0);
            roots.add(r);
        }
        return roots;
    }

    private static float[] randomGaussian(int n) {
        float[] v = new float[n];
        for (int i = 0; i < n; i++) {
            v[i] = (float) random().nextGaussian();
        }
        return v;
    }

    private static double dot(float[] a, float[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) {
            s += (double) a[i] * b[i];
        }
        return s;
    }

    /** The ASH codebook and the shared E8Lattice (used by the scan kernels and the DiskBBQ encoding) code identically. */
    public void testSameCodesAsE8Lattice() {
        float[] x = new float[8];
        float[] out = new float[8];
        for (int iter = 0; iter < 1000; iter++) {
            for (int t = 0; t < 8; t++) {
                x[t] = (float) random().nextGaussian();
            }
            byte a = E8Codebook.encodeBlock(x, 0, out, 0);
            byte b = org.elasticsearch.simdvec.E8Lattice.encodeBlock(x, 0, 8);
            assertEquals(a, b);
            for (int t = 0; t < 8; t++) {
                int c = a & 0xFF;
                float expected = ((c & 0x80) != 0 ? -1 : 1) * org.elasticsearch.simdvec.E8Lattice.representative(c & 0x7F, t);
                assertEquals(expected, out[t], 0f);
            }
        }
    }
}
