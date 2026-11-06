/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.ash;

/**
 * E8 block code for ASH (experimental).
 * <p>
 * The projected vector is split into blocks of eight coordinates and each block is replaced by the nearest of the 240
 * minimal vectors (roots) of the E8 lattice: 112 "spiky" roots {@code (±1, ±1, 0^6)} and 128 "flat" roots
 * {@code (±1/2)^8} with an even number of minus signs. All roots have norm √2, so the nearest root is the one with the
 * largest inner product, which is found exactly in O(8) without searching the codebook.
 * <p>
 * Roots come in ± pairs, so a root is stored in one byte as {@code sign << 7 | pair}, with {@code pair} in [0, 120):
 * <ul>
 *   <li>{@code pair < 56}: spiky, {@code pair = 2 * p + k}, where p indexes the position pair {@code (i < j)} and the
 *       representative is {@code e_i + e_j} for {@code k = 0} or {@code e_i - e_j} for {@code k = 1};</li>
 *   <li>{@code pair >= 56}: flat, representative {@code (1/2) * (+1, s_1, ..., s_7)}, whose minus signs (a 7-bit mask
 *       with an even number of set bits) are the {@code (pair - 56)}-th such mask in increasing order.</li>
 * </ul>
 * With two stages (two bits per dimension), the first stage codes the block as above; the vector is then fitted with a
 * least-squares gain on its first-stage reconstruction, and the second stage codes what is left. The reconstruction is
 * {@code r1 + beta * r2}, and the codes are stored stage by stage: all first-stage bytes, then all second-stage bytes.
 * This keeps the storage of the per-coordinate grid: {@code stages * nDims / 8} bytes per vector.
 */
public final class E8Codebook {

    /** Number of ± pairs of roots, and so the number of table entries per block. */
    public static final int PAIRS = 120;
    /** Weight of the second stage in the reconstruction; fitted on several text-embedding datasets. */
    public static final float TWO_STAGE_BETA = 0.55f;

    private static final int SPIKY_PAIRS = 56;
    private static final int SIGN_BIT = 0x80;

    /** Canonical representative of each pair, row-major (PAIRS x 8). */
    static final float[] REPS = new float[PAIRS * 8];
    /** Position pair (i, j), i lt j, packed as i * 8 + j, to its index p in [0, 28). */
    private static final int[] POSITION_PAIR = new int[64];
    /** 7-bit minus-sign mask of positions 1..7 to its flat index in [0, 64), or -1 if the mask has odd parity. */
    private static final int[] FLAT_MASK = new int[128];

    static {
        int p = 0;
        for (int i = 0; i < 8; i++) {
            for (int j = i + 1; j < 8; j++) {
                POSITION_PAIR[i * 8 + j] = p;
                for (int k = 0; k < 2; k++) {
                    int pair = 2 * p + k;
                    REPS[pair * 8 + i] = 1f;
                    REPS[pair * 8 + j] = k == 0 ? 1f : -1f;
                }
                p++;
            }
        }
        assert p == 28;
        int m = 0;
        for (int mask = 0; mask < 128; mask++) {
            if ((Integer.bitCount(mask) & 1) == 0) {
                FLAT_MASK[mask] = m;
                int pair = SPIKY_PAIRS + m;
                REPS[pair * 8] = 0.5f;
                for (int t = 1; t < 8; t++) {
                    REPS[pair * 8 + t] = ((mask >>> (t - 1)) & 1) != 0 ? -0.5f : 0.5f;
                }
                m++;
            } else {
                FLAT_MASK[mask] = -1;
            }
        }
        assert m == 64;
    }

    private E8Codebook() {}

    /**
     * Validates that a projected dimension can be coded by E8 blocks.
     *
     * @throws IllegalArgumentException if {@code nDims} is not a positive multiple of 8
     */
    public static void validateDims(int nDims) {
        if (nDims <= 0 || (nDims & 7) != 0) {
            throw new IllegalArgumentException("ASH E8 codebook needs a projected dimension that is a multiple of 8, got: " + nDims);
        }
    }

    /** Bytes per vector for {@code stages} stages over {@code nDims} projected dimensions. */
    public static int codeBytes(int nDims, int stages) {
        return stages * (nDims >>> 3);
    }

    /**
     * Codes one block {@code x[off..off+8)} with its nearest root, writes the root into {@code out[outOff..outOff+8)}
     * and returns the code byte.
     */
    static byte encodeBlock(float[] x, int off, float[] out, int outOff) {
        // Best spiky root: the two largest magnitudes, with their signs
        int i1 = 0, i2 = -1;
        float a1 = Math.abs(x[off]), a2 = -1f;
        for (int t = 1; t < 8; t++) {
            float a = Math.abs(x[off + t]);
            if (a > a1) {
                i2 = i1;
                a2 = a1;
                i1 = t;
                a1 = a;
            } else if (a > a2) {
                i2 = t;
                a2 = a;
            }
        }
        float spikyScore = a1 + a2;

        // Best flat root: the signs of x, with the smallest magnitude flipped if the number of minus signs is odd
        int minusMask = 0;
        int smallest = 0;
        float smallestAbs = Float.MAX_VALUE;
        float flatScore = 0f;
        for (int t = 0; t < 8; t++) {
            float v = x[off + t];
            float a = Math.abs(v);
            flatScore += a;
            if (v < 0) {
                minusMask |= 1 << t;
            }
            if (a < smallestAbs) {
                smallestAbs = a;
                smallest = t;
            }
        }
        if ((Integer.bitCount(minusMask) & 1) != 0) {
            minusMask ^= 1 << smallest;
            flatScore -= 2 * smallestAbs;
        }
        flatScore *= 0.5f;

        int code;
        if (spikyScore >= flatScore) {
            int lo = Math.min(i1, i2), hi = Math.max(i1, i2);
            boolean loNeg = x[off + lo] < 0;
            boolean hiNeg = x[off + hi] < 0;
            // representative has +1 at lo; the root is that representative times the sign at lo
            int k = loNeg == hiNeg ? 0 : 1;
            code = 2 * POSITION_PAIR[lo * 8 + hi] + k;
            if (loNeg) {
                code |= SIGN_BIT;
            }
        } else {
            // representative has +1/2 at position 0; the root is that representative times the sign at position 0
            boolean neg = (minusMask & 1) != 0;
            int repMask = neg ? (~minusMask & 0xFF) : minusMask;
            code = SPIKY_PAIRS + FLAT_MASK[(repMask >>> 1) & 0x7F];
            if (neg) {
                code |= SIGN_BIT;
            }
        }
        decodeBlock((byte) code, out, outOff);
        return (byte) code;
    }

    /** Writes the root coded by {@code code} into {@code out[off..off+8)}. */
    static void decodeBlock(byte code, float[] out, int off) {
        int c = code & 0xFF;
        int pair = c & 0x7F;
        float sign = (c & SIGN_BIT) != 0 ? -1f : 1f;
        for (int t = 0; t < 8; t++) {
            out[off + t] = sign * REPS[pair * 8 + t];
        }
    }

    /** Thread-local scratch for the two-stage encoder: the residual and the second-stage roots. */
    private static final ThreadLocal<float[][]> SCRATCH = new ThreadLocal<>();

    private static float[][] scratch(int nDims) {
        float[][] s = SCRATCH.get();
        if (s == null || s[0].length != nDims) {
            s = new float[][] { new float[nDims], new float[nDims] };
            SCRATCH.set(s);
        }
        return s;
    }

    /**
     * Codes the vector {@code x[xOff..xOff+nDims)} with {@code stages} E8 stages.
     *
     * @param out receives the reconstruction at {@code out[outOff..outOff+nDims)}
     * @param codes receives {@link #codeBytes(int, int)} code bytes, stage by stage
     * @return the norm of the reconstruction
     */
    public static float encode(float[] x, int xOff, int nDims, int stages, float[] out, int outOff, byte[] codes) {
        assert (nDims & 7) == 0;
        int nBlocks = nDims >>> 3;
        for (int b = 0; b < nBlocks; b++) {
            codes[b] = encodeBlock(x, xOff + 8 * b, out, outOff + 8 * b);
        }
        if (stages == 1) {
            return (float) Math.sqrt(2.0 * nBlocks);
        }
        assert stages == 2;
        // least-squares gain of the first-stage reconstruction, then code the leftover
        double dot = 0;
        for (int t = 0; t < nDims; t++) {
            dot += (double) x[xOff + t] * out[outOff + t];
        }
        float gain = (float) (dot / (2.0 * nBlocks));
        float[][] s = scratch(nDims);
        float[] residual = s[0];
        float[] second = s[1];
        for (int t = 0; t < nDims; t++) {
            residual[t] = x[xOff + t] - gain * out[outOff + t];
        }
        for (int b = 0; b < nBlocks; b++) {
            codes[nBlocks + b] = encodeBlock(residual, 8 * b, second, 8 * b);
        }
        double normSq = 0;
        for (int t = 0; t < nDims; t++) {
            float v = out[outOff + t] + TWO_STAGE_BETA * second[t];
            out[outOff + t] = v;
            normSq += (double) v * v;
        }
        return (float) Math.sqrt(normSq);
    }

    /**
     * Builds the per-query lookup table {@code table[b * PAIRS + pair] = <q_b, representative(pair)>} for the projected
     * query {@code q} of {@code nDims} dimensions.
     */
    public static float[] buildTable(float[] q, int nDims) {
        int nBlocks = nDims >>> 3;
        float[] table = new float[nBlocks * PAIRS];
        for (int b = 0; b < nBlocks; b++) {
            int qOff = 8 * b;
            int tOff = b * PAIRS;
            for (int pair = 0; pair < PAIRS; pair++) {
                float s = 0f;
                int rOff = pair * 8;
                for (int t = 0; t < 8; t++) {
                    s += q[qOff + t] * REPS[rOff + t];
                }
                table[tOff + pair] = s;
            }
        }
        return table;
    }

    /**
     * The inner product between the query whose {@link #buildTable table} is given and the reconstruction coded by
     * {@code codes[off..off + codeBytes(nBlocks * 8, stages))}.
     */
    public static float score(float[] table, byte[] codes, int off, int nBlocks, int stages) {
        float s1 = 0f;
        for (int b = 0; b < nBlocks; b++) {
            int c = codes[off + b] & 0xFF;
            float v = table[b * PAIRS + (c & 0x7F)];
            s1 += (c & SIGN_BIT) != 0 ? -v : v;
        }
        if (stages == 1) {
            return s1;
        }
        float s2 = 0f;
        int off2 = off + nBlocks;
        for (int b = 0; b < nBlocks; b++) {
            int c = codes[off2 + b] & 0xFF;
            float v = table[b * PAIRS + (c & 0x7F)];
            s2 += (c & SIGN_BIT) != 0 ? -v : v;
        }
        return s1 + TWO_STAGE_BETA * s2;
    }
}
