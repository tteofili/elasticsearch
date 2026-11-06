/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.simdvec;

/**
 * E8 lattice block code (experimental): the encoder and the per-query lookup table shared by the scan kernels.
 * <p>
 * A vector is split into blocks of eight coordinates (zero-padded to a multiple of eight) and each block is replaced by
 * the nearest of the 240 minimal vectors (roots) of the E8 lattice: 112 "spiky" roots {@code (±1, ±1, 0^6)} and 128
 * "flat" roots {@code (±1/2)^8} with an even number of minus signs. All roots have norm √2, so the nearest root is the one
 * with the largest inner product, found exactly in O(8).
 * <p>
 * Roots come in ± pairs, so a root is one byte, {@code sign << 7 | pair} with {@code pair} in [0, 120): pairs below 56
 * are spiky ({@code pair = 2 * p + k}, p indexing the position pair {@code i < j}, representative {@code e_i + e_j} for
 * {@code k = 0} and {@code e_i - e_j} for {@code k = 1}); the others are flat, representative
 * {@code (1/2)(+1, s_1, ..., s_7)} whose minus-sign mask is the {@code (pair - 56)}-th even-parity 7-bit mask. This is the
 * byte format of the server's {@code E8Codebook} (ASH), so both paths share codes and tables.
 * <p>
 * The query side is a table of {@link #TABLE_STRIDE} int8 entries per block, {@code round(<q_b, rep(pair)> / scale)},
 * with one scale for the whole table, so a document's estimate is {@code scale * sum_b (±table[b][pair_b])}, an integer
 * sum the kernels compute exactly.
 */
public final class E8Lattice {

    /** Number of ± pairs of roots, and so the number of used table entries per block. */
    public static final int PAIRS = 120;
    /** Table entries per block (the pairs, padded to a power of two for the SIMD lookups). */
    public static final int TABLE_STRIDE = 128;
    /** Coordinates per block. */
    public static final int BLOCK = 8;

    private static final int SPIKY_PAIRS = 56;
    private static final int SIGN_BIT = 0x80;

    /** Canonical representative of each pair, row-major ({@link #PAIRS} x 8). */
    private static final float[] REPS = new float[PAIRS * BLOCK];
    private static final int[] POSITION_PAIR = new int[64];
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
        assert p == 28 && m == 64;
    }

    private E8Lattice() {}

    /** Number of eight-dimensional blocks for {@code dims} dimensions (the last block is zero-padded). */
    public static int blocks(int dims) {
        return (dims + BLOCK - 1) / BLOCK;
    }

    /** Coordinate {@code t} in [0, 8) of the representative of {@code pair}. */
    public static float representative(int pair, int t) {
        return REPS[pair * BLOCK + t];
    }

    /**
     * Codes the block {@code x[off .. off + len)} (len ≤ 8; missing coordinates are zero) with its nearest root.
     *
     * @return the code byte, {@code sign << 7 | pair}
     */
    public static byte encodeBlock(float[] x, int off, int len) {
        float v0 = len > 0 ? x[off] : 0f;
        // Best spiky root: the two largest magnitudes, with their signs
        int i1 = 0, i2 = -1;
        float a1 = Math.abs(v0), a2 = -1f;
        // Best flat root: the signs of x, with the smallest magnitude flipped if the number of minus signs is odd
        int minusMask = v0 < 0 ? 1 : 0;
        int smallest = 0;
        float smallestAbs = a1;
        float flatScore = a1;
        for (int t = 1; t < 8; t++) {
            float v = t < len ? x[off + t] : 0f;
            float a = Math.abs(v);
            if (a > a1) {
                i2 = i1;
                a2 = a1;
                i1 = t;
                a1 = a;
            } else if (a > a2) {
                i2 = t;
                a2 = a;
            }
            flatScore += a;
            if (v < 0) {
                minusMask |= 1 << t;
            }
            if (a < smallestAbs) {
                smallestAbs = a;
                smallest = t;
            }
        }
        float spikyScore = a1 + a2;
        if ((Integer.bitCount(minusMask) & 1) != 0) {
            minusMask ^= 1 << smallest;
            flatScore -= 2 * smallestAbs;
        }
        flatScore *= 0.5f;
        int code;
        if (spikyScore >= flatScore) {
            int lo = Math.min(i1, i2), hi = Math.max(i1, i2);
            boolean loNeg = (lo < len ? x[off + lo] : 0f) < 0;
            boolean hiNeg = (hi < len ? x[off + hi] : 0f) < 0;
            int k = loNeg == hiNeg ? 0 : 1;
            code = 2 * POSITION_PAIR[lo * 8 + hi] + k;
            if (loNeg) {
                code |= SIGN_BIT;
            }
        } else {
            boolean neg = (minusMask & 1) != 0;
            int repMask = neg ? (~minusMask & 0xFF) : minusMask;
            code = SPIKY_PAIRS + FLAT_MASK[(repMask >>> 1) & 0x7F];
            if (neg) {
                code |= SIGN_BIT;
            }
        }
        return (byte) code;
    }

    /** Inner product between {@code x[off .. off + len)} (zero-padded to 8) and the root coded by {@code code}. */
    public static float dotRoot(float[] x, int off, int len, byte code) {
        int c = code & 0xFF;
        int pair = c & 0x7F;
        float s = 0f;
        for (int t = 0; t < len && t < BLOCK; t++) {
            s += x[off + t] * REPS[pair * BLOCK + t];
        }
        return (c & SIGN_BIT) != 0 ? -s : s;
    }

    /**
     * Codes {@code x[0 .. dims)} block by block into {@code codes[codesOff .. codesOff + blocks(dims))}.
     *
     * @return {@code <x, x̂>}, the inner product between the vector and its (unscaled) reconstruction
     */
    public static float encode(float[] x, int dims, byte[] codes, int codesOff) {
        int nBlocks = blocks(dims);
        double dot = 0;
        for (int b = 0; b < nBlocks; b++) {
            int off = b * BLOCK;
            int len = Math.min(BLOCK, dims - off);
            byte code = encodeBlock(x, off, len);
            codes[codesOff + b] = code;
            dot += dotRoot(x, off, len, code);
        }
        return (float) dot;
    }

    /**
     * Fills {@code table[b * TABLE_STRIDE + pair]} with {@code <q_b, rep(pair)>} in float, for {@code q[0 .. dims)}
     * (zero-padded), entries {@code PAIRS .. TABLE_STRIDE} of every block set to zero.
     */
    public static void buildFloatTable(float[] q, int dims, float[] table) {
        int nBlocks = blocks(dims);
        assert table.length >= nBlocks * TABLE_STRIDE;
        float[] qb = new float[BLOCK];
        for (int b = 0; b < nBlocks; b++) {
            int off = b * BLOCK;
            int len = Math.min(BLOCK, dims - off);
            for (int t = 0; t < BLOCK; t++) {
                qb[t] = t < len ? q[off + t] : 0f;
            }
            int tOff = b * TABLE_STRIDE;
            // spiky: q_i ± q_j
            int p = 0;
            for (int i = 0; i < 8; i++) {
                for (int j = i + 1; j < 8; j++) {
                    table[tOff + 2 * p] = qb[i] + qb[j];
                    table[tOff + 2 * p + 1] = qb[i] - qb[j];
                    p++;
                }
            }
            // flat: (q_0 + sum_t s_t q_t) / 2
            for (int pair = SPIKY_PAIRS; pair < PAIRS; pair++) {
                float s = 0f;
                for (int t = 0; t < BLOCK; t++) {
                    s += qb[t] * REPS[pair * BLOCK + t];
                }
                table[tOff + pair] = s;
            }
            for (int pair = PAIRS; pair < TABLE_STRIDE; pair++) {
                table[tOff + pair] = 0f;
            }
        }
    }

    /**
     * Quantizes a float table (several stages may be concatenated, each scaled by its weight) to int8 with one scale.
     *
     * @param floatTables per stage, a table from {@link #buildFloatTable}
     * @param weights per stage, the weight of that stage in the reconstruction (1 for one stage)
     * @param nBlocks blocks per stage
     * @param table8 receives {@code stages * nBlocks * TABLE_STRIDE} int8 entries, stage by stage
     * @return the scale: a document's estimate is {@code scale * integer sum}
     */
    public static float quantizeTable(float[][] floatTables, float[] weights, int nBlocks, byte[] table8) {
        float max = 0f;
        for (int s = 0; s < floatTables.length; s++) {
            float w = Math.abs(weights[s]);
            for (int i = 0; i < nBlocks * TABLE_STRIDE; i++) {
                max = Math.max(max, Math.abs(floatTables[s][i]) * w);
            }
        }
        float scale = max > 0f ? max / 127f : 1f;
        float inv = 1f / scale;
        for (int s = 0; s < floatTables.length; s++) {
            int base = s * nBlocks * TABLE_STRIDE;
            for (int i = 0; i < nBlocks * TABLE_STRIDE; i++) {
                int v = Math.round(floatTables[s][i] * weights[s] * inv);
                table8[base + i] = (byte) Math.max(-127, Math.min(127, v));
            }
        }
        return scale;
    }

    /** Convenience for one stage: builds and quantizes the table of {@code q[0 .. dims)}; returns the scale. */
    public static float buildTable(float[] q, int dims, byte[] table8) {
        int nBlocks = blocks(dims);
        float[] t = new float[nBlocks * TABLE_STRIDE];
        buildFloatTable(q, dims, t);
        return quantizeTable(new float[][] { t }, new float[] { 1f }, nBlocks, table8);
    }

    /**
     * Reference integer sum for one document of a transposed bulk: {@code codes[b * count + j]} is the code of document
     * {@code j} for (virtual) block {@code b}.
     */
    public static int lookupOne(byte[] codes, int codesOff, byte[] table8, int blocks, int count, int j) {
        int s = 0;
        for (int b = 0; b < blocks; b++) {
            int c = codes[codesOff + b * count + j] & 0xFF;
            int x = table8[b * TABLE_STRIDE + (c & 0x7F)];
            s += (c & SIGN_BIT) != 0 ? -x : x;
        }
        return s;
    }
}
