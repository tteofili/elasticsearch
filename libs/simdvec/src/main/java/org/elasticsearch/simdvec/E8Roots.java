/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.simdvec;

import java.util.Arrays;

/**
 * The 240 roots of the E8 lattice as a residual code of about one bit per dimension.
 * <p>
 * A vector is split into blocks of {@link #BLOCK_DIMS} coordinates (the last block is zero-padded) and every block is
 * replaced by its nearest root, stored in one byte. Codewords are the roots scaled by two, so that every coordinate is an
 * integer and every codeword has squared norm 8:
 * <ul>
 *     <li>112 "spiky" codewords {@code (±2, ±2, 0, 0, 0, 0, 0, 0)}, the two non-zero entries in any of the 28 position pairs;</li>
 *     <li>128 "flat" codewords {@code (±1)^8} with an even number of minus signs.</li>
 * </ul>
 * Roots come in antipodal pairs, so a code byte is a sign in its top bit and one of 120 pair representatives in its
 * low seven bits: {@code 0..55} are the spiky pairs (index {@code 2 * pair + (signs differ)}, first coordinate positive),
 * {@code 56..119} the flat pairs (first coordinate positive, minus signs of coordinates 1..6 as bits, coordinate 7 fixed
 * by parity). Byte values {@code 120..127} are unused.
 * <p>
 * Encoding is exact and constant time: because all codewords have the same norm, the nearest one maximises the inner
 * product, and the best spiky and best flat codewords have closed forms. Scoring uses a per-query lookup table of
 * {@link #TABLE_SIZE} signed bytes per block holding {@code <query block, pair representative>}; a document's score is the
 * sum over blocks of the looked-up entry, negated when the sign bit is set. Within a bulk, codes are stored transposed
 * (block-major) so that the codes of one block for consecutive documents are contiguous.
 */
public final class E8Roots {

    /** Dimensions coded by one byte. */
    public static final int BLOCK_DIMS = 8;
    /** Lookup-table entries per block: 120 pair representatives padded to a power of two. */
    public static final int TABLE_SIZE = 128;
    /** Number of spiky pair representatives; flat representatives follow. */
    static final int SPIKY = 56;

    private static final int[] PAIR_I = new int[28];
    private static final int[] PAIR_J = new int[28];
    private static final int[] PAIR_INDEX = new int[BLOCK_DIMS * BLOCK_DIMS];

    static {
        for (int i = 0, p = 0; i < BLOCK_DIMS; i++) {
            for (int j = i + 1; j < BLOCK_DIMS; j++, p++) {
                PAIR_I[p] = i;
                PAIR_J[p] = j;
                PAIR_INDEX[i * BLOCK_DIMS + j] = p;
            }
        }
    }

    private E8Roots() {}

    /** Number of code bytes for a vector of {@code dims} dimensions. */
    public static int blocks(int dims) {
        return (dims + BLOCK_DIMS - 1) / BLOCK_DIMS;
    }

    /**
     * Encodes {@code x} block by block with the nearest codeword.
     *
     * @param x     the vector to encode
     * @param codes destination, one code (0..255) per block
     * @return {@code <x, x̂>}, the inner product of {@code x} with its reconstruction (the concatenated codewords)
     */
    public static float encode(float[] x, int[] codes) {
        float dot = 0f;
        for (int b = 0, off = 0; off < x.length; b++, off += BLOCK_DIMS) {
            long r = encodeBlock(x, off, Math.min(BLOCK_DIMS, x.length - off));
            codes[b] = (int) r & 0xFF;
            dot += Float.intBitsToFloat((int) (r >>> 32));
        }
        return dot;
    }

    /**
     * The largest magnitude of a table entry of {@code y}, which is the best codeword score of a block, computed in closed
     * form, maximised over the blocks. It fixes the scale of the table.
     */
    public static float maxEntry(float[] y) {
        float max = 0f;
        for (int off = 0; off < y.length; off += BLOCK_DIMS) {
            max = Math.max(max, bestScore(y, off, Math.min(BLOCK_DIMS, y.length - off)));
        }
        return max;
    }

    /**
     * The inner product of the block with its nearest codeword, as {@link #encodeBlock} computes it, but without the code and
     * without data dependent branches, which makes it several times cheaper.
     */
    static float bestScore(float[] x, int off, int len) {
        float a1 = 0f, a2 = 0f, sum = 0f, minAbs = Float.MAX_VALUE;
        int signs = 0;
        for (int k = 0; k < BLOCK_DIMS; k++) {
            float v = k < len ? x[off + k] : 0f;
            float a = Math.abs(v);
            sum += a;
            minAbs = Math.min(minAbs, a);
            a2 = Math.max(a2, Math.min(a1, a));
            a1 = Math.max(a1, a);
            signs ^= Float.floatToRawIntBits(v) >>> 31;
        }
        float flat = signs != 0 ? sum - 2f * minAbs : sum;
        return Math.max(2f * (a1 + a2), flat);
    }

    /** Returns the code of the nearest codeword in the low byte and its inner product with the block in the high int. */
    static long encodeBlock(float[] x, int off, int len) {
        int i1 = 0, i2 = 1, minIdx = 0, negMask = 0;
        float a1 = -1f, a2 = -1f, sum = 0f, minAbs = Float.MAX_VALUE;
        for (int k = 0; k < BLOCK_DIMS; k++) {
            float v = k < len ? x[off + k] : 0f;
            float a = Math.abs(v);
            sum += a;
            if (v < 0f) {
                negMask |= 1 << k;
            }
            if (a < minAbs) {
                minAbs = a;
                minIdx = k;
            }
            if (a > a1) {
                a2 = a1;
                i2 = i1;
                a1 = a;
                i1 = k;
            } else if (a > a2) {
                a2 = a;
                i2 = k;
            }
        }
        // best spiky codeword: the two largest magnitudes with their signs
        float spiky = 2f * (a1 + a2);
        // best flat codeword: the signs of x, with the smallest magnitude flipped if needed for even parity
        boolean odd = (Integer.bitCount(negMask) & 1) != 0;
        float flat = odd ? sum - 2f * minAbs : sum;
        int code;
        float score;
        if (spiky >= flat) {
            int i = Math.min(i1, i2), j = Math.max(i1, i2);
            int si = (negMask >>> i) & 1;
            code = (si << 7) | (PAIR_INDEX[i * BLOCK_DIMS + j] << 1) | (si ^ ((negMask >>> j) & 1));
            score = spiky;
        } else {
            if (odd) {
                negMask ^= 1 << minIdx;
            }
            int s0 = negMask & 1;
            int m = s0 == 0 ? negMask : ~negMask; // representative has a positive first coordinate
            code = (s0 << 7) | (SPIKY + ((m >>> 1) & 0x3F));
            score = flat;
        }
        return ((long) Float.floatToRawIntBits(score) << 32) | code;
    }

    /** Writes the (twice-scaled) codeword of {@code code} to {@code dst[off..off + 8)}. Codes 120..127 decode to zero. */
    public static void decode(int code, float[] dst, int off) {
        float sign = (code & 0x80) != 0 ? -1f : 1f;
        int idx = code & 0x7F;
        for (int k = 0; k < BLOCK_DIMS; k++) {
            dst[off + k] = 0f;
        }
        if (idx < SPIKY) {
            int p = idx >>> 1;
            dst[off + PAIR_I[p]] = 2f * sign;
            dst[off + PAIR_J[p]] = (idx & 1) != 0 ? -2f * sign : 2f * sign;
        } else if (idx < SPIKY + 64) {
            int m = idx - SPIKY;
            dst[off] = sign;
            for (int k = 1; k < 7; k++) {
                dst[off + k] = ((m >>> (k - 1)) & 1) != 0 ? -sign : sign;
            }
            dst[off + 7] = (Integer.bitCount(m) & 1) != 0 ? -sign : sign;
        }
    }

    /** {@code 1.5 * 2^23}: adding it to a float of magnitude below 2^22 leaves the value rounded to an integer in the low mantissa bits. */
    private static final float ROUND_MAGIC = 12582912f;

    /** Sign of the last coordinate of each flat representative: negative when the index has odd parity. */
    private static final float[] PARITY_SIGN = new float[64];

    static {
        for (int m = 0; m < 64; m++) {
            PARITY_SIGN[m] = (Integer.bitCount(m) & 1) != 0 ? -1f : 1f;
        }
    }

    /** Rounds {@code v}, which must lie in [-127, 127], to the nearest integer without branches (ties to even). */
    private static byte quantize(float v) {
        return (byte) Float.floatToRawIntBits(v + ROUND_MAGIC);
    }

    /**
     * Builds the int8 lookup table of a query: for every block, the inner products of the block with the 120 pair
     * representatives, padded with zeros to {@link #TABLE_SIZE}, all quantized with one scale.
     *
     * @param y     the (centred) query
     * @param table destination of {@code blocks(y.length) * TABLE_SIZE} bytes
     * @return the scale: {@code <y, codeword> ≈ scale * table entry}
     */
    public static float buildQueryTable(float[] y, byte[] table) {
        final int blocks = blocks(y.length);
        final float max = maxEntry(y);
        if (max == 0f) {
            Arrays.fill(table, 0, blocks * TABLE_SIZE, (byte) 0);
            return 0f;
        }
        final float inv = 127f / max;
        final float inv2 = 2f * inv;
        final float[] yb = new float[BLOCK_DIMS];
        final float[] flat = new float[64];
        for (int b = 0, off = 0; b < blocks; b++, off += BLOCK_DIMS) {
            final int n = Math.min(BLOCK_DIMS, y.length - off);
            for (int k = 0; k < n; k++) {
                yb[k] = y[off + k];
            }
            for (int k = n; k < BLOCK_DIMS; k++) {
                yb[k] = 0f;
            }
            final int base = b * TABLE_SIZE;
            for (int p = 0; p < 28; p++) {
                float yi = yb[PAIR_I[p]], yj = yb[PAIR_J[p]];
                table[base + 2 * p] = quantize((yi + yj) * inv2);
                table[base + 2 * p + 1] = quantize((yi - yj) * inv2);
            }
            // flat representatives by doubling: flipping coordinate k subtracts 2 * y_k
            flat[0] = yb[0] + yb[1] + yb[2] + yb[3] + yb[4] + yb[5] + yb[6];
            for (int m = 1; m < 64; m++) {
                flat[m] = flat[m & (m - 1)] - 2f * yb[Integer.numberOfTrailingZeros(m) + 1];
            }
            // the last coordinate is signed by the parity of the representative, folded into a table of +-1
            final float y7 = yb[7] * inv;
            for (int m = 0; m < 64; m++) {
                table[base + SPIKY + m] = quantize(flat[m] * inv + PARITY_SIGN[m] * y7);
            }
            Arrays.fill(table, base + SPIKY + 64, base + TABLE_SIZE, (byte) 0);
        }
        return max / 127f;
    }

    /**
     * Scores {@code count} documents whose codes are stored transposed ({@code codes[offset + block * count + doc]}).
     * Scores are integers (at most {@code 127 * blocks} in magnitude) and therefore exact as floats.
     */
    public static void scoreBulk(byte[] codes, int offset, byte[] table, int blocks, int count, float[] scores) {
        Arrays.fill(scores, 0, count, 0f);
        for (int b = 0; b < blocks; b++) {
            final int c0 = offset + b * count, t0 = b * TABLE_SIZE;
            for (int d = 0; d < count; d++) {
                final int c = codes[c0 + d];
                final int v = table[t0 + (c & 0x7F)];
                scores[d] += c < 0 ? -v : v;
            }
        }
    }
}
