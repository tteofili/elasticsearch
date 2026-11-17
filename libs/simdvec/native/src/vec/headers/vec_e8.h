/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

// Shared pieces of the E8 root code kernels (see org.elasticsearch.simdvec.E8Roots).
// Codes are stored transposed within a bulk: codes[block * count + doc]. A code byte holds the sign of the root in its
// top bit and the index of an antipodal pair in its low 7 bits; the query table holds, for each block, table_size
// (128) signed bytes. A document's score is the sum over blocks of +/- table[block][code & 0x7F].

#ifndef VEC_E8_INCLUDED
#define VEC_E8_INCLUDED

#include <stddef.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include "vec.h"

// int16 accumulators absorb this many blocks of int8 entries (|entry| <= 127) without overflow
#define E8_FLUSH_BLOCKS 256

static inline void e8_dot_scalar(
    const int8_t* codes,
    const int8_t* table,
    const int32_t blocks,
    const int32_t table_size,
    const int32_t count,
    int32_t from,
    f32_t* results
) {
    for (int32_t d = from; d < count; d++) {
        int32_t acc = 0;
        for (int32_t b = 0; b < blocks; b++) {
            const int8_t c = codes[(size_t) b * count + d];
            const int32_t v = table[(size_t) b * table_size + (c & 0x7F)];
            acc += c < 0 ? -v : v;
        }
        results[d] = (f32_t) acc;
    }
}

// Query tables of the E8 root code. For every block of 8 query coordinates the table holds the inner products with the 120
// pair representatives (see E8Roots.buildQueryTable and E8Roots.decode: 0..55 are the spiky pairs, 2 * (y_i + y_j) and
// 2 * (y_i - y_j) for the 28 position pairs in lexicographic order; 56..119 are the flat pairs, coordinate 0 positive, bit j
// of the index flipping coordinate j + 1 and coordinate 7 signed by the parity of the index), scaled so that the largest
// magnitude of the whole table is 127; entries 120..127 are zero.
//
// The entries are computed in 16 bit fixed point with 8 fractional bits: each y * inv is rounded once, every entry is a sum
// of at most eight of them, so it is within 8 * 2^-9 of the exact value before the final rounding, and fits an int16 (the
// intermediate sums may wrap, which is harmless in modular arithmetic as the entry itself is at most 127 * 256). Eight
// entries then fill one 128 bit register, the gathers of the spiky entries are shuffles, the flat entries are built from two
// 8 entry tables and a rank one term (their separable structure), and the rounding of 8 entries is one shift and a narrowing.
// The portable vector extensions compile to NEON, SSE and AVX shuffles and arithmetic alike.

// The best codeword score of a block, as E8Roots.bestScore computes it: the larger of the best spiky codeword (the two
// largest magnitudes) and the best flat one (all signs, with the smallest magnitude flipped if the parity is odd).
static inline f32_t e8_best_score(const f32_t* x) {
    // fmaxf / fminf rather than comparisons: they compile to branch free min / max instructions, which the compiler can also
    // vectorize across the blocks of the caller's loop
    f32_t a1 = 0.f, a2 = 0.f, sum = 0.f, min_abs = 3.4028235e38f;
    uint32_t signs = 0;
    for (int32_t k = 0; k < 8; k++) {
        const f32_t a = fabsf(x[k]);
        sum += a;
        min_abs = fminf(min_abs, a);
        a2 = fmaxf(a2, fminf(a1, a));
        a1 = fmaxf(a1, a);
        uint32_t bits;
        memcpy(&bits, x + k, sizeof(bits));
        signs ^= bits >> 31;
    }
    const f32_t flat = signs != 0 ? sum - 2.f * min_abs : sum;
    return fmaxf(2.f * (a1 + a2), flat);
}

// The largest magnitude of a table entry of y: the largest best score of its blocks.
static inline f32_t e8_max_entry(const f32_t* y, const int32_t dims) {
    const int32_t blocks = (dims + 7) / 8;
    const int32_t whole = dims / 8;
    f32_t max = 0.f;
    for (int32_t b = 0; b < whole; b++) {
        max = fmaxf(max, e8_best_score(y + b * 8));
    }
    if (blocks > whole) {
        f32_t tail[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        for (int32_t k = 0; k < dims - whole * 8; k++) {
            tail[k] = y[whole * 8 + k];
        }
        max = fmaxf(max, e8_best_score(tail));
    }
    return max;
}

typedef int16_t v8s __attribute__((vector_size(16)));
typedef int8_t v8b __attribute__((vector_size(8)));
typedef int8_t v16b __attribute__((vector_size(16)));
typedef float v4f __attribute__((vector_size(16)));
typedef int32_t v4i __attribute__((vector_size(16)));
typedef int16_t v4s __attribute__((vector_size(8)));

static inline void e8_build_table(const f32_t* y, const int32_t dims, int8_t* table, const int32_t table_size, const f32_t inv) {
    const v8s SGN = {1, -1, 1, -1, 1, -1, 1, -1};
    const v8s PAR = {1, -1, -1, 1, -1, 1, 1, -1};
    const v8s S1 = {1, -1, 1, -1, 1, -1, 1, -1};
    const v8s S2 = {1, 1, -1, -1, 1, 1, -1, -1};
    const v8s S3 = {1, 1, 1, 1, -1, -1, -1, -1};
    // y * inv * 256 + 1.5 * 2^23 leaves round(y * inv * 256) in the low mantissa bits
    const f32_t scale = inv * 256.f;
    const v4f magic = {12582912.f, 12582912.f, 12582912.f, 12582912.f};
    const int32_t blocks = (dims + 7) / 8;
    // rounds 8 entries (fixed point, 8 fractional bits) to int8
    auto quantize = [](v8s v) {
        return __builtin_convertvector((v + (v8s){128, 128, 128, 128, 128, 128, 128, 128}) >> 8, v8b);
    };
    auto put2 = [](int8_t* dst, v8b a, v8b b) {
        const v16b ab = __builtin_shufflevector(a, b, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        __builtin_memcpy(dst, &ab, 16);
    };
    for (int32_t blk = 0; blk < blocks; blk++) {
        int8_t* out = table + (size_t) blk * table_size;
        f32_t buf[8];
        const f32_t* src = y + blk * 8;
        if (blk * 8 + 8 > dims) {
            for (int k = 0; k < 8; k++) buf[k] = blk * 8 + k < dims ? y[blk * 8 + k] : 0.f;
            src = buf;
        }
        v4f g0, g1;
        __builtin_memcpy(&g0, src, 16);
        __builtin_memcpy(&g1, src + 4, 16);
        g0 = g0 * scale + magic;
        g1 = g1 * scale + magic;
        const v4i i0 = (v4i) g0, i1 = (v4i) g1;
        const v4s h0 = __builtin_convertvector(i0, v4s), h1 = __builtin_convertvector(i1, v4s);
        const v8s y16 = __builtin_shufflevector(h0, h1, 0, 1, 2, 3, 4, 5, 6, 7);
        const v8s y2 = y16 + y16;
        const v8b s0 = quantize(__builtin_shufflevector(y2, y2, 0, 0, 0, 0, 0, 0, 0, 0) + __builtin_shufflevector(y2, y2, 1, 1, 2, 2, 3, 3, 4, 4) * SGN);
        const v8b s1 = quantize(__builtin_shufflevector(y2, y2, 0, 0, 0, 0, 0, 0, 1, 1) + __builtin_shufflevector(y2, y2, 5, 5, 6, 6, 7, 7, 2, 2) * SGN);
        const v8b s2 = quantize(__builtin_shufflevector(y2, y2, 1, 1, 1, 1, 1, 1, 1, 1) + __builtin_shufflevector(y2, y2, 3, 3, 4, 4, 5, 5, 6, 6) * SGN);
        const v8b s3 = quantize(__builtin_shufflevector(y2, y2, 1, 1, 2, 2, 2, 2, 2, 2) + __builtin_shufflevector(y2, y2, 7, 7, 3, 3, 4, 4, 5, 5) * SGN);
        const v8b s4 = quantize(__builtin_shufflevector(y2, y2, 2, 2, 2, 2, 3, 3, 3, 3) + __builtin_shufflevector(y2, y2, 6, 6, 7, 7, 4, 4, 5, 5) * SGN);
        const v8b s5 = quantize(__builtin_shufflevector(y2, y2, 3, 3, 3, 3, 4, 4, 4, 4) + __builtin_shufflevector(y2, y2, 6, 6, 7, 7, 5, 5, 6, 6) * SGN);
        const v8b s6 = quantize(__builtin_shufflevector(y2, y2, 4, 4, 5, 5, 5, 5, 6, 6) + __builtin_shufflevector(y2, y2, 7, 7, 6, 6, 7, 7, 7, 7) * SGN);
        const v8s b1 = __builtin_shufflevector(y16, y16, 1, 1, 1, 1, 1, 1, 1, 1);
        const v8s b2 = __builtin_shufflevector(y16, y16, 2, 2, 2, 2, 2, 2, 2, 2);
        const v8s b3 = __builtin_shufflevector(y16, y16, 3, 3, 3, 3, 3, 3, 3, 3);
        const v8s b4 = __builtin_shufflevector(y16, y16, 4, 4, 4, 4, 4, 4, 4, 4);
        const v8s b5 = __builtin_shufflevector(y16, y16, 5, 5, 5, 5, 5, 5, 5, 5);
        const v8s b6 = __builtin_shufflevector(y16, y16, 6, 6, 6, 6, 6, 6, 6, 6);
        const v8s b7 = __builtin_shufflevector(y16, y16, 7, 7, 7, 7, 7, 7, 7, 7);
        const v8s b0 = __builtin_shufflevector(y16, y16, 0, 0, 0, 0, 0, 0, 0, 0);
        const v8s lo = b1 * S1 + b2 * S2 + b3 * S3;
        const v8s hi = b0 + b4 * S1 + b5 * S2 + b6 * S3;
        const v8s p7 = b7 * PAR;
        const v8b f0 = quantize(lo + __builtin_shufflevector(hi, hi, 0, 0, 0, 0, 0, 0, 0, 0) + PAR * __builtin_shufflevector(p7, p7, 0, 0, 0, 0, 0, 0, 0, 0));
        const v8b f1 = quantize(lo + __builtin_shufflevector(hi, hi, 1, 1, 1, 1, 1, 1, 1, 1) + PAR * __builtin_shufflevector(p7, p7, 1, 1, 1, 1, 1, 1, 1, 1));
        const v8b f2 = quantize(lo + __builtin_shufflevector(hi, hi, 2, 2, 2, 2, 2, 2, 2, 2) + PAR * __builtin_shufflevector(p7, p7, 2, 2, 2, 2, 2, 2, 2, 2));
        const v8b f3 = quantize(lo + __builtin_shufflevector(hi, hi, 3, 3, 3, 3, 3, 3, 3, 3) + PAR * __builtin_shufflevector(p7, p7, 3, 3, 3, 3, 3, 3, 3, 3));
        const v8b f4 = quantize(lo + __builtin_shufflevector(hi, hi, 4, 4, 4, 4, 4, 4, 4, 4) + PAR * __builtin_shufflevector(p7, p7, 4, 4, 4, 4, 4, 4, 4, 4));
        const v8b f5 = quantize(lo + __builtin_shufflevector(hi, hi, 5, 5, 5, 5, 5, 5, 5, 5) + PAR * __builtin_shufflevector(p7, p7, 5, 5, 5, 5, 5, 5, 5, 5));
        const v8b f6 = quantize(lo + __builtin_shufflevector(hi, hi, 6, 6, 6, 6, 6, 6, 6, 6) + PAR * __builtin_shufflevector(p7, p7, 6, 6, 6, 6, 6, 6, 6, 6));
        const v8b f7 = quantize(lo + __builtin_shufflevector(hi, hi, 7, 7, 7, 7, 7, 7, 7, 7) + PAR * __builtin_shufflevector(p7, p7, 7, 7, 7, 7, 7, 7, 7, 7));
        const v8b z = {0, 0, 0, 0, 0, 0, 0, 0};
        put2(out, s0, s1);
        put2(out + 16, s2, s3);
        put2(out + 32, s4, s5);
        put2(out + 48, s6, f0);
        put2(out + 64, f1, f2);
        put2(out + 80, f3, f4);
        put2(out + 96, f5, f6);
        put2(out + 112, f7, z);
    }
}

#endif // VEC_E8_INCLUDED
