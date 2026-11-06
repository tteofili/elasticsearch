/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

// E8 lattice block codes (experimental): per-query int8 table lookup over a transposed bulk of codes.
//
// Layout. A bulk of `count` documents with `blocks` code bytes each is stored block-major ("transposed"):
// codes[b * count + j] is the code of document j for block b. A code byte is sign << 7 | pair, with pair in [0, 120).
// The table holds, for every block, 128 int8 entries (pairs 0..119, the rest unused): table[b * 128 + pair].
// The result is sums[j] = sum_b (sign ? -1 : 1) * table[b * 128 + pair], an exact integer.
//
// NEON: the 128-entry table of a block sits in 8 q-registers; tbl4 looks up pairs 0..63 and tbx4 pairs 64..127 for
// 16 documents at a time. Sums are accumulated in 16-bit lanes, flushed to 32 bits every 256 blocks (|entry| <= 127).

#include <stddef.h>
#include <stdint.h>
#include <arm_neon.h>
#include "vec.h"
#include "vec_common.h"
#include "aarch64/aarch64_vec_common.h"

static inline int8x16_t e8_lookup16(const int8x16x4_t& t0, const int8x16x4_t& t1, const uint8x16_t c) {
    const uint8x16_t idx = vandq_u8(c, vdupq_n_u8(127));
    int8x16_t r = vqtbl4q_s8(t0, idx);                                // pairs 0..63 (0 for idx >= 64)
    r = vqtbx4q_s8(r, t1, vsubq_u8(idx, vdupq_n_u8(64)));             // pairs 64..127 (idx < 64 wraps out of range: kept)
    const int8x16_t s = vshrq_n_s8(vreinterpretq_s8_u8(c), 7);        // 0 or -1 from the sign bit
    return vsubq_s8(veorq_s8(r, s), s);                               // conditional negate
}

static inline int32_t e8_scalar_one(const int8_t* codes, const int8_t* table, int32_t blocks, int32_t count, int32_t j) {
    int32_t s = 0;
    for (int32_t b = 0; b < blocks; b++) {
        const uint8_t c = (uint8_t)codes[(size_t)b * count + j];
        const int32_t x = table[(size_t)b * 128 + (c & 127)];
        s += (c & 128) ? -x : x;
    }
    return s;
}

EXPORT void vec_e8_lookup_bulk(
    const int8_t* codes,
    const int8_t* table,
    const int32_t blocks,
    const int32_t count,
    int32_t* sums) {
    int32_t j = 0;
    // two chunks of 16 documents per table load (the common bulk of 32)
    for (; j + 32 <= count; j += 32) {
        int32x4_t w[8];
        for (int k = 0; k < 8; k++) w[k] = vdupq_n_s32(0);
        int32_t b = 0;
        while (b < blocks) {
            const int32_t end = b + 256 < blocks ? b + 256 : blocks;
            int16x8_t a0 = vdupq_n_s16(0), a1 = vdupq_n_s16(0), a2 = vdupq_n_s16(0), a3 = vdupq_n_s16(0);
            for (; b < end; b++) {
                const int8x16x4_t t0 = vld1q_s8_x4(table + (size_t)b * 128);
                const int8x16x4_t t1 = vld1q_s8_x4(table + (size_t)b * 128 + 64);
                const uint8_t* p = (const uint8_t*)codes + (size_t)b * count + j;
                const int8x16_t r0 = e8_lookup16(t0, t1, vld1q_u8(p));
                const int8x16_t r1 = e8_lookup16(t0, t1, vld1q_u8(p + 16));
                a0 = vaddw_s8(a0, vget_low_s8(r0));
                a1 = vaddw_high_s8(a1, r0);
                a2 = vaddw_s8(a2, vget_low_s8(r1));
                a3 = vaddw_high_s8(a3, r1);
            }
            w[0] = vaddw_s16(w[0], vget_low_s16(a0)); w[1] = vaddw_high_s16(w[1], a0);
            w[2] = vaddw_s16(w[2], vget_low_s16(a1)); w[3] = vaddw_high_s16(w[3], a1);
            w[4] = vaddw_s16(w[4], vget_low_s16(a2)); w[5] = vaddw_high_s16(w[5], a2);
            w[6] = vaddw_s16(w[6], vget_low_s16(a3)); w[7] = vaddw_high_s16(w[7], a3);
        }
        for (int k = 0; k < 8; k++) vst1q_s32(sums + j + 4 * k, w[k]);
    }
    for (; j + 16 <= count; j += 16) {
        int32x4_t w0 = vdupq_n_s32(0), w1 = vdupq_n_s32(0), w2 = vdupq_n_s32(0), w3 = vdupq_n_s32(0);
        int32_t b = 0;
        while (b < blocks) {
            const int32_t end = b + 256 < blocks ? b + 256 : blocks;
            int16x8_t a0 = vdupq_n_s16(0), a1 = vdupq_n_s16(0);
            for (; b < end; b++) {
                const int8x16x4_t t0 = vld1q_s8_x4(table + (size_t)b * 128);
                const int8x16x4_t t1 = vld1q_s8_x4(table + (size_t)b * 128 + 64);
                const int8x16_t r = e8_lookup16(t0, t1, vld1q_u8((const uint8_t*)codes + (size_t)b * count + j));
                a0 = vaddw_s8(a0, vget_low_s8(r));
                a1 = vaddw_high_s8(a1, r);
            }
            w0 = vaddw_s16(w0, vget_low_s16(a0)); w1 = vaddw_high_s16(w1, a0);
            w2 = vaddw_s16(w2, vget_low_s16(a1)); w3 = vaddw_high_s16(w3, a1);
        }
        vst1q_s32(sums + j, w0); vst1q_s32(sums + j + 4, w1); vst1q_s32(sums + j + 8, w2); vst1q_s32(sums + j + 12, w3);
    }
    // fewer than 16 documents left: a vector load would read past this bulk's rows
    for (; j < count; j++) {
        sums[j] = e8_scalar_one(codes, table, blocks, count, j);
    }
}
