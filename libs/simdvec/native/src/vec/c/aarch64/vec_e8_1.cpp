/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

// E8 root code scoring for NEON ("1st tier"). Each 128-entry block table sits in eight registers; one tbl looks up
// entries 0-63 and one tbx entries 64-127 for 16 documents. The code's sign bit becomes a +/-1 factor that a widening
// multiply-accumulate applies while adding into the int16 accumulators.

#include <stddef.h>
#include <arm_neon.h>
#include "vec.h"
#include "vec_e8.h"

static inline int8x16_t e8_lookup(const int8x16x4_t lo, const int8x16x4_t hi, const uint8x16_t c) {
    const uint8x16_t idx = vandq_u8(c, vdupq_n_u8(0x7F));
    const int8x16_t v = vqtbl4q_s8(lo, idx);
    // indices below 64 wrap to 192..255, out of range for tbx, which then keeps the tbl result
    return vqtbx4q_s8(v, hi, vsubq_u8(idx, vdupq_n_u8(64)));
}

// Scores 16 * G consecutive documents, sharing the table loads between the groups of 16.
template <int G>
static inline void e8_dot_group(
    const int8_t* codes,
    const int8_t* table,
    const int32_t blocks,
    const int32_t table_size,
    const int32_t stride,
    f32_t* results
) {
    int32x4_t sum[G][4];
    for (int g = 0; g < G; g++) {
        for (int k = 0; k < 4; k++) {
            sum[g][k] = vdupq_n_s32(0);
        }
    }
    for (int32_t b0 = 0; b0 < blocks; b0 += E8_FLUSH_BLOCKS) {
        const int32_t b1 = blocks - b0 < E8_FLUSH_BLOCKS ? blocks : b0 + E8_FLUSH_BLOCKS;
        int16x8_t acc[G][2];
        for (int g = 0; g < G; g++) {
            acc[g][0] = vdupq_n_s16(0);
            acc[g][1] = vdupq_n_s16(0);
        }
        for (int32_t b = b0; b < b1; b++) {
            const int8_t* t = table + (size_t) b * table_size;
            const int8x16x4_t lo = vld1q_s8_x4(t);
            const int8x16x4_t hi = vld1q_s8_x4(t + 64);
            const uint8_t* c = (const uint8_t*) codes + (size_t) b * stride;
            for (int g = 0; g < G; g++) {
                const uint8x16_t cg = vld1q_u8(c + 16 * g);
                const int8x16_t v = e8_lookup(lo, hi, cg);
                const int8x16_t sign = vorrq_s8(vshrq_n_s8(vreinterpretq_s8_u8(cg), 7), vdupq_n_s8(1));
                acc[g][0] = vmlal_s8(acc[g][0], vget_low_s8(v), vget_low_s8(sign));
                acc[g][1] = vmlal_high_s8(acc[g][1], v, sign);
            }
        }
        for (int g = 0; g < G; g++) {
            sum[g][0] = vaddw_s16(sum[g][0], vget_low_s16(acc[g][0]));
            sum[g][1] = vaddw_high_s16(sum[g][1], acc[g][0]);
            sum[g][2] = vaddw_s16(sum[g][2], vget_low_s16(acc[g][1]));
            sum[g][3] = vaddw_high_s16(sum[g][3], acc[g][1]);
        }
    }
    for (int g = 0; g < G; g++) {
        for (int k = 0; k < 4; k++) {
            vst1q_f32(results + 16 * g + 4 * k, vcvtq_f32_s32(sum[g][k]));
        }
    }
}

EXPORT void vec_dote8_bulk(
    const int8_t* codes,
    const int8_t* table,
    const int32_t blocks,
    const int32_t table_size,
    const int32_t count,
    f32_t* results
) {
    int32_t d = 0;
    for (; d + 32 <= count; d += 32) {
        e8_dot_group<2>(codes + d, table, blocks, table_size, count, results + d);
    }
    for (; d + 16 <= count; d += 16) {
        e8_dot_group<1>(codes + d, table, blocks, table_size, count, results + d);
    }
    e8_dot_scalar(codes, table, blocks, table_size, count, d, results);
}

// Builds the int8 query table of y and returns its scale, such that <y, codeword> is about scale * entry. The table is zero,
// and so is the scale, when y is.
EXPORT f32_t vec_e8_table(const f32_t* y, int8_t* table, const int32_t dims, const int32_t blocks, const int32_t table_size) {
    const f32_t max = e8_max_entry(y, dims);
    if (max == 0.f) {
        memset(table, 0, (size_t) blocks * table_size);
        return 0.f;
    }
    e8_build_table(y, dims, table, table_size, 127.f / max);
    return max / 127.f;
}
