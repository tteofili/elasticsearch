/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

// E8 root code scoring for AVX2 ("1st tier"). pshufb looks up 16 entries; the 128-entry table is split into eight
// 16-entry chunks stored as prefix XORs (t'0 = t0, t'k = tk ^ tk-1). Subtracting 16 * k from the index leaves it
// non-negative exactly for the chunks at or below the entry's chunk K, and pshufb zeroes negative indices, so XOR-ing
// the eight lookups telescopes to tK. 32 documents per register.

#include <stddef.h>
#include <immintrin.h>
#include "vec.h"
#include "vec_e8.h"

EXPORT void vec_dote8_bulk(
    const int8_t* codes,
    const int8_t* table,
    const int32_t blocks,
    const int32_t table_size,
    const int32_t count,
    f32_t* results
) {
    const __m256i low7 = _mm256_set1_epi8(0x7F);
    const __m256i one = _mm256_set1_epi8(1);
    const __m256i sixteen = _mm256_set1_epi8(16);
    int32_t d = 0;
    for (; d + 32 <= count; d += 32) {
        __m256i sum[4] = { _mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256() };
        for (int32_t b0 = 0; b0 < blocks; b0 += E8_FLUSH_BLOCKS) {
            const int32_t b1 = blocks - b0 < E8_FLUSH_BLOCKS ? blocks : b0 + E8_FLUSH_BLOCKS;
            __m256i acc0 = _mm256_setzero_si256(), acc1 = _mm256_setzero_si256();
            for (int32_t b = b0; b < b1; b++) {
                const int8_t* t = table + (size_t) b * table_size;
                const __m256i c = _mm256_loadu_si256((const __m256i*) (codes + (size_t) b * count + d));
                __m256i idx = _mm256_and_si256(c, low7);
                __m128i prev = _mm_loadu_si128((const __m128i*) t);
                __m256i v = _mm256_shuffle_epi8(_mm256_broadcastsi128_si256(prev), idx);
                for (int k = 1; k < 8; k++) {
                    const __m128i cur = _mm_loadu_si128((const __m128i*) (t + 16 * k));
                    idx = _mm256_sub_epi8(idx, sixteen);
                    v = _mm256_xor_si256(v, _mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_xor_si128(cur, prev)), idx));
                    prev = cur;
                }
                // negate where the code's sign bit is set (c | 1 is never zero)
                v = _mm256_sign_epi8(v, _mm256_or_si256(c, one));
                acc0 = _mm256_add_epi16(acc0, _mm256_cvtepi8_epi16(_mm256_castsi256_si128(v)));
                acc1 = _mm256_add_epi16(acc1, _mm256_cvtepi8_epi16(_mm256_extracti128_si256(v, 1)));
            }
            sum[0] = _mm256_add_epi32(sum[0], _mm256_cvtepi16_epi32(_mm256_castsi256_si128(acc0)));
            sum[1] = _mm256_add_epi32(sum[1], _mm256_cvtepi16_epi32(_mm256_extracti128_si256(acc0, 1)));
            sum[2] = _mm256_add_epi32(sum[2], _mm256_cvtepi16_epi32(_mm256_castsi256_si128(acc1)));
            sum[3] = _mm256_add_epi32(sum[3], _mm256_cvtepi16_epi32(_mm256_extracti128_si256(acc1, 1)));
        }
        for (int k = 0; k < 4; k++) {
            _mm256_storeu_ps(results + d + 8 * k, _mm256_cvtepi32_ps(sum[k]));
        }
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
