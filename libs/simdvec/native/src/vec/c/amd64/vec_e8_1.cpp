/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

// E8 lattice block codes (experimental), tier 1 (AVX2). See aarch64/vec_e8_1.cpp for the layout and contract.
// AVX2 has no byte permute wider than 16 entries, so the 120-entry table is looked up as 8 pshufb over 16-entry
// slices, selected by the high nibble of the pair index: 32 documents per ymm register.

#include <stddef.h>
#include <stdint.h>
#include <immintrin.h>
#include "vec.h"
#include "vec_common.h"
#include "amd64/amd64_vec_common.h"

static inline int32_t e8_scalar_one(const int8_t* codes, const int8_t* table, int32_t blocks, int32_t count, int32_t j) {
    int32_t s = 0;
    for (int32_t b = 0; b < blocks; b++) {
        const uint8_t c = (uint8_t)codes[(size_t)b * count + j];
        const int32_t x = table[(size_t)b * 128 + (c & 127)];
        s += (c & 128) ? -x : x;
    }
    return s;
}

static inline __m256i e8_lookup32_avx2(const int8_t* t, const __m256i c) {
    const __m256i m127 = _mm256_set1_epi8(127), m15 = _mm256_set1_epi8(15);
    const __m256i idx = _mm256_and_si256(c, m127);
    const __m256i lo = _mm256_and_si256(idx, m15);
    const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(idx, 4), m15);
    __m256i r = _mm256_setzero_si256();
    for (int k = 0; k < 8; k++) {
        const __m256i slice = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(t + 16 * k)));
        const __m256i v = _mm256_shuffle_epi8(slice, lo);
        r = _mm256_blendv_epi8(r, v, _mm256_cmpeq_epi8(hi, _mm256_set1_epi8((char)k)));
    }
    const __m256i s = _mm256_cmpgt_epi8(_mm256_setzero_si256(), c);   // -1 where the sign bit is set
    return _mm256_sub_epi8(_mm256_xor_si256(r, s), s);
}

EXPORT void vec_e8_lookup_bulk(
    const int8_t* codes,
    const int8_t* table,
    const int32_t blocks,
    const int32_t count,
    int32_t* sums) {
    int32_t j = 0;
    for (; j + 32 <= count; j += 32) {
        __m256i w0 = _mm256_setzero_si256(), w1 = _mm256_setzero_si256(), w2 = _mm256_setzero_si256(), w3 = _mm256_setzero_si256();
        int32_t b = 0;
        while (b < blocks) {
            const int32_t end = b + 256 < blocks ? b + 256 : blocks;
            __m256i a0 = _mm256_setzero_si256(), a1 = _mm256_setzero_si256();   // 16 x int16 each
            for (; b < end; b++) {
                const __m256i c = _mm256_loadu_si256((const __m256i*)(codes + (size_t)b * count + j));
                const __m256i r = e8_lookup32_avx2(table + (size_t)b * 128, c);
                a0 = _mm256_add_epi16(a0, _mm256_cvtepi8_epi16(_mm256_castsi256_si128(r)));
                a1 = _mm256_add_epi16(a1, _mm256_cvtepi8_epi16(_mm256_extracti128_si256(r, 1)));
            }
            w0 = _mm256_add_epi32(w0, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(a0)));
            w1 = _mm256_add_epi32(w1, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(a0, 1)));
            w2 = _mm256_add_epi32(w2, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(a1)));
            w3 = _mm256_add_epi32(w3, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(a1, 1)));
        }
        _mm256_storeu_si256((__m256i*)(sums + j), w0);
        _mm256_storeu_si256((__m256i*)(sums + j + 8), w1);
        _mm256_storeu_si256((__m256i*)(sums + j + 16), w2);
        _mm256_storeu_si256((__m256i*)(sums + j + 24), w3);
    }
    for (; j < count; j++) {
        sums[j] = e8_scalar_one(codes, table, blocks, count, j);
    }
}
