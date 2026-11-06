/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

// E8 lattice block codes (experimental), tier 2 (AVX-512 with VBMI, Ice Lake and later). See aarch64/vec_e8_1.cpp
// for the layout and contract. The 128-entry table of a block sits in two zmm registers and one vpermi2b looks up
// 64 documents; with the bulk of 32 used by DiskBBQ, a ymm-masked half is used.

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

EXPORT void vec_e8_lookup_bulk_2(
    const int8_t* codes,
    const int8_t* table,
    const int32_t blocks,
    const int32_t count,
    int32_t* sums) {
    const __m512i m127 = _mm512_set1_epi8(127);
    int32_t j = 0;
    for (; j + 32 <= count; j += 32) {
        __m512i w0 = _mm512_setzero_si512(), w1 = _mm512_setzero_si512();
        int32_t b = 0;
        while (b < blocks) {
            const int32_t end = b + 256 < blocks ? b + 256 : blocks;
            __m512i a = _mm512_setzero_si512();                                   // 32 x int16
            for (; b < end; b++) {
                const __m512i t0 = _mm512_loadu_si512((const void*)(table + (size_t)b * 128));
                const __m512i t1 = _mm512_loadu_si512((const void*)(table + (size_t)b * 128 + 64));
                const __m256i c = _mm256_loadu_si256((const __m256i*)(codes + (size_t)b * count + j));
                const __m512i cz = _mm512_castsi256_si512(c);
                const __m512i r = _mm512_permutex2var_epi8(t0, _mm512_and_si512(cz, m127), t1);
                const __m256i r8 = _mm512_castsi512_si256(r);
                const __m256i s = _mm256_cmpgt_epi8(_mm256_setzero_si256(), c);
                const __m256i v = _mm256_sub_epi8(_mm256_xor_si256(r8, s), s);
                a = _mm512_add_epi16(a, _mm512_cvtepi8_epi16(v));
            }
            w0 = _mm512_add_epi32(w0, _mm512_cvtepi16_epi32(_mm512_castsi512_si256(a)));
            w1 = _mm512_add_epi32(w1, _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(a, 1)));
        }
        _mm512_storeu_si512((void*)(sums + j), w0);
        _mm512_storeu_si512((void*)(sums + j + 16), w1);
    }
    for (; j < count; j++) {
        sums[j] = e8_scalar_one(codes, table, blocks, count, j);
    }
}
