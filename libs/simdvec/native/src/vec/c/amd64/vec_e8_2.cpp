/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

// E8 root code scoring for AVX-512 ("2nd tier"). vpermi2b looks up all 128 table entries, held in two 512-bit
// registers, with the low 7 bits of each code byte: one instruction per block for 32 documents.

#include <stddef.h>
#include <immintrin.h>
#include "vec.h"
#include "vec_e8.h"

EXPORT void vec_dote8_bulk_2(
    const int8_t* codes,
    const int8_t* table,
    const int32_t blocks,
    const int32_t table_size,
    const int32_t count,
    f32_t* results
) {
    const __m256i one = _mm256_set1_epi8(1);
    int32_t d = 0;
    for (; d + 32 <= count; d += 32) {
        __m512i sum0 = _mm512_setzero_si512(), sum1 = _mm512_setzero_si512();
        for (int32_t b0 = 0; b0 < blocks; b0 += E8_FLUSH_BLOCKS) {
            const int32_t b1 = blocks - b0 < E8_FLUSH_BLOCKS ? blocks : b0 + E8_FLUSH_BLOCKS;
            __m512i acc = _mm512_setzero_si512();
            for (int32_t b = b0; b < b1; b++) {
                const int8_t* t = table + (size_t) b * table_size;
                const __m256i c = _mm256_loadu_si256((const __m256i*) (codes + (size_t) b * count + d));
                const __m512i lut = _mm512_permutex2var_epi8(
                    _mm512_loadu_si512((const void*) t),
                    _mm512_zextsi256_si512(c),
                    _mm512_loadu_si512((const void*) (t + 64))
                );
                const __m256i v = _mm256_sign_epi8(_mm512_castsi512_si256(lut), _mm256_or_si256(c, one));
                acc = _mm512_add_epi16(acc, _mm512_cvtepi8_epi16(v));
            }
            sum0 = _mm512_add_epi32(sum0, _mm512_cvtepi16_epi32(_mm512_castsi512_si256(acc)));
            sum1 = _mm512_add_epi32(sum1, _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(acc, 1)));
        }
        _mm512_storeu_ps(results + d, _mm512_cvtepi32_ps(sum0));
        _mm512_storeu_ps(results + d + 16, _mm512_cvtepi32_ps(sum1));
    }
    e8_dot_scalar(codes, table, blocks, table_size, count, d, results);
}
