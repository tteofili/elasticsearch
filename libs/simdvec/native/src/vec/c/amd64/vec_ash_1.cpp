/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

// ASH (Asymmetric Scalar Hashing) scoring for AMD64.
// Scalar baseline implementation (tier 1). AVX2/AVX-512 optimized versions can
// be added as vec_ash_2.cpp when needed.

#include <stddef.h>
#include <stdint.h>
#include <cstring>
#include <immintrin.h>
#include "vec.h"

// Scalar ipFloatBit: sum of query[j] where bit j is set in packed_plane.
// Bit order: MSB-first within each byte (big-endian bit ordering).
static inline float ipFloatBitScalar(const float* query, const uint8_t* packed_plane, int planeBytes) {
    float sum = 0;
    for (int i = 0; i < planeBytes; i++) {
        uint8_t byte_val = packed_plane[i];
        if (byte_val == 0) continue;
        const float* qBase = query + i * 8;
        if (byte_val & 0x80) sum += qBase[0];
        if (byte_val & 0x40) sum += qBase[1];
        if (byte_val & 0x20) sum += qBase[2];
        if (byte_val & 0x10) sum += qBase[3];
        if (byte_val & 0x08) sum += qBase[4];
        if (byte_val & 0x04) sum += qBase[5];
        if (byte_val & 0x02) sum += qBase[6];
        if (byte_val & 0x01) sum += qBase[7];
    }
    return sum;
}

// Decode a float16 value to float32 using F16C intrinsics (available on AVX2+).
static inline float f16_to_f32(uint16_t h) {
    __m128i vi = _mm_set1_epi16(h);
    __m128 vf = _mm_cvtph_ps(vi);
    return _mm_cvtss_f32(vf);
}

// Single-vector ASH 2-bit scoring.
// Returns: planeSum0 + 2*planeSum1 (caller subtracts 1.5*sumAllQt and applies scale/offset)
EXPORT f32_t vec_dotash2bit(const f32_t* query, const uint8_t* packed_codes, const int32_t planeBytes) {
    float planeSum0 = ipFloatBitScalar(query, packed_codes, planeBytes);
    float planeSum1 = ipFloatBitScalar(query, packed_codes + planeBytes, planeBytes);
    return planeSum0 + 2.0f * planeSum1;
}

// Bulk scoring: scores count vectors against the same query.
EXPORT void vec_dotash2bit_bulk(const f32_t* query, const uint8_t* all_codes,
                                 const int32_t packedCodeBytes, const int32_t planeBytes,
                                 const int32_t count, f32_t* results) {
    for (int v = 0; v < count; v++) {
        results[v] = vec_dotash2bit(query, all_codes + v * packedCodeBytes, planeBytes);
    }
}

// Fused bulk scoring with scale/offset application.
EXPORT void vec_dotash2bit_fused_bulk(const f32_t* query, const uint8_t* all_codes,
                                       const uint16_t* scales_f16, const uint16_t* offsets_f16,
                                       const int32_t packedCodeBytes, const int32_t planeBytes,
                                       const int32_t count,
                                       const f32_t sumAllQt, const f32_t queryDotCentroid,
                                       f32_t* results) {
    const float centerOffset = 1.5f;
    for (int v = 0; v < count; v++) {
        float dot = vec_dotash2bit(query, all_codes + v * packedCodeBytes, planeBytes);
        dot -= centerOffset * sumAllQt;

        float scale = f16_to_f32(scales_f16[v]);
        float offset = f16_to_f32(offsets_f16[v]);

        results[v] = dot * scale + queryDotCentroid + offset;
    }
}
