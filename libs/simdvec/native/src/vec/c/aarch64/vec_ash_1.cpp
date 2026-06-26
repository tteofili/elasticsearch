/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

// ASH (Asymmetric Scalar Hashing) scoring for AArch64 NEON.
//
// Computes: dot = ipFloatBit(qt, plane0) + 2 * ipFloatBit(qt, plane1) - 1.5 * sumAllQt
// where ipFloatBit sums float query values where corresponding bit is set in the document.
// Bit order: MSB-first within each byte (big-endian bit ordering).
//
// Optimized using vectorized bit-to-mask expansion and batch=4 document processing
// to amortize query float loads across multiple documents.

#include <stddef.h>
#include <stdint.h>
#include <arm_neon.h>
#include "vec.h"

// Bit position constants for MSB-first extraction within a byte.
// bit_positions[0] = 0x80 corresponds to float index 0 (MSB = first float).
static const uint8x8_t BIT_POSITIONS = {0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01};

// Expands a single byte's bits into two uint32x4 masks suitable for masking float32x4 vectors.
// Returns masks in mask_lo (bits 7-4 → floats 0-3) and mask_hi (bits 3-0 → floats 4-7).
// Uses signed widening so that 0xFF (from vcgt) sign-extends to 0xFFFFFFFF.
static inline void expand_byte_to_masks(uint8_t byte_val, uint32x4_t& mask_lo, uint32x4_t& mask_hi) {
    uint8x8_t bcast = vdup_n_u8(byte_val);
    uint8x8_t isolated = vand_u8(bcast, BIT_POSITIONS);
    uint8x8_t nonzero = vcgt_u8(isolated, vdup_n_u8(0)); // 0xFF where bit set, 0x00 where not
    // Signed-widen: 0xFF → 0xFFFF → 0xFFFFFFFF; 0x00 → 0x0000 → 0x00000000
    int16x8_t wide16 = vmovl_s8(vreinterpret_s8_u8(nonzero));
    mask_lo = vreinterpretq_u32_s32(vmovl_s16(vget_low_s16(wide16)));
    mask_hi = vreinterpretq_u32_s32(vmovl_s16(vget_high_s16(wide16)));
}

// Accumulates masked query floats into accumulators.
// acc_lo += q_lo & mask_lo; acc_hi += q_hi & mask_hi
static inline void masked_accumulate(
    float32x4_t q_lo, float32x4_t q_hi,
    uint32x4_t mask_lo, uint32x4_t mask_hi,
    float32x4_t& acc_lo, float32x4_t& acc_hi
) {
    acc_lo = vaddq_f32(acc_lo, vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(q_lo), mask_lo)));
    acc_hi = vaddq_f32(acc_hi, vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(q_hi), mask_hi)));
}

// Single-vector ASH 2-bit scoring with vectorized mask generation.
// Processes both bit-planes simultaneously, sharing query float loads.
// Returns: planeSum0 + 2*planeSum1
EXPORT f32_t vec_dotash2bit(const f32_t* query, const uint8_t* packed_codes, const int32_t planeBytes) {
    const uint8_t* plane0 = packed_codes;
    const uint8_t* plane1 = packed_codes + planeBytes;

    // 2 accumulators per plane (lo/hi of the 8-float group)
    float32x4_t acc0_lo = vdupq_n_f32(0);
    float32x4_t acc0_hi = vdupq_n_f32(0);
    float32x4_t acc1_lo = vdupq_n_f32(0);
    float32x4_t acc1_hi = vdupq_n_f32(0);

    for (int i = 0; i < planeBytes; i++) {
        // Load 8 query floats (shared across both planes)
        float32x4_t q_lo = vld1q_f32(query + i * 8);
        float32x4_t q_hi = vld1q_f32(query + i * 8 + 4);

        // Plane 0: expand byte to masks and accumulate
        uint32x4_t m0_lo, m0_hi;
        expand_byte_to_masks(plane0[i], m0_lo, m0_hi);
        masked_accumulate(q_lo, q_hi, m0_lo, m0_hi, acc0_lo, acc0_hi);

        // Plane 1: expand byte to masks and accumulate (reuses q_lo, q_hi)
        uint32x4_t m1_lo, m1_hi;
        expand_byte_to_masks(plane1[i], m1_lo, m1_hi);
        masked_accumulate(q_lo, q_hi, m1_lo, m1_hi, acc1_lo, acc1_hi);
    }

    // Horizontal reduction
    float planeSum0 = vaddvq_f32(vaddq_f32(acc0_lo, acc0_hi));
    float planeSum1 = vaddvq_f32(vaddq_f32(acc1_lo, acc1_hi));
    return planeSum0 + 2.0f * planeSum1;
}

// Inner scoring for a batch of 4 documents, sharing query loads.
// Writes raw dot products (planeSum0 + 2*planeSum1) to results[0..3].
static inline void dot4_inner(
    const f32_t* query,
    const uint8_t* codes_a, const uint8_t* codes_b,
    const uint8_t* codes_c, const uint8_t* codes_d,
    const int32_t planeBytes,
    f32_t* results
) {
    // 4 documents × 2 planes × (lo + hi) = 16 accumulators
    float32x4_t a0_lo = vdupq_n_f32(0), a0_hi = vdupq_n_f32(0);
    float32x4_t a1_lo = vdupq_n_f32(0), a1_hi = vdupq_n_f32(0);
    float32x4_t b0_lo = vdupq_n_f32(0), b0_hi = vdupq_n_f32(0);
    float32x4_t b1_lo = vdupq_n_f32(0), b1_hi = vdupq_n_f32(0);
    float32x4_t c0_lo = vdupq_n_f32(0), c0_hi = vdupq_n_f32(0);
    float32x4_t c1_lo = vdupq_n_f32(0), c1_hi = vdupq_n_f32(0);
    float32x4_t d0_lo = vdupq_n_f32(0), d0_hi = vdupq_n_f32(0);
    float32x4_t d1_lo = vdupq_n_f32(0), d1_hi = vdupq_n_f32(0);

    const uint8_t* a_p0 = codes_a;
    const uint8_t* a_p1 = codes_a + planeBytes;
    const uint8_t* b_p0 = codes_b;
    const uint8_t* b_p1 = codes_b + planeBytes;
    const uint8_t* c_p0 = codes_c;
    const uint8_t* c_p1 = codes_c + planeBytes;
    const uint8_t* d_p0 = codes_d;
    const uint8_t* d_p1 = codes_d + planeBytes;

    for (int i = 0; i < planeBytes; i++) {
        // Load query floats ONCE (shared across all 4 docs × 2 planes = 8 uses)
        float32x4_t q_lo = vld1q_f32(query + i * 8);
        float32x4_t q_hi = vld1q_f32(query + i * 8 + 4);

        uint32x4_t m_lo, m_hi;

        // Doc A, Plane 0
        expand_byte_to_masks(a_p0[i], m_lo, m_hi);
        masked_accumulate(q_lo, q_hi, m_lo, m_hi, a0_lo, a0_hi);
        // Doc A, Plane 1
        expand_byte_to_masks(a_p1[i], m_lo, m_hi);
        masked_accumulate(q_lo, q_hi, m_lo, m_hi, a1_lo, a1_hi);

        // Doc B, Plane 0
        expand_byte_to_masks(b_p0[i], m_lo, m_hi);
        masked_accumulate(q_lo, q_hi, m_lo, m_hi, b0_lo, b0_hi);
        // Doc B, Plane 1
        expand_byte_to_masks(b_p1[i], m_lo, m_hi);
        masked_accumulate(q_lo, q_hi, m_lo, m_hi, b1_lo, b1_hi);

        // Doc C, Plane 0
        expand_byte_to_masks(c_p0[i], m_lo, m_hi);
        masked_accumulate(q_lo, q_hi, m_lo, m_hi, c0_lo, c0_hi);
        // Doc C, Plane 1
        expand_byte_to_masks(c_p1[i], m_lo, m_hi);
        masked_accumulate(q_lo, q_hi, m_lo, m_hi, c1_lo, c1_hi);

        // Doc D, Plane 0
        expand_byte_to_masks(d_p0[i], m_lo, m_hi);
        masked_accumulate(q_lo, q_hi, m_lo, m_hi, d0_lo, d0_hi);
        // Doc D, Plane 1
        expand_byte_to_masks(d_p1[i], m_lo, m_hi);
        masked_accumulate(q_lo, q_hi, m_lo, m_hi, d1_lo, d1_hi);
    }

    // Reduce each document: planeSum0 + 2 * planeSum1
    results[0] = vaddvq_f32(vaddq_f32(a0_lo, a0_hi)) + 2.0f * vaddvq_f32(vaddq_f32(a1_lo, a1_hi));
    results[1] = vaddvq_f32(vaddq_f32(b0_lo, b0_hi)) + 2.0f * vaddvq_f32(vaddq_f32(b1_lo, b1_hi));
    results[2] = vaddvq_f32(vaddq_f32(c0_lo, c0_hi)) + 2.0f * vaddvq_f32(vaddq_f32(c1_lo, c1_hi));
    results[3] = vaddvq_f32(vaddq_f32(d0_lo, d0_hi)) + 2.0f * vaddvq_f32(vaddq_f32(d1_lo, d1_hi));
}

// Bulk scoring: scores count vectors against the same query using batch=4.
EXPORT void vec_dotash2bit_bulk(const f32_t* query, const uint8_t* all_codes,
                                 const int32_t packedCodeBytes, const int32_t planeBytes,
                                 const int32_t count, f32_t* results) {
    int v = 0;
    // Process groups of 4 documents
    for (; v + 3 < count; v += 4) {
        dot4_inner(
            query,
            all_codes + v * packedCodeBytes,
            all_codes + (v + 1) * packedCodeBytes,
            all_codes + (v + 2) * packedCodeBytes,
            all_codes + (v + 3) * packedCodeBytes,
            planeBytes,
            results + v
        );
    }
    // Handle remaining 1-3 documents individually
    for (; v < count; v++) {
        results[v] = vec_dotash2bit(query, all_codes + v * packedCodeBytes, planeBytes);
    }
}

// Fused bulk scoring with scale/offset application.
// Computes: scores[v] = (rawDot - 1.5*sumAllQt) * scale[v] + queryDotCentroid + offset[v]
EXPORT void vec_dotash2bit_fused_bulk(const f32_t* query, const uint8_t* all_codes,
                                       const uint16_t* scales_f16, const uint16_t* offsets_f16,
                                       const int32_t packedCodeBytes, const int32_t planeBytes,
                                       const int32_t count,
                                       const f32_t sumAllQt, const f32_t queryDotCentroid,
                                       f32_t* results) {
    // Phase 1: Compute raw dot products using batch=4
    vec_dotash2bit_bulk(query, all_codes, packedCodeBytes, planeBytes, count, results);

    // Phase 2: Apply corrections with vectorized fp16 decode
    const float centerOffset = 1.5f;
    const float centeredSumAllQt = centerOffset * sumAllQt;

    int v = 0;
    // Process 4 results at a time with NEON
    float32x4_t v_centeredSum = vdupq_n_f32(centeredSumAllQt);
    float32x4_t v_qdc = vdupq_n_f32(queryDotCentroid);

    for (; v + 3 < count; v += 4) {
        // Load 4 raw dot products
        float32x4_t dots = vld1q_f32(results + v);
        // Subtract centerOffset * sumAllQt
        dots = vsubq_f32(dots, v_centeredSum);

        // Decode 4 fp16 scales and offsets
        // Load 4 uint16 values, convert via ARM fp16 extension
        float16x4_t s_f16 = vreinterpret_f16_u16(vld1_u16(scales_f16 + v));
        float16x4_t o_f16 = vreinterpret_f16_u16(vld1_u16(offsets_f16 + v));
        float32x4_t scales = vcvt_f32_f16(s_f16);
        float32x4_t offsets = vcvt_f32_f16(o_f16);

        // results[v] = dots * scales + queryDotCentroid + offsets
        float32x4_t scored = vfmaq_f32(vaddq_f32(v_qdc, offsets), dots, scales);
        vst1q_f32(results + v, scored);
    }

    // Scalar tail
    for (; v < count; v++) {
        float dot = results[v] - centeredSumAllQt;
        __fp16 s16, o16;
        __builtin_memcpy(&s16, &scales_f16[v], sizeof(__fp16));
        __builtin_memcpy(&o16, &offsets_f16[v], sizeof(__fp16));
        float scale = (float)s16;
        float offset = (float)o16;
        results[v] = dot * scale + queryDotCentroid + offset;
    }
}
