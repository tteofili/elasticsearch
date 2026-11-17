/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.simdvec.internal.vectorization;

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorShuffle;
import jdk.incubator.vector.VectorSpecies;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.IndexInput;
import org.elasticsearch.lucene.store.IndexInputUtils;
import org.elasticsearch.simdvec.BulkMask;
import org.elasticsearch.simdvec.E8Roots;
import org.elasticsearch.simdvec.ScoringQuery;

import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.Arrays;

/**
 * Panama scorer for E8 root codes (see {@link E8Roots}). For 16 documents at a time, the 128-entry table of a block is
 * looked up with eight 16-entry rearranges, combined by a binary tree of blends on bits 4, 5 and 6 of the codes, then
 * negated where the code's sign bit is set and widened into int16 accumulators.
 */
final class MSE8ES940OSQVectorsScorer extends MemorySegmentES940OSQVectorsScorer.MemorySegmentScorer {

    private static final VectorSpecies<Byte> BYTE_128 = ByteVector.SPECIES_128;
    private static final VectorSpecies<Short> SHORT_128 = ShortVector.SPECIES_128;
    private static final int LANES = 16;
    /** int16 accumulators absorb this many blocks of int8 entries (|entry| &lt;= 127) without overflow */
    private static final int FLUSH_BLOCKS = 256;

    private final short[] partial = new short[LANES];

    MSE8ES940OSQVectorsScorer(IndexInput in, int dimensions, int dataLength, int bulkSize) {
        super(in, dimensions, dataLength, bulkSize);
    }

    @Override
    boolean quantizeScoreBulk(ScoringQuery q, int count, float[] scores) throws IOException {
        final byte[] table = q.bytes();
        IndexInputUtils.withVoidSlice(in, (long) length * count, scratch, codes -> score(codes, table, count, scores));
        return true;
    }

    @Override
    boolean quantizeScoreBulkMasked(ScoringQuery q, int mask, float[] scores, int count) throws IOException {
        // codes are interleaved, scoring the whole bulk is as cheap as skipping
        quantizeScoreBulk(q, count, scores);
        for (int i = 0; i < count; i++) {
            if (BulkMask.isSelected(mask, i) == false) {
                scores[i] = 0f;
            }
        }
        return true;
    }

    @Override
    float scoreBulk(
        ScoringQuery q,
        float queryLowerInterval,
        float queryUpperInterval,
        int queryComponentSum,
        float queryAdditionalCorrection,
        VectorSimilarityFunction similarityFunction,
        float centroidDp,
        float[] scores,
        int bulkSize
    ) throws IOException {
        quantizeScoreBulk(q, bulkSize, scores);
        return applyCorrectionsBulk(
            queryLowerInterval,
            queryUpperInterval,
            queryComponentSum,
            queryAdditionalCorrection,
            similarityFunction,
            centroidDp,
            scores,
            bulkSize,
            bitScale(8),
            bitScale(1)
        );
    }

    private void score(MemorySegment codes, byte[] table, int count, float[] scores) {
        int d = 0;
        for (; d + LANES <= count; d += LANES) {
            scoreLanes(codes, table, count, d, scores);
        }
        for (; d < count; d++) {
            int acc = 0;
            for (int b = 0; b < length; b++) {
                final byte c = codes.get(ValueLayout.JAVA_BYTE, (long) b * count + d);
                final int v = table[b * E8Roots.TABLE_SIZE + (c & 0x7F)];
                acc += c < 0 ? -v : v;
            }
            scores[d] = acc;
        }
    }

    // Kept as one method with the lookup inline: when C2 compiles the lookup separately it may refuse to inline it and box the vectors.
    private void scoreLanes(MemorySegment codes, byte[] table, int stride, int d, float[] scores) {
        Arrays.fill(scores, d, d + LANES, 0f);
        for (int b0 = 0; b0 < length; b0 += FLUSH_BLOCKS) {
            final int b1 = Math.min(length, b0 + FLUSH_BLOCKS);
            ShortVector acc0 = ShortVector.zero(SHORT_128);
            ShortVector acc1 = ShortVector.zero(SHORT_128);
            for (int b = b0; b < b1; b++) {
                final int t = b * E8Roots.TABLE_SIZE;
                final ByteVector c = ByteVector.fromMemorySegment(BYTE_128, codes, (long) b * stride + d, ByteOrder.LITTLE_ENDIAN);
                final ByteVector idx = c.and((byte) 0x7F);
                final VectorShuffle<Byte> lo = idx.and((byte) 0x0F).toShuffle();
                final VectorMask<Byte> bit4 = idx.lanewise(VectorOperators.LSHL, 3).lt((byte) 0);
                final VectorMask<Byte> bit5 = idx.lanewise(VectorOperators.LSHL, 2).lt((byte) 0);
                final VectorMask<Byte> bit6 = idx.lanewise(VectorOperators.LSHL, 1).lt((byte) 0);
                final ByteVector v01 = ByteVector.fromArray(BYTE_128, table, t)
                    .rearrange(lo)
                    .blend(ByteVector.fromArray(BYTE_128, table, t + 16).rearrange(lo), bit4);
                final ByteVector v23 = ByteVector.fromArray(BYTE_128, table, t + 32)
                    .rearrange(lo)
                    .blend(ByteVector.fromArray(BYTE_128, table, t + 48).rearrange(lo), bit4);
                final ByteVector v45 = ByteVector.fromArray(BYTE_128, table, t + 64)
                    .rearrange(lo)
                    .blend(ByteVector.fromArray(BYTE_128, table, t + 80).rearrange(lo), bit4);
                final ByteVector v67 = ByteVector.fromArray(BYTE_128, table, t + 96)
                    .rearrange(lo)
                    .blend(ByteVector.fromArray(BYTE_128, table, t + 112).rearrange(lo), bit4);
                final ByteVector v = v01.blend(v23, bit5).blend(v45.blend(v67, bit5), bit6).lanewise(VectorOperators.NEG, c.lt((byte) 0));
                acc0 = acc0.add((ShortVector) v.convertShape(VectorOperators.B2S, SHORT_128, 0));
                acc1 = acc1.add((ShortVector) v.convertShape(VectorOperators.B2S, SHORT_128, 1));
            }
            acc0.intoArray(partial, 0);
            acc1.intoArray(partial, LANES / 2);
            for (int i = 0; i < LANES; i++) {
                scores[d + i] += partial[i];
            }
        }
    }
}
