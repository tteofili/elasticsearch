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
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorShuffle;
import jdk.incubator.vector.VectorSpecies;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.FilterIndexInput;
import org.apache.lucene.store.IndexInput;
import org.elasticsearch.lucene.store.IndexInputUtils;
import org.elasticsearch.lucene.store.MemorySegmentAccessInputAccess;
import org.elasticsearch.lucene.store.SegmentBufferScratch;
import org.elasticsearch.simdvec.BulkMask;
import org.elasticsearch.simdvec.E8Lattice;
import org.elasticsearch.simdvec.E8VectorsScorer;
import org.elasticsearch.simdvec.SimdVecLibrary;

import java.io.IOException;
import java.lang.foreign.MemorySegment;

/**
 * Panama (Java Vector API) and native kernels for {@link E8VectorsScorer}. The native kernel reads the transposed codes
 * of a bulk as a {@link MemorySegment} slice of the input; the Panama kernel copies them to the heap first (see
 * {@link PanamaE8VectorsScorer#lookupBulk}). The scalar fallback lives in {@link E8VectorsScorer} itself.
 */
public final class E8Kernels {

    private static final SimdVecLibrary NATIVE = SimdVecLibrary.instance().orElse(null);

    private E8Kernels() {}

    /**
     * Returns a Panama or native scorer for {@code in}, or {@code null} if the input cannot be read as memory segments.
     */
    public static E8VectorsScorer create(IndexInput in, int blocks, int bulkSize, boolean preferNative) {
        IndexInput unwrapped = MemorySegmentAccessInputAccess.unwrap(FilterIndexInput.unwrapOnlyTest(in));
        if (IndexInputUtils.canUseSegmentSlices(unwrapped) == false) {
            return null;
        }
        if (preferNative && NATIVE != null && JdkFeatures.SUPPORTS_HEAP_SEGMENTS) {
            return new NativeE8VectorsScorer(unwrapped, blocks, bulkSize);
        }
        return new PanamaE8VectorsScorer(unwrapped, blocks, bulkSize);
    }

    /** Applies the score corrections with the Vector API (both kernels run where the Vector API is available). */
    abstract static class VectorizedCorrections extends E8VectorsScorer {
        VectorizedCorrections(IndexInput in, int blocks, int bulkSize) {
            super(in, blocks, bulkSize);
        }

        @Override
        protected float applyCorrections(
            float tableScale,
            float queryAdditionalCorrection,
            VectorSimilarityFunction similarityFunction,
            int mask,
            float[] scores,
            int count
        ) {
            if (similarityFunction == VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT) {
                return super.applyCorrections(tableScale, queryAdditionalCorrection, similarityFunction, mask, scores, count);
            }
            final boolean euclidean = similarityFunction == VectorSimilarityFunction.EUCLIDEAN;
            int j = 0;
            final int bound = F.loopBound(count);
            for (; j < bound; j += F.length()) {
                // same operation order as E8VectorsScorer#applyCorrection, so both give bit-identical scores
                FloatVector est = FloatVector.fromArray(F, alphas, j)
                    .mul(tableScale)
                    .mul((FloatVector) IntVector.fromArray(I, sums, j).convert(VectorOperators.I2F, 0));
                FloatVector add = FloatVector.fromArray(F, additional, j);
                FloatVector score;
                if (euclidean) {
                    FloatVector d = FloatVector.broadcast(F, queryAdditionalCorrection).add(add).sub(est.mul(2f));
                    score = FloatVector.broadcast(F, 1f).div(d.add(1f)).max(0f);
                } else {
                    score = est.add(queryAdditionalCorrection).add(add).add(1f).div(2f).max(0f);
                }
                score.intoArray(scores, j);
            }
            for (; j < count; j++) {
                scores[j] = applyCorrection(alphas[j] * tableScale * sums[j], queryAdditionalCorrection, additional[j], similarityFunction);
            }
            float maxScore = Float.NEGATIVE_INFINITY;
            for (int k = 0; k < count; k++) {
                if (BulkMask.isSelected(mask, k)) {
                    maxScore = Math.max(maxScore, scores[k]);
                } else {
                    scores[k] = 0f;
                }
            }
            return maxScore;
        }
    }

    static final class NativeE8VectorsScorer extends VectorizedCorrections {
        private final SegmentBufferScratch scratch = new SegmentBufferScratch();
        private byte[] cachedTable;
        private MemorySegment cachedTableSegment;
        private int[] cachedSums;
        private MemorySegment cachedSumsSegment;

        NativeE8VectorsScorer(IndexInput in, int blocks, int bulkSize) {
            super(in, blocks, bulkSize);
        }

        @Override
        public Kernel kernel() {
            return Kernel.NATIVE;
        }

        @Override
        public void lookupBulk(byte[] table8, int count, int[] sums) throws IOException {
            if (table8 != cachedTable) {
                cachedTable = table8;
                cachedTableSegment = MemorySegment.ofArray(table8);
            }
            if (sums != cachedSums) {
                cachedSums = sums;
                cachedSumsSegment = MemorySegment.ofArray(sums);
            }
            final MemorySegment tableSegment = cachedTableSegment;
            final MemorySegment sumsSegment = cachedSumsSegment;
            IndexInputUtils.withVoidSlice(
                in,
                (long) blocks * count,
                scratch,
                codes -> NATIVE.e8LookupBulk(codes, tableSegment, blocks, count, sumsSegment)
            );
        }
    }

    static final class PanamaE8VectorsScorer extends VectorizedCorrections {
        private final byte[] codes;

        PanamaE8VectorsScorer(IndexInput in, int blocks, int bulkSize) {
            super(in, blocks, bulkSize);
            this.codes = new byte[blocks * bulkSize];
        }

        @Override
        public Kernel kernel() {
            return Kernel.PANAMA;
        }

        @Override
        public void lookupBulk(byte[] table8, int count, int[] sums) throws IOException {
            // Copying the bulk to the heap (one memcpy of blocks * count bytes) lets C2 drop the per-load segment checks
            // in the inner loop; reading straight from the mmapped segment measured ~1.4x slower on ARM (JDK 25).
            in.readBytes(codes, 0, blocks * count);
            panamaLookup(codes, table8, blocks, count, sums);
        }
    }

    private static final VectorSpecies<Float> F = FloatVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Integer> I = IntVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Byte> B = ByteVector.SPECIES_128;
    private static final VectorSpecies<Short> S = ShortVector.SPECIES_128;
    /** Blocks accumulated in 16-bit lanes before widening: 256 * 127 fits in a short. */
    private static final int FLUSH = 256;

    /**
     * Integer sums of a transposed bulk with the Vector API. The 120-entry table of a block is looked up as eight
     * 16-entry slices ({@code rearrange}) selected by the high bits of the pair; 16 documents per vector, one
     * 16-document chunk at a time over all blocks (chunk-major), so only two accumulators are live. Every vector is a
     * local so that C2 can keep it in a register.
     */
    public static void panamaLookup(byte[] codes, byte[] table, int blocks, int count, int[] sums) {
        int j = 0;
        for (; j + 16 <= count; j += 16) {
            int b = 0;
            while (b < blocks) {
                final int end = Math.min(blocks, b + FLUSH);
                ShortVector e0 = ShortVector.zero(S), o0 = ShortVector.zero(S);
                for (; b < end; b++) {
                    // The lookup is written inline on purpose: as a separate method C2 may compile it on its own first
                    // and then refuse to inline it here (InlineSmallCode), which boxes every vector (~1.5x slower).
                    final int t = b * E8Lattice.TABLE_STRIDE;
                    final ByteVector c = ByteVector.fromArray(B, codes, b * count + j);
                    final ByteVector idx = c.and((byte) 0x7F);
                    final VectorShuffle<Byte> sh = idx.and((byte) 0x0F).toShuffle();
                    final ByteVector hi = idx.lanewise(VectorOperators.LSHR, 4);
                    ByteVector r = ByteVector.fromArray(B, table, t).rearrange(sh);
                    r = r.blend(ByteVector.fromArray(B, table, t + 16).rearrange(sh), hi.eq((byte) 1));
                    r = r.blend(ByteVector.fromArray(B, table, t + 32).rearrange(sh), hi.eq((byte) 2));
                    r = r.blend(ByteVector.fromArray(B, table, t + 48).rearrange(sh), hi.eq((byte) 3));
                    r = r.blend(ByteVector.fromArray(B, table, t + 64).rearrange(sh), hi.eq((byte) 4));
                    r = r.blend(ByteVector.fromArray(B, table, t + 80).rearrange(sh), hi.eq((byte) 5));
                    r = r.blend(ByteVector.fromArray(B, table, t + 96).rearrange(sh), hi.eq((byte) 6));
                    r = r.blend(ByteVector.fromArray(B, table, t + 112).rearrange(sh), hi.eq((byte) 7));
                    final ByteVector sign = c.lanewise(VectorOperators.ASHR, 7);
                    r = r.lanewise(VectorOperators.XOR, sign).sub(sign);
                    ShortVector w0 = r.reinterpretAsShorts();
                    e0 = e0.add(w0.lanewise(VectorOperators.LSHL, 8).lanewise(VectorOperators.ASHR, 8));
                    o0 = o0.add(w0.lanewise(VectorOperators.ASHR, 8));
                }
                accumulate(e0, o0, sums, j, b <= FLUSH);
            }
        }
        // fewer than 16 documents left: a vector load would read past this bulk's rows
        for (; j < count; j++) {
            int s = 0;
            for (int b = 0; b < blocks; b++) {
                int c = codes[b * count + j] & 0xFF;
                int x = table[b * E8Lattice.TABLE_STRIDE + (c & 0x7F)];
                s += (c & 0x80) != 0 ? -x : x;
            }
            sums[j] = s;
        }
    }

    /** Adds 8 even-lane and 8 odd-lane short sums into {@code sums[j .. j + 16)} (sets them on the first flush). */
    private static void accumulate(ShortVector even, ShortVector odd, int[] sums, int j, boolean first) {
        for (int e = 0; e < 8; e++) {
            int ev = even.lane(e), ov = odd.lane(e);
            if (first) {
                sums[j + 2 * e] = ev;
                sums[j + 2 * e + 1] = ov;
            } else {
                sums[j + 2 * e] += ev;
                sums[j + 2 * e + 1] += ov;
            }
        }
    }
}
