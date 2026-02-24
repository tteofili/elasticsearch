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
import jdk.incubator.vector.LongVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorOperators;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.BitUtil;
import org.apache.lucene.util.VectorUtil;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.Arrays;

import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;
import static org.apache.lucene.index.VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT;
import static org.elasticsearch.simdvec.internal.Similarities.dotProductD1Q4;

/** Panamized scorer for quantized vectors stored as a {@link MemorySegment}. Uses the first
 * {@link #scoreLength} bytes (contiguous prefix) for scoring; traversal still uses full {@link #length}. */
final class MSBitToInt4ESNextOSQVectorsScorer extends MemorySegmentESNextOSQVectorsScorer.MemorySegmentScorer {

    public static final float SLICING_RATIO = 0.9f;
    /** Byte length used for dot-product accumulation. Traversal still uses full {@link #length}. */
    private final int scoreLength;
    /** Effective dimensions for correction formulas */
    private final int effectiveDimensions;
    /** Reusable buffer for one document's sliced bytes (prefix). */
    private final byte[] docBuffer;
    /** Reusable buffer for query bytes (4 packed vectors, prefix only). */
    private final byte[] reorderedQuery;
    /** Cached segment views over docBuffer and reorderedQuery for native/dot-product calls. */
    private final MemorySegment docBufferSegment;
    private final MemorySegment reorderedQuerySegment;
    /** Precomputed bounds for scalar tail loops. */
    private final int scoreLengthLongAligned;
    private final int scoreLengthIntAligned;
    /** Reused when heap segments not supported: arena and native segment for query copy. */
    private final Arena queryArena;
    private final MemorySegment queryNativeSegment;

    MSBitToInt4ESNextOSQVectorsScorer(IndexInput in, int dimensions, int dataLength, int bulkSize, MemorySegment memorySegment) {
        super(in, dimensions, dataLength, bulkSize, memorySegment);
        this.scoreLength = (int) (length * SLICING_RATIO);
        this.effectiveDimensions = (int) (dimensions * SLICING_RATIO);
        this.docBuffer = new byte[scoreLength];
        this.reorderedQuery = new byte[scoreLength * 4];
        this.docBufferSegment = MemorySegment.ofArray(docBuffer).asSlice(0, scoreLength);
        this.reorderedQuerySegment = MemorySegment.ofArray(reorderedQuery).asSlice(0, (long) scoreLength * 4);
        this.scoreLengthLongAligned = scoreLength & -Long.BYTES;
        this.scoreLengthIntAligned = scoreLength & -Integer.BYTES;
        this.queryArena = Arena.ofConfined();
        this.queryNativeSegment = queryArena.allocate((long) scoreLength * 4, 32);
    }

    /** Fills reorderedQuery with the first scoreLength bytes of each of the 4 query lanes (contiguous copy). */
    private void fillQueryPrefix(byte[] q, byte[] reorderedQuery) {
        System.arraycopy(q, 0, reorderedQuery, 0, scoreLength);
        System.arraycopy(q, length, reorderedQuery, scoreLength, scoreLength);
        System.arraycopy(q, length * 2, reorderedQuery, scoreLength * 2, scoreLength);
        System.arraycopy(q, length * 3, reorderedQuery, scoreLength * 3, scoreLength);
    }

    /** Fills docBuf with the first scoreLength bytes of the document at the given offset (bulk copy). */
    private void fillDocPrefix(long offset, byte[] docBuf) {
        MemorySegment.copy(memorySegment, ValueLayout.JAVA_BYTE, offset, docBuf, 0, scoreLength);
    }

    @Override
    public long quantizeScore(byte[] q) throws IOException {
        assert q.length == length * 4;
        fillQueryPrefix(q, reorderedQuery);
        // 128 / 8 == 16
        if (length >= 16) {
            if (NATIVE_SUPPORTED) {
                return nativeQuantizeScore();
            } else if (PanamaESVectorUtilSupport.HAS_FAST_INTEGER_VECTORS) {
                if (PanamaESVectorUtilSupport.VECTOR_BITSIZE >= 256) {
                    return quantizeScore256();
                } else if (PanamaESVectorUtilSupport.VECTOR_BITSIZE == 128) {
                    return quantizeScore128();
                }
            }
        }
        return Long.MIN_VALUE;
    }

    private long nativeQuantizeScore() throws IOException {
        long offset = in.getFilePointer();
        fillDocPrefix(offset, docBuffer);

        final long qScore;
        if (SUPPORTS_HEAP_SEGMENTS) {
            qScore = dotProductD1Q4(docBufferSegment, reorderedQuerySegment, scoreLength);
        } else {
            MemorySegment.copy(reorderedQuery, 0, queryNativeSegment, ValueLayout.JAVA_BYTE, 0, scoreLength * 4);
            qScore = dotProductD1Q4(docBufferSegment, queryNativeSegment, scoreLength);
        }
        in.skipBytes(length);
        return qScore;
    }

    private long quantizeScore256() throws IOException {
        long subRet0 = 0;
        long subRet1 = 0;
        long subRet2 = 0;
        long subRet3 = 0;
        int i = 0;
        long offset = in.getFilePointer();
        fillDocPrefix(offset, docBuffer);
        if (scoreLength >= ByteVector.SPECIES_256.vectorByteSize() * 2) {
            int limit = ByteVector.SPECIES_256.loopBound(scoreLength);
            var sum0 = LongVector.zero(LONG_SPECIES_256);
            var sum1 = LongVector.zero(LONG_SPECIES_256);
            var sum2 = LongVector.zero(LONG_SPECIES_256);
            var sum3 = LongVector.zero(LONG_SPECIES_256);
            for (; i < limit; i += ByteVector.SPECIES_256.length()) {
                var vq0 = ByteVector.fromArray(BYTE_SPECIES_256, reorderedQuery, i).reinterpretAsLongs();
                var vq1 = ByteVector.fromArray(BYTE_SPECIES_256, reorderedQuery, i + scoreLength).reinterpretAsLongs();
                var vq2 = ByteVector.fromArray(BYTE_SPECIES_256, reorderedQuery, i + scoreLength * 2).reinterpretAsLongs();
                var vq3 = ByteVector.fromArray(BYTE_SPECIES_256, reorderedQuery, i + scoreLength * 3).reinterpretAsLongs();
                var vd = ByteVector.fromArray(BYTE_SPECIES_256, docBuffer, i).reinterpretAsLongs();
                sum0 = sum0.add(vq0.and(vd).lanewise(VectorOperators.BIT_COUNT));
                sum1 = sum1.add(vq1.and(vd).lanewise(VectorOperators.BIT_COUNT));
                sum2 = sum2.add(vq2.and(vd).lanewise(VectorOperators.BIT_COUNT));
                sum3 = sum3.add(vq3.and(vd).lanewise(VectorOperators.BIT_COUNT));
            }
            subRet0 += sum0.reduceLanes(VectorOperators.ADD);
            subRet1 += sum1.reduceLanes(VectorOperators.ADD);
            subRet2 += sum2.reduceLanes(VectorOperators.ADD);
            subRet3 += sum3.reduceLanes(VectorOperators.ADD);
        }

        if (scoreLength - i >= ByteVector.SPECIES_128.vectorByteSize()) {
            var sum0 = LongVector.zero(LONG_SPECIES_128);
            var sum1 = LongVector.zero(LONG_SPECIES_128);
            var sum2 = LongVector.zero(LONG_SPECIES_128);
            var sum3 = LongVector.zero(LONG_SPECIES_128);
            int limit = ByteVector.SPECIES_128.loopBound(scoreLength);
            for (; i < limit; i += ByteVector.SPECIES_128.length()) {
                var vq0 = ByteVector.fromArray(BYTE_SPECIES_128, reorderedQuery, i).reinterpretAsLongs();
                var vq1 = ByteVector.fromArray(BYTE_SPECIES_128, reorderedQuery, i + scoreLength).reinterpretAsLongs();
                var vq2 = ByteVector.fromArray(BYTE_SPECIES_128, reorderedQuery, i + scoreLength * 2).reinterpretAsLongs();
                var vq3 = ByteVector.fromArray(BYTE_SPECIES_128, reorderedQuery, i + scoreLength * 3).reinterpretAsLongs();
                var vd = ByteVector.fromArray(BYTE_SPECIES_128, docBuffer, i).reinterpretAsLongs();
                sum0 = sum0.add(vq0.and(vd).lanewise(VectorOperators.BIT_COUNT));
                sum1 = sum1.add(vq1.and(vd).lanewise(VectorOperators.BIT_COUNT));
                sum2 = sum2.add(vq2.and(vd).lanewise(VectorOperators.BIT_COUNT));
                sum3 = sum3.add(vq3.and(vd).lanewise(VectorOperators.BIT_COUNT));
            }
            subRet0 += sum0.reduceLanes(VectorOperators.ADD);
            subRet1 += sum1.reduceLanes(VectorOperators.ADD);
            subRet2 += sum2.reduceLanes(VectorOperators.ADD);
            subRet3 += sum3.reduceLanes(VectorOperators.ADD);
        }
        // process scalar tail (only up to scoreLength)
        for (final int upperBound = scoreLengthLongAligned; i < upperBound; i += Long.BYTES) {
            final long value = (Long) BitUtil.VH_LE_LONG.get(docBuffer, i);
            subRet0 += Long.bitCount((Long) BitUtil.VH_LE_LONG.get(reorderedQuery, i) & value);
            subRet1 += Long.bitCount((Long) BitUtil.VH_LE_LONG.get(reorderedQuery, i + scoreLength) & value);
            subRet2 += Long.bitCount((Long) BitUtil.VH_LE_LONG.get(reorderedQuery, i + 2 * scoreLength) & value);
            subRet3 += Long.bitCount((Long) BitUtil.VH_LE_LONG.get(reorderedQuery, i + 3 * scoreLength) & value);
        }
        for (final int upperBound = scoreLengthIntAligned; i < upperBound; i += Integer.BYTES) {
            final int value = (Integer) BitUtil.VH_LE_INT.get(docBuffer, i);
            subRet0 += Integer.bitCount((Integer) BitUtil.VH_LE_INT.get(reorderedQuery, i) & value);
            subRet1 += Integer.bitCount((Integer) BitUtil.VH_LE_INT.get(reorderedQuery, i + scoreLength) & value);
            subRet2 += Integer.bitCount((Integer) BitUtil.VH_LE_INT.get(reorderedQuery, i + 2 * scoreLength) & value);
            subRet3 += Integer.bitCount((Integer) BitUtil.VH_LE_INT.get(reorderedQuery, i + 3 * scoreLength) & value);
        }
        for (; i < scoreLength; i++) {
            int dValue = docBuffer[i] & 0xFF;
            subRet0 += Integer.bitCount((reorderedQuery[i] & dValue) & 0xFF);
            subRet1 += Integer.bitCount((reorderedQuery[i + scoreLength] & dValue) & 0xFF);
            subRet2 += Integer.bitCount((reorderedQuery[i + 2 * scoreLength] & dValue) & 0xFF);
            subRet3 += Integer.bitCount((reorderedQuery[i + 3 * scoreLength] & dValue) & 0xFF);
        }
        in.skipBytes(length);
        return subRet0 + (subRet1 << 1) + (subRet2 << 2) + (subRet3 << 3);
    }

    private long quantizeScore128() throws IOException {
        long subRet0 = 0;
        long subRet1 = 0;
        long subRet2 = 0;
        long subRet3 = 0;
        int i = 0;
        long offset = in.getFilePointer();
        fillDocPrefix(offset, docBuffer);

        var sum0 = IntVector.zero(INT_SPECIES_128);
        var sum1 = IntVector.zero(INT_SPECIES_128);
        var sum2 = IntVector.zero(INT_SPECIES_128);
        var sum3 = IntVector.zero(INT_SPECIES_128);
        int limit = ByteVector.SPECIES_128.loopBound(scoreLength);
        for (; i < limit; i += ByteVector.SPECIES_128.length()) {
            var vd = ByteVector.fromArray(BYTE_SPECIES_128, docBuffer, i).reinterpretAsInts();
            var vq0 = ByteVector.fromArray(BYTE_SPECIES_128, reorderedQuery, i).reinterpretAsInts();
            var vq1 = ByteVector.fromArray(BYTE_SPECIES_128, reorderedQuery, i + scoreLength).reinterpretAsInts();
            var vq2 = ByteVector.fromArray(BYTE_SPECIES_128, reorderedQuery, i + scoreLength * 2).reinterpretAsInts();
            var vq3 = ByteVector.fromArray(BYTE_SPECIES_128, reorderedQuery, i + scoreLength * 3).reinterpretAsInts();
            sum0 = sum0.add(vd.and(vq0).lanewise(VectorOperators.BIT_COUNT));
            sum1 = sum1.add(vd.and(vq1).lanewise(VectorOperators.BIT_COUNT));
            sum2 = sum2.add(vd.and(vq2).lanewise(VectorOperators.BIT_COUNT));
            sum3 = sum3.add(vd.and(vq3).lanewise(VectorOperators.BIT_COUNT));
        }
        subRet0 += sum0.reduceLanes(VectorOperators.ADD);
        subRet1 += sum1.reduceLanes(VectorOperators.ADD);
        subRet2 += sum2.reduceLanes(VectorOperators.ADD);
        subRet3 += sum3.reduceLanes(VectorOperators.ADD);
        // process scalar tail (only up to scoreLength)
        for (final int upperBound = scoreLengthLongAligned; i < upperBound; i += Long.BYTES) {
            final long value = (Long) BitUtil.VH_LE_LONG.get(docBuffer, i);
            subRet0 += Long.bitCount((Long) BitUtil.VH_LE_LONG.get(reorderedQuery, i) & value);
            subRet1 += Long.bitCount((Long) BitUtil.VH_LE_LONG.get(reorderedQuery, i + scoreLength) & value);
            subRet2 += Long.bitCount((Long) BitUtil.VH_LE_LONG.get(reorderedQuery, i + 2 * scoreLength) & value);
            subRet3 += Long.bitCount((Long) BitUtil.VH_LE_LONG.get(reorderedQuery, i + 3 * scoreLength) & value);
        }
        for (final int upperBound = scoreLengthIntAligned; i < upperBound; i += Integer.BYTES) {
            final int value = (Integer) BitUtil.VH_LE_INT.get(docBuffer, i);
            subRet0 += Integer.bitCount((Integer) BitUtil.VH_LE_INT.get(reorderedQuery, i) & value);
            subRet1 += Integer.bitCount((Integer) BitUtil.VH_LE_INT.get(reorderedQuery, i + scoreLength) & value);
            subRet2 += Integer.bitCount((Integer) BitUtil.VH_LE_INT.get(reorderedQuery, i + 2 * scoreLength) & value);
            subRet3 += Integer.bitCount((Integer) BitUtil.VH_LE_INT.get(reorderedQuery, i + 3 * scoreLength) & value);
        }
        for (; i < scoreLength; i++) {
            int dValue = docBuffer[i] & 0xFF;
            subRet0 += Integer.bitCount((reorderedQuery[i] & dValue) & 0xFF);
            subRet1 += Integer.bitCount((reorderedQuery[i + scoreLength] & dValue) & 0xFF);
            subRet2 += Integer.bitCount((reorderedQuery[i + 2 * scoreLength] & dValue) & 0xFF);
            subRet3 += Integer.bitCount((reorderedQuery[i + 3 * scoreLength] & dValue) & 0xFF);
        }
        in.skipBytes(length);
        return subRet0 + (subRet1 << 1) + (subRet2 << 2) + (subRet3 << 3);
    }

    @Override
    public boolean quantizeScoreBulk(byte[] q, int count, float[] scores) throws IOException {
        assert q.length == length * 4;
        fillQueryPrefix(q, reorderedQuery);
        // 128 / 8 == 16
        if (length >= 16) {
            if (NATIVE_SUPPORTED) {
                if (SUPPORTS_HEAP_SEGMENTS) {
                    var scoresSegment = MemorySegment.ofArray(scores);
                    nativeQuantizeScoreBulk(count, scoresSegment);
                } else {
                    try (var arena = Arena.ofConfined()) {
                        var scoresSegment = arena.allocate((long) scores.length * Float.BYTES, 32);
                        nativeQuantizeScoreBulk(count, scoresSegment);
                        MemorySegment.copy(scoresSegment, ValueLayout.JAVA_FLOAT, 0, scores, 0, scores.length);
                    }
                }
                return true;
            } else if (PanamaESVectorUtilSupport.HAS_FAST_INTEGER_VECTORS) {
                if (PanamaESVectorUtilSupport.VECTOR_BITSIZE >= 256) {
                    quantizeScore256Bulk(count, scores);
                    return true;
                } else if (PanamaESVectorUtilSupport.VECTOR_BITSIZE == 128) {
                    quantizeScore128Bulk(count, scores);
                    return true;
                }
            }
        }
        return false;
    }

    private void nativeQuantizeScoreBulk(int count, MemorySegment scoresSegment) throws IOException {
        long initialOffset = in.getFilePointer();
        var datasetLengthInBytes = (long) length * count;
        for (int i = 0; i < count; i++) {
            fillDocPrefix(initialOffset + (long) i * length, docBuffer);
            long qScore = dotProductD1Q4(docBufferSegment, reorderedQuerySegment, scoreLength);
            scoresSegment.set(ValueLayout.JAVA_FLOAT, (long) i * Float.BYTES, (float) qScore);
        }
        in.skipBytes(datasetLengthInBytes);
    }

    private void quantizeScore128Bulk(int count, float[] scores) throws IOException {
        for (int iter = 0; iter < count; iter++) {
            long subRet0 = 0;
            long subRet1 = 0;
            long subRet2 = 0;
            long subRet3 = 0;
            int i = 0;
            long offset = in.getFilePointer();
            fillDocPrefix(offset, docBuffer);

            var sum0 = IntVector.zero(INT_SPECIES_128);
            var sum1 = IntVector.zero(INT_SPECIES_128);
            var sum2 = IntVector.zero(INT_SPECIES_128);
            var sum3 = IntVector.zero(INT_SPECIES_128);
            int limit = ByteVector.SPECIES_128.loopBound(scoreLength);
            for (; i < limit; i += ByteVector.SPECIES_128.length()) {
                var vd = ByteVector.fromArray(BYTE_SPECIES_128, docBuffer, i).reinterpretAsInts();
                var vq0 = ByteVector.fromArray(BYTE_SPECIES_128, reorderedQuery, i).reinterpretAsInts();
                var vq1 = ByteVector.fromArray(BYTE_SPECIES_128, reorderedQuery, i + scoreLength).reinterpretAsInts();
                var vq2 = ByteVector.fromArray(BYTE_SPECIES_128, reorderedQuery, i + scoreLength * 2).reinterpretAsInts();
                var vq3 = ByteVector.fromArray(BYTE_SPECIES_128, reorderedQuery, i + scoreLength * 3).reinterpretAsInts();
                sum0 = sum0.add(vd.and(vq0).lanewise(VectorOperators.BIT_COUNT));
                sum1 = sum1.add(vd.and(vq1).lanewise(VectorOperators.BIT_COUNT));
                sum2 = sum2.add(vd.and(vq2).lanewise(VectorOperators.BIT_COUNT));
                sum3 = sum3.add(vd.and(vq3).lanewise(VectorOperators.BIT_COUNT));
            }
            subRet0 += sum0.reduceLanes(VectorOperators.ADD);
            subRet1 += sum1.reduceLanes(VectorOperators.ADD);
            subRet2 += sum2.reduceLanes(VectorOperators.ADD);
            subRet3 += sum3.reduceLanes(VectorOperators.ADD);
            // process scalar tail (only up to scoreLength); then skip rest of vector
            for (final int upperBound = scoreLengthLongAligned; i < upperBound; i += Long.BYTES) {
                final long value = (Long) BitUtil.VH_LE_LONG.get(docBuffer, i);
                subRet0 += Long.bitCount((Long) BitUtil.VH_LE_LONG.get(reorderedQuery, i) & value);
                subRet1 += Long.bitCount((Long) BitUtil.VH_LE_LONG.get(reorderedQuery, i + scoreLength) & value);
                subRet2 += Long.bitCount((Long) BitUtil.VH_LE_LONG.get(reorderedQuery, i + 2 * scoreLength) & value);
                subRet3 += Long.bitCount((Long) BitUtil.VH_LE_LONG.get(reorderedQuery, i + 3 * scoreLength) & value);
            }
            for (final int upperBound = scoreLengthIntAligned; i < upperBound; i += Integer.BYTES) {
                final int value = (Integer) BitUtil.VH_LE_INT.get(docBuffer, i);
                subRet0 += Integer.bitCount((Integer) BitUtil.VH_LE_INT.get(reorderedQuery, i) & value);
                subRet1 += Integer.bitCount((Integer) BitUtil.VH_LE_INT.get(reorderedQuery, i + scoreLength) & value);
                subRet2 += Integer.bitCount((Integer) BitUtil.VH_LE_INT.get(reorderedQuery, i + 2 * scoreLength) & value);
                subRet3 += Integer.bitCount((Integer) BitUtil.VH_LE_INT.get(reorderedQuery, i + 3 * scoreLength) & value);
            }
            for (; i < scoreLength; i++) {
                int dValue = docBuffer[i] & 0xFF;
                subRet0 += Integer.bitCount((reorderedQuery[i] & dValue) & 0xFF);
                subRet1 += Integer.bitCount((reorderedQuery[i + scoreLength] & dValue) & 0xFF);
                subRet2 += Integer.bitCount((reorderedQuery[i + 2 * scoreLength] & dValue) & 0xFF);
                subRet3 += Integer.bitCount((reorderedQuery[i + 3 * scoreLength] & dValue) & 0xFF);
            }
            scores[iter] = subRet0 + (subRet1 << 1) + (subRet2 << 2) + (subRet3 << 3);
            in.skipBytes(length);
        }
    }

    private void quantizeScore256Bulk(int count, float[] scores) throws IOException {
        for (int iter = 0; iter < count; iter++) {
            long subRet0 = 0;
            long subRet1 = 0;
            long subRet2 = 0;
            long subRet3 = 0;
            int i = 0;
            long offset = in.getFilePointer();
            fillDocPrefix(offset, docBuffer);
            if (scoreLength >= ByteVector.SPECIES_256.vectorByteSize() * 2) {
                int limit = ByteVector.SPECIES_256.loopBound(scoreLength);
                var sum0 = LongVector.zero(LONG_SPECIES_256);
                var sum1 = LongVector.zero(LONG_SPECIES_256);
                var sum2 = LongVector.zero(LONG_SPECIES_256);
                var sum3 = LongVector.zero(LONG_SPECIES_256);
                for (; i < limit; i += ByteVector.SPECIES_256.length()) {
                    var vq0 = ByteVector.fromArray(BYTE_SPECIES_256, reorderedQuery, i).reinterpretAsLongs();
                    var vq1 = ByteVector.fromArray(BYTE_SPECIES_256, reorderedQuery, i + scoreLength).reinterpretAsLongs();
                    var vq2 = ByteVector.fromArray(BYTE_SPECIES_256, reorderedQuery, i + scoreLength * 2).reinterpretAsLongs();
                    var vq3 = ByteVector.fromArray(BYTE_SPECIES_256, reorderedQuery, i + scoreLength * 3).reinterpretAsLongs();
                    var vd = ByteVector.fromArray(BYTE_SPECIES_256, docBuffer, i).reinterpretAsLongs();
                    sum0 = sum0.add(vq0.and(vd).lanewise(VectorOperators.BIT_COUNT));
                    sum1 = sum1.add(vq1.and(vd).lanewise(VectorOperators.BIT_COUNT));
                    sum2 = sum2.add(vq2.and(vd).lanewise(VectorOperators.BIT_COUNT));
                    sum3 = sum3.add(vq3.and(vd).lanewise(VectorOperators.BIT_COUNT));
                }
                subRet0 += sum0.reduceLanes(VectorOperators.ADD);
                subRet1 += sum1.reduceLanes(VectorOperators.ADD);
                subRet2 += sum2.reduceLanes(VectorOperators.ADD);
                subRet3 += sum3.reduceLanes(VectorOperators.ADD);
            }

            if (scoreLength - i >= ByteVector.SPECIES_128.vectorByteSize()) {
                var sum0 = LongVector.zero(LONG_SPECIES_128);
                var sum1 = LongVector.zero(LONG_SPECIES_128);
                var sum2 = LongVector.zero(LONG_SPECIES_128);
                var sum3 = LongVector.zero(LONG_SPECIES_128);
                int limit = ByteVector.SPECIES_128.loopBound(scoreLength);
                for (; i < limit; i += ByteVector.SPECIES_128.length()) {
                    var vq0 = ByteVector.fromArray(BYTE_SPECIES_128, reorderedQuery, i).reinterpretAsLongs();
                    var vq1 = ByteVector.fromArray(BYTE_SPECIES_128, reorderedQuery, i + scoreLength).reinterpretAsLongs();
                    var vq2 = ByteVector.fromArray(BYTE_SPECIES_128, reorderedQuery, i + scoreLength * 2).reinterpretAsLongs();
                    var vq3 = ByteVector.fromArray(BYTE_SPECIES_128, reorderedQuery, i + scoreLength * 3).reinterpretAsLongs();
                    var vd = ByteVector.fromArray(BYTE_SPECIES_128, docBuffer, i).reinterpretAsLongs();
                    sum0 = sum0.add(vq0.and(vd).lanewise(VectorOperators.BIT_COUNT));
                    sum1 = sum1.add(vq1.and(vd).lanewise(VectorOperators.BIT_COUNT));
                    sum2 = sum2.add(vq2.and(vd).lanewise(VectorOperators.BIT_COUNT));
                    sum3 = sum3.add(vq3.and(vd).lanewise(VectorOperators.BIT_COUNT));
                }
                subRet0 += sum0.reduceLanes(VectorOperators.ADD);
                subRet1 += sum1.reduceLanes(VectorOperators.ADD);
                subRet2 += sum2.reduceLanes(VectorOperators.ADD);
                subRet3 += sum3.reduceLanes(VectorOperators.ADD);
            }
            // process scalar tail (only up to scoreLength); then skip rest of vector
            for (final int upperBound = scoreLengthLongAligned; i < upperBound; i += Long.BYTES) {
                final long value = (Long) BitUtil.VH_LE_LONG.get(docBuffer, i);
                subRet0 += Long.bitCount((Long) BitUtil.VH_LE_LONG.get(reorderedQuery, i) & value);
                subRet1 += Long.bitCount((Long) BitUtil.VH_LE_LONG.get(reorderedQuery, i + scoreLength) & value);
                subRet2 += Long.bitCount((Long) BitUtil.VH_LE_LONG.get(reorderedQuery, i + 2 * scoreLength) & value);
                subRet3 += Long.bitCount((Long) BitUtil.VH_LE_LONG.get(reorderedQuery, i + 3 * scoreLength) & value);
            }
            for (final int upperBound = scoreLengthIntAligned; i < upperBound; i += Integer.BYTES) {
                final int value = (Integer) BitUtil.VH_LE_INT.get(docBuffer, i);
                subRet0 += Integer.bitCount((Integer) BitUtil.VH_LE_INT.get(reorderedQuery, i) & value);
                subRet1 += Integer.bitCount((Integer) BitUtil.VH_LE_INT.get(reorderedQuery, i + scoreLength) & value);
                subRet2 += Integer.bitCount((Integer) BitUtil.VH_LE_INT.get(reorderedQuery, i + 2 * scoreLength) & value);
                subRet3 += Integer.bitCount((Integer) BitUtil.VH_LE_INT.get(reorderedQuery, i + 3 * scoreLength) & value);
            }
            for (; i < scoreLength; i++) {
                int dValue = docBuffer[i] & 0xFF;
                subRet0 += Integer.bitCount((reorderedQuery[i] & dValue) & 0xFF);
                subRet1 += Integer.bitCount((reorderedQuery[i + scoreLength] & dValue) & 0xFF);
                subRet2 += Integer.bitCount((reorderedQuery[i + 2 * scoreLength] & dValue) & 0xFF);
                subRet3 += Integer.bitCount((reorderedQuery[i + 3 * scoreLength] & dValue) & 0xFF);
            }
            scores[iter] = subRet0 + (subRet1 << 1) + (subRet2 << 2) + (subRet3 << 3);
            in.skipBytes(length);
        }
    }

    @Override
    public float scoreBulk(
        byte[] q,
        float queryLowerInterval,
        float queryUpperInterval,
        int queryComponentSum,
        float queryAdditionalCorrection,
        VectorSimilarityFunction similarityFunction,
        float centroidDp,
        float[] scores,
        int bulkSize
    ) throws IOException {
        queryAdditionalCorrection *= SLICING_RATIO;
        centroidDp *= SLICING_RATIO;
        queryComponentSum = Math.round(queryComponentSum * SLICING_RATIO);

        assert q.length == length * 4;
        fillQueryPrefix(q, reorderedQuery);
        // 128 / 8 == 16
        if (length >= 16) {
            if (PanamaESVectorUtilSupport.HAS_FAST_INTEGER_VECTORS) {
                if (NATIVE_SUPPORTED) {
                    if (SUPPORTS_HEAP_SEGMENTS) {
                        var scoresSegment = MemorySegment.ofArray(scores);
                        nativeQuantizeScoreBulk(bulkSize, scoresSegment);
                        return nativeApplyCorrectionsBulk(
                            queryLowerInterval,
                            queryUpperInterval,
                            queryComponentSum,
                            queryAdditionalCorrection,
                            similarityFunction,
                            centroidDp,
                            scoresSegment,
                            bulkSize
                        );
                    } else {
                        try (var arena = Arena.ofConfined()) {
                            var scoresSegment = arena.allocate((long) scores.length * Float.BYTES, 32);
                            nativeQuantizeScoreBulk(bulkSize, scoresSegment);
                            var maxScore = nativeApplyCorrectionsBulk(
                                queryLowerInterval,
                                queryUpperInterval,
                                queryComponentSum,
                                queryAdditionalCorrection,
                                similarityFunction,
                                centroidDp,
                                scoresSegment,
                                bulkSize
                            );
                            MemorySegment.copy(scoresSegment, ValueLayout.JAVA_FLOAT, 0, scores, 0, scores.length);
                            return maxScore;
                        }
                    }
                } else if (PanamaESVectorUtilSupport.VECTOR_BITSIZE >= 256) {
                    quantizeScore256Bulk(bulkSize, scores);
                    return applyCorrections256Bulk(
                        queryLowerInterval,
                        queryUpperInterval,
                        queryComponentSum,
                        queryAdditionalCorrection,
                        similarityFunction,
                        centroidDp,
                        scores,
                        bulkSize
                    );
                } else if (PanamaESVectorUtilSupport.VECTOR_BITSIZE == 128) {
                    quantizeScore128Bulk(bulkSize, scores);
                    return applyCorrections128Bulk(
                        queryLowerInterval,
                        queryUpperInterval,
                        queryComponentSum,
                        queryAdditionalCorrection,
                        similarityFunction,
                        centroidDp,
                        scores,
                        bulkSize
                    );
                }
            }
        }
        return Float.NEGATIVE_INFINITY;
    }

    private float nativeApplyCorrectionsBulk(
        float queryLowerInterval,
        float queryUpperInterval,
        int queryComponentSum,
        float queryAdditionalCorrection,
        VectorSimilarityFunction similarityFunction,
        float centroidDp,
        MemorySegment scoresSegment,
        int bulkSize
    ) throws IOException {
        long offset = in.getFilePointer();

        final float maxScore = ScoreCorrections.nativeApplyCorrectionsBulk(
            similarityFunction,
            memorySegment.asSlice(offset),
            bulkSize,
            effectiveDimensions,
            queryLowerInterval,
            queryUpperInterval,
            queryComponentSum,
            queryAdditionalCorrection,
            FOUR_BIT_SCALE,
            ONE_BIT_SCALE,
            centroidDp,
            scoresSegment
        );
        in.seek(offset + 16L * bulkSize);
        return maxScore;
    }

    private float applyCorrections128Bulk(
        float queryLowerInterval,
        float queryUpperInterval,
        int queryComponentSum,
        float queryAdditionalCorrection,
        VectorSimilarityFunction similarityFunction,
        float centroidDp,
        float[] scores,
        int bulkSize
    ) throws IOException {
        int limit = FLOAT_SPECIES_128.loopBound(bulkSize);
        int i = 0;
        long offset = in.getFilePointer();
        float ay = queryLowerInterval;
        float ly = (queryUpperInterval - ay) * FOUR_BIT_SCALE;
        float y1 = queryComponentSum;
        float maxScore = Float.NEGATIVE_INFINITY;
        for (; i < limit; i += FLOAT_SPECIES_128.length()) {
            var ax = FloatVector.fromMemorySegment(FLOAT_SPECIES_128, memorySegment, offset + i * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
            var lx = FloatVector.fromMemorySegment(
                FLOAT_SPECIES_128,
                memorySegment,
                offset + 4L * bulkSize + i * Float.BYTES,
                ByteOrder.LITTLE_ENDIAN
            ).sub(ax);
            var targetComponentSums = IntVector.fromMemorySegment(
                INT_SPECIES_128,
                memorySegment,
                offset + 8L * bulkSize + i * Integer.BYTES,
                ByteOrder.LITTLE_ENDIAN
            ).convert(VectorOperators.I2F, 0);
            var additionalCorrections = FloatVector.fromMemorySegment(
                FLOAT_SPECIES_128,
                memorySegment,
                offset + 12L * bulkSize + i * Float.BYTES,
                ByteOrder.LITTLE_ENDIAN
            ).mul(SLICING_RATIO);
            var qcDist = FloatVector.fromArray(FLOAT_SPECIES_128, scores, i);
            // ax * ay * effectiveDimensions + ay * lx * (float) targetComponentSum + ax * ly * y1 + lx * ly *
            // qcDist;
            var res1 = ax.mul(ay).mul(effectiveDimensions);
            var res2 = lx.mul(ay).mul(targetComponentSums).mul(SLICING_RATIO);
            var res3 = ax.mul(ly).mul(y1);
            var res4 = lx.mul(ly).mul(qcDist);
            var res = res1.add(res2).add(res3).add(res4);
            // For euclidean, we need to invert the score and apply the additional correction, which is
            // assumed to be the squared l2norm of the centroid centered vectors.
            if (similarityFunction == EUCLIDEAN) {
                res = res.mul(-2).add(additionalCorrections).add(queryAdditionalCorrection).add(1f);
                res = FloatVector.broadcast(FLOAT_SPECIES_128, 1).div(res).max(0);
                VectorMask<Float> finiteMask = res.test(VectorOperators.IS_FINITE);
                maxScore = Math.max(maxScore, res.reduceLanes(VectorOperators.MAX, finiteMask));
                if (maxScore < 0) {
                    float[] resArray = res.toArray();
                    Arrays.sort(resArray);
                    maxScore = resArray[resArray.length - 1];
                }
                res.intoArray(scores, i);
            } else {
                // For cosine and max inner product, we need to apply the additional correction, which is
                // assumed to be the non-centered dot-product between the vector and the centroid
                res = res.add(queryAdditionalCorrection).add(additionalCorrections).sub(centroidDp);
                if (similarityFunction == MAXIMUM_INNER_PRODUCT) {
                    res.intoArray(scores, i);
                    // not sure how to do it better
                    for (int j = 0; j < FLOAT_SPECIES_128.length(); j++) {
                        scores[i + j] = VectorUtil.scaleMaxInnerProductScore(scores[i + j]);
                        maxScore = Math.max(maxScore, scores[i + j]);
                    }
                } else {
                    res = res.add(1f).mul(0.5f).max(0);
                    res.intoArray(scores, i);
                    VectorMask<Float> finiteMask = res.test(VectorOperators.IS_FINITE);
                    maxScore = Math.max(maxScore, res.reduceLanes(VectorOperators.MAX, finiteMask));
                    if (maxScore < 0) {
                        float[] resArray = res.toArray();
                        Arrays.sort(resArray);
                        maxScore = resArray[resArray.length - 1];
                    }
                }
            }
        }
        if (limit < bulkSize) {
            maxScore = applyCorrectionsIndividually(
                queryAdditionalCorrection,
                similarityFunction,
                centroidDp,
                ONE_BIT_SCALE,
                scores,
                bulkSize,
                limit,
                offset,
                ay,
                ly,
                y1,
                maxScore
            );
        }
        in.seek(offset + 16L * bulkSize);
        return maxScore;
    }

    private float applyCorrections256Bulk(
        float queryLowerInterval,
        float queryUpperInterval,
        int queryComponentSum,
        float queryAdditionalCorrection,
        VectorSimilarityFunction similarityFunction,
        float centroidDp,
        float[] scores,
        int bulkSize
    ) throws IOException {
        int limit = FLOAT_SPECIES_256.loopBound(bulkSize);
        int i = 0;
        long offset = in.getFilePointer();
        float ay = queryLowerInterval;
        float ly = (queryUpperInterval - ay) * FOUR_BIT_SCALE;
        float y1 = queryComponentSum;
        float maxScore = Float.NEGATIVE_INFINITY;
        for (; i < limit; i += FLOAT_SPECIES_256.length()) {
            var ax = FloatVector.fromMemorySegment(FLOAT_SPECIES_256, memorySegment, offset + i * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
            var lx = FloatVector.fromMemorySegment(
                FLOAT_SPECIES_256,
                memorySegment,
                offset + 4L * bulkSize + i * Float.BYTES,
                ByteOrder.LITTLE_ENDIAN
            ).sub(ax);
            var targetComponentSums = IntVector.fromMemorySegment(
                INT_SPECIES_256,
                memorySegment,
                offset + 8L * bulkSize + i * Integer.BYTES,
                ByteOrder.LITTLE_ENDIAN
            ).convert(VectorOperators.I2F, 0);
            var additionalCorrections = FloatVector.fromMemorySegment(
                FLOAT_SPECIES_256,
                memorySegment,
                offset + 12L * bulkSize + i * Float.BYTES,
                ByteOrder.LITTLE_ENDIAN
            ).mul(SLICING_RATIO);
            var qcDist = FloatVector.fromArray(FLOAT_SPECIES_256, scores, i);
            // ax * ay * effectiveDimensions + ay * lx * (float) targetComponentSum + ax * ly * y1 + lx * ly *
            // qcDist;
            var res1 = ax.mul(ay).mul(effectiveDimensions);
            var res2 = lx.mul(ay).mul(targetComponentSums).mul(SLICING_RATIO);
            var res3 = ax.mul(ly).mul(y1);
            var res4 = lx.mul(ly).mul(qcDist);
            var res = res1.add(res2).add(res3).add(res4);
            // For euclidean, we need to invert the score and apply the additional correction, which is
            // assumed to be the squared l2norm of the centroid centered vectors.
            if (similarityFunction == EUCLIDEAN) {
                res = res.mul(-2).add(additionalCorrections).add(queryAdditionalCorrection).add(1f);
                res = FloatVector.broadcast(FLOAT_SPECIES_256, 1).div(res).max(0);
                maxScore = Math.max(maxScore, res.reduceLanes(VectorOperators.MAX));
                res.intoArray(scores, i);
            } else {
                // For cosine and max inner product, we need to apply the additional correction, which is
                // assumed to be the non-centered dot-product between the vector and the centroid
                res = res.add(queryAdditionalCorrection).add(additionalCorrections).sub(centroidDp);
                if (similarityFunction == MAXIMUM_INNER_PRODUCT) {
                    res.intoArray(scores, i);
                    // not sure how to do it better
                    for (int j = 0; j < FLOAT_SPECIES_256.length(); j++) {
                        scores[i + j] = VectorUtil.scaleMaxInnerProductScore(scores[i + j]);
                        maxScore = Math.max(maxScore, scores[i + j]);
                    }
                } else {
                    res = res.add(1f).mul(0.5f).max(0);
                    maxScore = Math.max(maxScore, res.reduceLanes(VectorOperators.MAX));
                    res.intoArray(scores, i);
                }
            }
        }
        if (limit < bulkSize) {
            maxScore = applyCorrectionsIndividually(
                queryAdditionalCorrection,
                similarityFunction,
                centroidDp,
                ONE_BIT_SCALE,
                scores,
                bulkSize,
                limit,
                offset,
                ay,
                ly,
                y1,
                maxScore
            );
        }
        in.seek(offset + 16L * bulkSize);
        return maxScore;
    }

    @Override
    protected float applyCorrectionsIndividually(
        float queryAdditionalCorrection,
        VectorSimilarityFunction similarityFunction,
        float centroidDp,
        float indexBitScale,
        float[] scores,
        int bulkSize,
        int limit,
        long offset,
        float ay,
        float ly,
        float y1,
        float maxScore
    ) {
        for (int j = limit; j < bulkSize; j++) {
            float ax = memorySegment.get(
                ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN),
                offset + (long) j * Float.BYTES
            );

            float lx = memorySegment.get(
                ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN),
                offset + 4L * bulkSize + (long) j * Float.BYTES
            );
            lx = (lx - ax) * indexBitScale;

            int targetComponentSum = Math.round(memorySegment.get(
                ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN),
                offset + 8L * bulkSize + (long) j * Integer.BYTES
            )  * SLICING_RATIO);

            float additionalCorrection = memorySegment.get(
                ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN),
                offset + 12L * bulkSize + (long) j * Float.BYTES
            ) * SLICING_RATIO;

            float qcDist = scores[j];

            float res = ax * ay * effectiveDimensions + lx * ay * targetComponentSum + ax * ly * y1 + lx * ly * qcDist;

            if (similarityFunction == EUCLIDEAN) {
                res = res * -2f + additionalCorrection + queryAdditionalCorrection + 1f;
                res = Math.max(1f / res, 0f);
                scores[j] = res;
                if (Float.isFinite(res)) {
                    maxScore = Math.max(maxScore, res);
                }
            } else {
                res = res + queryAdditionalCorrection + additionalCorrection - centroidDp;

                if (similarityFunction == MAXIMUM_INNER_PRODUCT) {
                    res = VectorUtil.scaleMaxInnerProductScore(res);
                    scores[j] = res;
                    maxScore = Math.max(maxScore, res);
                } else {
                    res = Math.max((res + 1f) * 0.5f, 0f);
                    scores[j] = res;
                    maxScore = Math.max(maxScore, res);
                }
            }
        }
        return maxScore;
    }
}
