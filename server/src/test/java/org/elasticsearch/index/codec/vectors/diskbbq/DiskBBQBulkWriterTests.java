/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq;

import org.apache.lucene.store.ByteBuffersDataOutput;
import org.apache.lucene.store.ByteBuffersIndexOutput;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.elasticsearch.common.lucene.store.ByteArrayIndexInput;
import org.elasticsearch.index.codec.vectors.OptimizedScalarQuantizer;
import org.elasticsearch.test.ESTestCase;

import java.io.IOException;

public class DiskBBQBulkWriterTests extends ESTestCase {

    private static final Integer[] VALID_BIT_SIZES = { 1, 2, 4, 7 };
    private static final Integer[] INVALID_BIT_SIZES = { 0, 3, 5, 6, 8, 16 };

    public void testBulkWriterLayoutMatrix() throws Exception {
        for (int bits : VALID_BIT_SIZES) {
            int dimensions = randomIntBetween(2, 64);
            int bulkSize = randomIntBetween(2, 16);
            int[] counts = new int[] { randomIntBetween(1, bulkSize - 1), bulkSize, bulkSize + randomIntBetween(1, bulkSize - 1) };
            for (int numVectors : counts) {
                for (boolean blockEncodeTailVectors : new boolean[] { false, true }) {
                    for (boolean writeComponentSumAsInt : new boolean[] { false, true }) {
                        if (bits == 7 && writeComponentSumAsInt == false) {
                            // 7-bit writer always stores component sum as int.
                            continue;
                        }
                        assertEncodingLayout(bits, dimensions, bulkSize, numVectors, blockEncodeTailVectors, writeComponentSumAsInt);
                    }
                }
            }
        }
    }

    public void testFromBitSizeValidValues() throws IOException {
        int bits = randomFrom(VALID_BIT_SIZES);
        try (IndexOutput out = new ByteBuffersIndexOutput(new ByteBuffersDataOutput(), "test", "test")) {
            DiskBBQBulkWriter writer = DiskBBQBulkWriter.fromBitSize(bits, 32, out);
            assertNotNull(writer);
        }
    }

    public void testFromBitSizeInvalidValues() throws IOException {
        int bits = randomFrom(INVALID_BIT_SIZES);
        try (IndexOutput out = new ByteBuffersIndexOutput(new ByteBuffersDataOutput(), "test", "test")) {
            expectThrows(IllegalArgumentException.class, () -> DiskBBQBulkWriter.fromBitSize(bits, 32, out));
        }
    }

    private void assertEncodingLayout(
        int bits,
        int dimensions,
        int bulkSize,
        int numVectors,
        boolean blockEncodeTailVectors,
        boolean writeComponentSumAsInt
    ) throws IOException {
        byte[][] vectors = new byte[numVectors][dimensions];
        OptimizedScalarQuantizer.QuantizationResult[] corrections = new OptimizedScalarQuantizer.QuantizationResult[numVectors];
        for (int i = 0; i < numVectors; i++) {
            random().nextBytes(vectors[i]);
            corrections[i] = new OptimizedScalarQuantizer.QuantizationResult(
                randomFloat(),
                randomFloat(),
                randomFloat(),
                randomIntBetween(0, 60_000)
            );
        }

        ByteBuffersDataOutput buffer = new ByteBuffersDataOutput();
        try (IndexOutput out = new ByteBuffersIndexOutput(buffer, "diskbbq", "diskbbq")) {
            DiskBBQBulkWriter writer = DiskBBQBulkWriter.fromBitSize(bits, bulkSize, out, blockEncodeTailVectors, writeComponentSumAsInt);
            writer.writeVectors(new TestQuantizedVectorValues(vectors, corrections), null);
        }

        try (IndexInput in = new ByteArrayIndexInput("diskbbq", buffer.toArrayCopy())) {
            int fullBlocks = numVectors / bulkSize;
            int tailSize = numVectors % bulkSize;
            for (int block = 0; block < fullBlocks; block++) {
                int base = block * bulkSize;
                assertBlockLayout(in, vectors, corrections, dimensions, base, bulkSize, bits, writeComponentSumAsInt);
            }
            if (tailSize == 0) {
                assertEquals(in.length(), in.getFilePointer());
                return;
            }
            int tailStart = fullBlocks * bulkSize;
            if (blockEncodeTailVectors) {
                assertBlockLayout(in, vectors, corrections, dimensions, tailStart, tailSize, bits, writeComponentSumAsInt);
            } else {
                assertTailInterleavedLayout(in, vectors, corrections, dimensions, tailStart, numVectors, bits, writeComponentSumAsInt);
            }
            assertEquals(in.length(), in.getFilePointer());
        }
    }

    private static void assertBlockLayout(
        IndexInput in,
        byte[][] vectors,
        OptimizedScalarQuantizer.QuantizationResult[] corrections,
        int dimensions,
        int start,
        int count,
        int bits,
        boolean writeComponentSumAsInt
    ) throws IOException {
        for (int i = start; i < start + count; i++) {
            assertVectorEquals(in, vectors[i], dimensions);
        }
        for (int i = start; i < start + count; i++) {
            assertEquals(corrections[i].lowerInterval(), readFloat(in), 0.0f);
        }
        for (int i = start; i < start + count; i++) {
            assertEquals(corrections[i].upperInterval(), readFloat(in), 0.0f);
        }
        for (int i = start; i < start + count; i++) {
            assertComponentSum(in, corrections[i], bits, writeComponentSumAsInt);
        }
        for (int i = start; i < start + count; i++) {
            assertEquals(corrections[i].additionalCorrection(), readFloat(in), 0.0f);
        }
    }

    private static void assertTailInterleavedLayout(
        IndexInput in,
        byte[][] vectors,
        OptimizedScalarQuantizer.QuantizationResult[] corrections,
        int dimensions,
        int start,
        int end,
        int bits,
        boolean writeComponentSumAsInt
    ) throws IOException {
        for (int i = start; i < end; i++) {
            assertVectorEquals(in, vectors[i], dimensions);
            assertEquals(corrections[i].lowerInterval(), readFloat(in), 0.0f);
            assertEquals(corrections[i].upperInterval(), readFloat(in), 0.0f);
            assertEquals(corrections[i].additionalCorrection(), readFloat(in), 0.0f);
            assertComponentSum(in, corrections[i], bits, writeComponentSumAsInt);
        }
    }

    private static void assertComponentSum(
        IndexInput in,
        OptimizedScalarQuantizer.QuantizationResult correction,
        int bits,
        boolean writeComponentSumAsInt
    ) throws IOException {
        if (bits == 7 || writeComponentSumAsInt) {
            assertEquals(correction.quantizedComponentSum(), in.readInt());
        } else {
            assertEquals(correction.quantizedComponentSum(), in.readShort() & 0xffff);
        }
    }

    private static float readFloat(IndexInput in) throws IOException {
        return Float.intBitsToFloat(in.readInt());
    }

    private static void assertVectorEquals(IndexInput in, byte[] expected, int dimensions) throws IOException {
        byte[] actual = new byte[dimensions];
        in.readBytes(actual, 0, dimensions);
        assertArrayEquals(expected, actual);
    }

    private static class TestQuantizedVectorValues implements QuantizedVectorValues {
        private final byte[][] vectors;
        private final OptimizedScalarQuantizer.QuantizationResult[] corrections;
        private int index = -1;

        TestQuantizedVectorValues(byte[][] vectors, OptimizedScalarQuantizer.QuantizationResult[] corrections) {
            this.vectors = vectors;
            this.corrections = corrections;
        }

        @Override
        public int count() {
            return vectors.length;
        }

        @Override
        public byte[] next() {
            index++;
            return vectors[index];
        }

        @Override
        public OptimizedScalarQuantizer.QuantizationResult getCorrections() {
            return corrections[index];
        }
    }
}
