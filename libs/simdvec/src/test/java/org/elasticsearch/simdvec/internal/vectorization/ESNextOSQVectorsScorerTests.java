/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.simdvec.internal.vectorization;

import com.carrotsearch.randomizedtesting.annotations.ParametersFactory;

import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.store.NIOFSDirectory;
import org.apache.lucene.util.VectorUtil;
import org.elasticsearch.index.codec.vectors.OptimizedScalarQuantizer;
import org.elasticsearch.index.codec.vectors.diskbbq.next.ESNextDiskBBQVectorsFormat;
import org.elasticsearch.simdvec.ESNextOSQVectorsScorer;
import org.elasticsearch.xpack.searchablesnapshots.store.SearchableSnapshotDirectoryFactory;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;

import static org.elasticsearch.simdvec.internal.vectorization.VectorScorerTestUtils.createOSQIndexData;
import static org.elasticsearch.simdvec.internal.vectorization.VectorScorerTestUtils.createOSQQueryData;
import static org.elasticsearch.simdvec.internal.vectorization.VectorScorerTestUtils.randomVector;
import static org.elasticsearch.simdvec.internal.vectorization.VectorScorerTestUtils.writeBulkOSQVectorData;
import static org.elasticsearch.simdvec.internal.vectorization.VectorScorerTestUtils.writeSingleOSQVectorData;

public class ESNextOSQVectorsScorerTests extends BaseVectorizationTests {

    private final DirectoryType directoryType;
    private final byte indexBits;
    private final byte queryBits;
    private final VectorSimilarityFunction similarityFunction;

    public enum DirectoryType {
        NIOFS,
        MMAP,
        SNAP
    }

    public ESNextOSQVectorsScorerTests(
        DirectoryType directoryType,
        byte indexBits,
        byte queryBits,
        VectorSimilarityFunction similarityFunction
    ) {
        this.directoryType = directoryType;
        this.indexBits = indexBits;
        this.queryBits = queryBits;
        this.similarityFunction = similarityFunction;
    }

    public void testQuantizeScore() throws Exception {

        final int dimensions = random().nextInt(1, 2000);
        final int numVectors = random().nextInt(1, 100);

        final int length = ESNextDiskBBQVectorsFormat.QuantEncoding.fromBits(indexBits).getDocPackedLength(dimensions);

        final byte[] vector = new byte[length];
        final int queryBytes = length * (queryBits / indexBits);

        try (Directory dir = newParametrizedDirectory()) {
            try (IndexOutput out = dir.createOutput("tests.bin", IOContext.DEFAULT)) {
                for (int i = 0; i < numVectors; i++) {
                    random().nextBytes(vector);
                    if (indexBits == 7) clampTo7Bit(vector, dimensions);
                    out.writeBytes(vector, 0, length);
                }
                CodecUtil.writeFooter(out);
            }
            final byte[] query = new byte[queryBytes];
            random().nextBytes(query);
            if (indexBits == 7) clampTo7Bit(query, dimensions);
            try (IndexInput in = dir.openInput("tests.bin", IOContext.DEFAULT)) {
                // Work on a slice that has just the right number of bytes to make the test fail with an
                // index-out-of-bounds in case the implementation reads more than the allowed number of
                // padding bytes.
                final IndexInput slice = in.slice("test", 0, (long) length * numVectors);
                final ESNextOSQVectorsScorer defaultScorer = defaultProvider().newESNextOSQVectorsScorer(
                    slice,
                    queryBits,
                    indexBits,
                    dimensions,
                    length,
                    ESNextOSQVectorsScorer.BULK_SIZE
                );
                final ESNextOSQVectorsScorer panamaScorer = maybePanamaProvider().newESNextOSQVectorsScorer(
                    in,
                    queryBits,
                    indexBits,
                    dimensions,
                    length,
                    ESNextOSQVectorsScorer.BULK_SIZE
                );
                for (int i = 0; i < numVectors; i++) {
                    assertEquals(defaultScorer.quantizeScore(query), panamaScorer.quantizeScore(query));
                    assertEquals(in.getFilePointer(), slice.getFilePointer());
                }
                assertEquals((long) length * numVectors, slice.getFilePointer());
                assertEquals((long) length * numVectors, in.getFilePointer());
            }
        }
    }

    public void testScore() throws Exception {
        final int maxDims = random().nextInt(1, 1000) * 2;
        final int dimensions = random().nextInt(1, maxDims);
        final int numVectors = random().nextInt(10, 50);

        final int indexVectorPackedLengthInBytes = ESNextDiskBBQVectorsFormat.QuantEncoding.fromBits(indexBits)
            .getDocPackedLength(dimensions);

        final float[] centroid = new float[dimensions];
        randomVector(random(), centroid, similarityFunction);
        OptimizedScalarQuantizer quantizer = new OptimizedScalarQuantizer(similarityFunction);
        int padding = random().nextInt(100);
        byte[] paddingBytes = new byte[padding];
        try (Directory dir = newParametrizedDirectory()) {
            try (IndexOutput out = dir.createOutput("testScore.bin", IOContext.DEFAULT)) {
                random().nextBytes(paddingBytes);
                out.writeBytes(paddingBytes, 0, padding);

                float[] vector = new float[dimensions];
                for (int i = 0; i < numVectors; i++) {
                    randomVector(random(), vector, similarityFunction);
                    var vectorData = createOSQIndexData(vector, centroid, quantizer, dimensions, indexBits, indexVectorPackedLengthInBytes);
                    writeSingleOSQVectorData(out, vectorData);
                }
                CodecUtil.writeFooter(out);
            }
            final float[] query = new float[dimensions];
            randomVector(random(), query, similarityFunction);

            final int queryVectorPackedLengthInBytes = indexBits == 7
                ? ESNextDiskBBQVectorsFormat.QuantEncoding.fromBits(indexBits).getQueryPackedLength(dimensions)
                : indexVectorPackedLengthInBytes * (queryBits / indexBits);
            var queryData = createOSQQueryData(query, centroid, quantizer, dimensions, queryBits, queryVectorPackedLengthInBytes);

            final float centroidDp = VectorUtil.dotProduct(centroid, centroid);
            final float[] floatScratch = new float[3];
            try (IndexInput in = dir.openInput("testScore.bin", IOContext.DEFAULT)) {
                in.seek(padding);
                final int perVectorBytes = indexVectorPackedLengthInBytes + 16;
                assertEquals(in.length(), padding + (long) numVectors * perVectorBytes + CodecUtil.footerLength());
                final IndexInput slice = in.slice("test", in.getFilePointer(), (long) perVectorBytes * numVectors);
                // Work on a slice that has just the right number of bytes to make the test fail with an
                // index-out-of-bounds in case the implementation reads more than the allowed number of
                // padding bytes.
                for (int i = 0; i < numVectors; i++) {
                    final var defaultScorer = defaultProvider().newESNextOSQVectorsScorer(
                        slice,
                        queryBits,
                        indexBits,
                        dimensions,
                        indexVectorPackedLengthInBytes,
                        ESNextOSQVectorsScorer.BULK_SIZE
                    );
                    final var panamaScorer = maybePanamaProvider().newESNextOSQVectorsScorer(
                        in,
                        queryBits,
                        indexBits,
                        dimensions,
                        indexVectorPackedLengthInBytes,
                        ESNextOSQVectorsScorer.BULK_SIZE
                    );
                    long qDist = defaultScorer.quantizeScore(queryData.quantizedVector());
                    slice.readFloats(floatScratch, 0, 3);
                    int quantizedComponentSum = slice.readInt();
                    float defaultScore = defaultScorer.applyCorrectionsIndividually(
                        queryData.lowerInterval(),
                        queryData.upperInterval(),
                        queryData.quantizedComponentSum(),
                        queryData.additionalCorrection(),
                        similarityFunction,
                        centroidDp,
                        floatScratch[0],
                        floatScratch[1],
                        quantizedComponentSum,
                        floatScratch[2],
                        qDist
                    );
                    qDist = panamaScorer.quantizeScore(queryData.quantizedVector());
                    in.readFloats(floatScratch, 0, 3);
                    quantizedComponentSum = in.readInt();
                    float panamaScore = panamaScorer.applyCorrectionsIndividually(
                        queryData.lowerInterval(),
                        queryData.upperInterval(),
                        queryData.quantizedComponentSum(),
                        queryData.additionalCorrection(),
                        similarityFunction,
                        centroidDp,
                        floatScratch[0],
                        floatScratch[1],
                        quantizedComponentSum,
                        floatScratch[2],
                        qDist
                    );
                    assertEquals(defaultScore, panamaScore, 1e-2f);
                    assertEquals(((long) (i + 1) * perVectorBytes), slice.getFilePointer());
                    assertEquals(padding + ((long) (i + 1) * perVectorBytes), in.getFilePointer());
                }
            }
        }
    }

    public void testScoreBulk() throws Exception {
        doTestScoreBulk(ESNextOSQVectorsScorer.BULK_SIZE);
    }

    public void testScoreBulkNonAlignedBulkSize() throws Exception {
        // Pick a bulkSize that is NOT a multiple of 8, so that the tail path in
        // applyCorrectionsIndividually is exercised for both 128-bit (species length 4)
        // and 256-bit (species length 8) vector implementations.
        final int bulkSize = randomIntBetween(1, ESNextOSQVectorsScorer.BULK_SIZE - 1) | 1; // ensure odd
        doTestScoreBulk(bulkSize);
    }

    private void doTestScoreBulk(int bulkSize) throws Exception {
        final int maxDims = random().nextInt(1, 1000) * 2;
        final int dimensions = random().nextInt(1, maxDims);
        final int numVectors = bulkSize * random().nextInt(1, 10);

        final int indexVectorPackedLengthInBytes = ESNextDiskBBQVectorsFormat.QuantEncoding.fromBits(indexBits)
            .getDocPackedLength(dimensions);

        final float[] centroid = new float[dimensions];
        randomVector(random(), centroid, similarityFunction);

        OptimizedScalarQuantizer quantizer = new OptimizedScalarQuantizer(similarityFunction);
        int padding = random().nextInt(100);
        byte[] paddingBytes = new byte[padding];
        try (Directory dir = newParametrizedDirectory()) {
            try (IndexOutput out = dir.createOutput("testScore.bin", IOContext.DEFAULT)) {
                random().nextBytes(paddingBytes);
                out.writeBytes(paddingBytes, 0, padding);

                var vectors = new VectorScorerTestUtils.OSQVectorData[bulkSize];

                for (int i = 0; i < numVectors; i += bulkSize) {
                    for (int j = 0; j < bulkSize; j++) {
                        var vector = new float[dimensions];
                        randomVector(random(), vector, similarityFunction);
                        vectors[j] = createOSQIndexData(vector, centroid, quantizer, dimensions, indexBits, indexVectorPackedLengthInBytes);
                    }
                    writeBulkOSQVectorData(bulkSize, out, vectors);
                }
                CodecUtil.writeFooter(out);
            }
            final float[] query = new float[dimensions];
            randomVector(random(), query, similarityFunction);
            final int queryVectorPackedLengthInBytes = indexBits == 7
                ? ESNextDiskBBQVectorsFormat.QuantEncoding.fromBits(indexBits).getQueryPackedLength(dimensions)
                : indexVectorPackedLengthInBytes * (queryBits / indexBits);
            var queryData = createOSQQueryData(query, centroid, quantizer, dimensions, queryBits, queryVectorPackedLengthInBytes);

            final float centroidDp = VectorUtil.dotProduct(centroid, centroid);
            final float[] scoresDefault = new float[ESNextOSQVectorsScorer.BULK_SIZE];
            final float[] scoresPanama = new float[ESNextOSQVectorsScorer.BULK_SIZE];
            try (IndexInput in = dir.openInput("testScore.bin", IOContext.DEFAULT)) {
                in.seek(padding);
                final int perVectorBytes = indexVectorPackedLengthInBytes + 16;
                assertEquals(in.length(), padding + (long) numVectors * perVectorBytes + CodecUtil.footerLength());
                // Work on a slice that has just the right number of bytes to make the test fail with an
                // index-out-of-bounds in case the implementation reads more than the allowed number of
                // padding bytes.
                for (int i = 0; i < numVectors; i += bulkSize) {
                    final IndexInput slice = in.slice("test", in.getFilePointer(), (long) perVectorBytes * bulkSize);
                    final var defaultScorer = defaultProvider().newESNextOSQVectorsScorer(
                        slice,
                        queryBits,
                        indexBits,
                        dimensions,
                        indexVectorPackedLengthInBytes,
                        ESNextOSQVectorsScorer.BULK_SIZE
                    );
                    final var panamaScorer = maybePanamaProvider().newESNextOSQVectorsScorer(
                        in,
                        queryBits,
                        indexBits,
                        dimensions,
                        indexVectorPackedLengthInBytes,
                        ESNextOSQVectorsScorer.BULK_SIZE
                    );
                    float defaultMaxScore = defaultScorer.scoreBulk(
                        queryData.quantizedVector(),
                        queryData.lowerInterval(),
                        queryData.upperInterval(),
                        queryData.quantizedComponentSum(),
                        queryData.additionalCorrection(),
                        similarityFunction,
                        centroidDp,
                        scoresDefault,
                        bulkSize
                    );
                    float panamaMaxScore = panamaScorer.scoreBulk(
                        queryData.quantizedVector(),
                        queryData.lowerInterval(),
                        queryData.upperInterval(),
                        queryData.quantizedComponentSum(),
                        queryData.additionalCorrection(),
                        similarityFunction,
                        centroidDp,
                        scoresPanama,
                        bulkSize
                    );
                    assertEquals(defaultMaxScore, panamaMaxScore, 1e-2f);
                    assertArrayEqualsPercent(scoresDefault, scoresPanama, 0.05f, 1e-2f);
                    assertEquals(((long) bulkSize * perVectorBytes), slice.getFilePointer());
                    assertEquals(padding + ((long) (i + bulkSize) * perVectorBytes), in.getFilePointer());
                }
            }
        }
    }

    public void testScoreBulkOffsets() throws Exception {
        final int bulkSize = ESNextOSQVectorsScorer.BULK_SIZE;
        int filtered = random().nextInt(0, bulkSize);
        final int[] offsets = VectorScorerTestUtils.generateFilteredOffsets(random(), bulkSize, filtered);
        var offsetsCount = bulkSize - filtered;
        doTestScoreBulkOffsets(offsets, offsetsCount, bulkSize);
    }

    public void testScoreBulkOffsetsOneVector() throws Exception {
        final int bulkSize = ESNextOSQVectorsScorer.BULK_SIZE;
        int filtered = bulkSize - 1;
        final int[] offsets = VectorScorerTestUtils.generateFilteredOffsets(random(), bulkSize, filtered);
        assert offsets.length == 1;
        doTestScoreBulkOffsets(offsets, 1, bulkSize);
    }

    public void testScoreBulkOffsetsAllVectors() throws Exception {
        final int bulkSize = ESNextOSQVectorsScorer.BULK_SIZE;
        int filtered = 0;
        final int[] offsets = VectorScorerTestUtils.generateFilteredOffsets(random(), bulkSize, filtered);
        assert offsets.length == bulkSize;
        doTestScoreBulkOffsets(offsets, bulkSize, bulkSize);
    }

    public void testScoreBulkOffsetsAllVectorsButOne() throws Exception {
        final int bulkSize = ESNextOSQVectorsScorer.BULK_SIZE;
        int filtered = 1;
        final int[] offsets = VectorScorerTestUtils.generateFilteredOffsets(random(), bulkSize, filtered);
        assert offsets.length == bulkSize - 1;
        doTestScoreBulkOffsets(offsets, bulkSize - 1, bulkSize);
    }

    public void testScoreBulkOffsetsTail() throws Exception {
        final int bulkSize = ESNextOSQVectorsScorer.BULK_SIZE;
        int tailSize = random().nextInt(1, bulkSize);
        int filtered = random().nextInt(0, tailSize);
        final int[] offsets = VectorScorerTestUtils.generateFilteredOffsets(random(), tailSize, filtered);
        var offsetsCount = tailSize - filtered;
        doTestScoreBulkOffsets(offsets, offsetsCount, tailSize);
    }

    public void testScoreBulkOffsetsEquivalentToIndividualAligned() throws Exception {
        int count = ESNextOSQVectorsScorer.BULK_SIZE;
        int filtered = random().nextInt(0, count);
        int[] offsets = VectorScorerTestUtils.generateFilteredOffsets(random(), count, filtered);
        doTestScoreBulkOffsetsEquivalentToIndividual(offsets, count);
    }

    public void testScoreBulkOffsetsEquivalentToIndividualTail() throws Exception {
        int count = randomIntBetween(1, ESNextOSQVectorsScorer.BULK_SIZE - 1);
        int filtered = random().nextInt(0, count);
        int[] offsets = VectorScorerTestUtils.generateFilteredOffsets(random(), count, filtered);
        doTestScoreBulkOffsetsEquivalentToIndividual(offsets, count);
    }

    public void testScoreBulkEquivalentToIndividualAligned() throws Exception {
        doTestScoreBulkEquivalentToIndividual(ESNextOSQVectorsScorer.BULK_SIZE * randomIntBetween(1, 4));
    }

    public void testScoreBulkEquivalentToIndividualWithTail() throws Exception {
        final int bulkSize = ESNextOSQVectorsScorer.BULK_SIZE;
        final int tail = randomIntBetween(1, bulkSize - 1);
        doTestScoreBulkEquivalentToIndividual(bulkSize + tail);
    }

    public void testAdditionalCorrectionMonotonicity() throws Exception {
        final int iterations = 20;
        for (int it = 0; it < iterations; it++) {
            final int dimensions = randomIntBetween(4, 256);
            final int indexVectorPackedLengthInBytes = ESNextDiskBBQVectorsFormat.QuantEncoding.fromBits(indexBits)
                .getDocPackedLength(dimensions);
            final int queryVectorPackedLengthInBytes = indexBits == 7
                ? ESNextDiskBBQVectorsFormat.QuantEncoding.fromBits(indexBits).getQueryPackedLength(dimensions)
                : indexVectorPackedLengthInBytes * (queryBits / indexBits);

            final float[] centroid = new float[dimensions];
            randomVector(random(), centroid, similarityFunction);
            final OptimizedScalarQuantizer quantizer = new OptimizedScalarQuantizer(similarityFunction);

            final float[] vector = new float[dimensions];
            randomVector(random(), vector, similarityFunction);
            final var vectorData = createOSQIndexData(vector, centroid, quantizer, dimensions, indexBits, indexVectorPackedLengthInBytes);

            final float[] query = new float[dimensions];
            randomVector(random(), query, similarityFunction);
            final var queryData = createOSQQueryData(query, centroid, quantizer, dimensions, queryBits, queryVectorPackedLengthInBytes);
            final float centroidDp = VectorUtil.dotProduct(centroid, centroid);
            final float delta = randomFloatBetween(1e-3f, 0.5f, true);

            try (Directory dir = newParametrizedDirectory()) {
                try (IndexOutput out = dir.createOutput("monotonicity.bin", IOContext.DEFAULT)) {
                    writeSingleOSQVectorData(out, vectorData);
                    CodecUtil.writeFooter(out);
                }
                try (IndexInput in = dir.openInput("monotonicity.bin", IOContext.DEFAULT)) {
                    final IndexInput slice = in.slice("monotonicity", 0, indexVectorPackedLengthInBytes + 16L);
                    final var scorer = defaultProvider().newESNextOSQVectorsScorer(
                        slice,
                        queryBits,
                        indexBits,
                        dimensions,
                        indexVectorPackedLengthInBytes,
                        ESNextOSQVectorsScorer.BULK_SIZE
                    );
                    final long qDist = scorer.quantizeScore(queryData.quantizedVector());
                    final float baseline = scorer.applyCorrectionsIndividually(
                        queryData.lowerInterval(),
                        queryData.upperInterval(),
                        queryData.quantizedComponentSum(),
                        queryData.additionalCorrection(),
                        similarityFunction,
                        centroidDp,
                        vectorData.lowerInterval(),
                        vectorData.upperInterval(),
                        vectorData.quantizedComponentSum(),
                        vectorData.additionalCorrection(),
                        qDist
                    );
                    final float shifted = scorer.applyCorrectionsIndividually(
                        queryData.lowerInterval(),
                        queryData.upperInterval(),
                        queryData.quantizedComponentSum(),
                        queryData.additionalCorrection(),
                        similarityFunction,
                        centroidDp,
                        vectorData.lowerInterval(),
                        vectorData.upperInterval(),
                        vectorData.quantizedComponentSum(),
                        vectorData.additionalCorrection() + delta,
                        qDist
                    );

                    if (similarityFunction == VectorSimilarityFunction.EUCLIDEAN) {
                        assertTrue("Increasing additional correction should not increase EUCLIDEAN score", shifted <= baseline + 1e-5f);
                    } else {
                        assertTrue(
                            "Increasing additional correction should not decrease DOT_PRODUCT/MIP score",
                            shifted >= baseline - 1e-5f
                        );
                    }
                }
            }
        }
    }

    public void testQueryAdditionalCorrectionMonotonicity() throws Exception {
        final int iterations = 20;
        for (int it = 0; it < iterations; it++) {
            final int dimensions = randomIntBetween(4, 256);
            final int indexVectorPackedLengthInBytes = ESNextDiskBBQVectorsFormat.QuantEncoding.fromBits(indexBits)
                .getDocPackedLength(dimensions);
            final int queryVectorPackedLengthInBytes = indexBits == 7
                ? ESNextDiskBBQVectorsFormat.QuantEncoding.fromBits(indexBits).getQueryPackedLength(dimensions)
                : indexVectorPackedLengthInBytes * (queryBits / indexBits);

            final float[] centroid = new float[dimensions];
            randomVector(random(), centroid, similarityFunction);
            final OptimizedScalarQuantizer quantizer = new OptimizedScalarQuantizer(similarityFunction);

            final float[] vector = new float[dimensions];
            randomVector(random(), vector, similarityFunction);
            final var vectorData = createOSQIndexData(vector, centroid, quantizer, dimensions, indexBits, indexVectorPackedLengthInBytes);

            final float[] query = new float[dimensions];
            randomVector(random(), query, similarityFunction);
            final var queryData = createOSQQueryData(query, centroid, quantizer, dimensions, queryBits, queryVectorPackedLengthInBytes);
            final float centroidDp = VectorUtil.dotProduct(centroid, centroid);
            final float delta = randomFloatBetween(1e-3f, 0.5f, true);

            try (Directory dir = newParametrizedDirectory()) {
                try (IndexOutput out = dir.createOutput("queryMonotonicity.bin", IOContext.DEFAULT)) {
                    writeSingleOSQVectorData(out, vectorData);
                    CodecUtil.writeFooter(out);
                }
                try (IndexInput in = dir.openInput("queryMonotonicity.bin", IOContext.DEFAULT)) {
                    final IndexInput slice = in.slice("queryMonotonicity", 0, indexVectorPackedLengthInBytes + 16L);
                    final var scorer = defaultProvider().newESNextOSQVectorsScorer(
                        slice,
                        queryBits,
                        indexBits,
                        dimensions,
                        indexVectorPackedLengthInBytes,
                        ESNextOSQVectorsScorer.BULK_SIZE
                    );
                    final long qDist = scorer.quantizeScore(queryData.quantizedVector());
                    final float baseline = scorer.applyCorrectionsIndividually(
                        queryData.lowerInterval(),
                        queryData.upperInterval(),
                        queryData.quantizedComponentSum(),
                        queryData.additionalCorrection(),
                        similarityFunction,
                        centroidDp,
                        vectorData.lowerInterval(),
                        vectorData.upperInterval(),
                        vectorData.quantizedComponentSum(),
                        vectorData.additionalCorrection(),
                        qDist
                    );
                    final float shifted = scorer.applyCorrectionsIndividually(
                        queryData.lowerInterval(),
                        queryData.upperInterval(),
                        queryData.quantizedComponentSum(),
                        queryData.additionalCorrection() + delta,
                        similarityFunction,
                        centroidDp,
                        vectorData.lowerInterval(),
                        vectorData.upperInterval(),
                        vectorData.quantizedComponentSum(),
                        vectorData.additionalCorrection(),
                        qDist
                    );

                    if (similarityFunction == VectorSimilarityFunction.EUCLIDEAN) {
                        assertTrue(
                            "Increasing query additional correction should not increase EUCLIDEAN score",
                            shifted <= baseline + 1e-5f
                        );
                    } else {
                        assertTrue(
                            "Increasing query additional correction should not decrease DOT_PRODUCT/MIP score",
                            shifted >= baseline - 1e-5f
                        );
                    }
                }
            }
        }
    }

    public void testAdditionalCorrectionMonotonicityDeterministic() throws Exception {
        final int dimensions = 16;
        final int indexVectorPackedLengthInBytes = ESNextDiskBBQVectorsFormat.QuantEncoding.fromBits(indexBits)
            .getDocPackedLength(dimensions);
        final int queryVectorPackedLengthInBytes = indexBits == 7
            ? ESNextDiskBBQVectorsFormat.QuantEncoding.fromBits(indexBits).getQueryPackedLength(dimensions)
            : indexVectorPackedLengthInBytes * (queryBits / indexBits);

        final float[] centroid = new float[dimensions];
        Arrays.fill(centroid, 1.0f / (float) Math.sqrt(dimensions));
        final OptimizedScalarQuantizer quantizer = new OptimizedScalarQuantizer(similarityFunction);

        final float[] vector = new float[dimensions];
        final float[] query = new float[dimensions];
        Arrays.fill(vector, 0.5f);
        Arrays.fill(query, 0.25f);
        if (similarityFunction != VectorSimilarityFunction.EUCLIDEAN) {
            VectorUtil.l2normalize(vector);
            VectorUtil.l2normalize(query);
        }

        final var vectorData = createOSQIndexData(vector, centroid, quantizer, dimensions, indexBits, indexVectorPackedLengthInBytes);
        final var queryData = createOSQQueryData(query, centroid, quantizer, dimensions, queryBits, queryVectorPackedLengthInBytes);
        final float centroidDp = VectorUtil.dotProduct(centroid, centroid);
        final float delta = 0.25f;

        try (Directory dir = newParametrizedDirectory()) {
            try (IndexOutput out = dir.createOutput("deterministicMonotonicity.bin", IOContext.DEFAULT)) {
                writeSingleOSQVectorData(out, vectorData);
                CodecUtil.writeFooter(out);
            }
            try (IndexInput in = dir.openInput("deterministicMonotonicity.bin", IOContext.DEFAULT)) {
                final IndexInput slice = in.slice("deterministicMonotonicity", 0, indexVectorPackedLengthInBytes + 16L);
                final var scorer = defaultProvider().newESNextOSQVectorsScorer(
                    slice,
                    queryBits,
                    indexBits,
                    dimensions,
                    indexVectorPackedLengthInBytes,
                    ESNextOSQVectorsScorer.BULK_SIZE
                );
                final long qDist = scorer.quantizeScore(queryData.quantizedVector());

                final float base = scorer.applyCorrectionsIndividually(
                    queryData.lowerInterval(),
                    queryData.upperInterval(),
                    queryData.quantizedComponentSum(),
                    queryData.additionalCorrection(),
                    similarityFunction,
                    centroidDp,
                    vectorData.lowerInterval(),
                    vectorData.upperInterval(),
                    vectorData.quantizedComponentSum(),
                    vectorData.additionalCorrection(),
                    qDist
                );
                final float vectorShifted = scorer.applyCorrectionsIndividually(
                    queryData.lowerInterval(),
                    queryData.upperInterval(),
                    queryData.quantizedComponentSum(),
                    queryData.additionalCorrection(),
                    similarityFunction,
                    centroidDp,
                    vectorData.lowerInterval(),
                    vectorData.upperInterval(),
                    vectorData.quantizedComponentSum(),
                    vectorData.additionalCorrection() + delta,
                    qDist
                );
                final float queryShifted = scorer.applyCorrectionsIndividually(
                    queryData.lowerInterval(),
                    queryData.upperInterval(),
                    queryData.quantizedComponentSum(),
                    queryData.additionalCorrection() + delta,
                    similarityFunction,
                    centroidDp,
                    vectorData.lowerInterval(),
                    vectorData.upperInterval(),
                    vectorData.quantizedComponentSum(),
                    vectorData.additionalCorrection(),
                    qDist
                );

                if (similarityFunction == VectorSimilarityFunction.EUCLIDEAN) {
                    assertTrue(vectorShifted <= base + 1e-5f);
                    assertTrue(queryShifted <= base + 1e-5f);
                } else {
                    assertTrue(vectorShifted >= base - 1e-5f);
                    assertTrue(queryShifted >= base - 1e-5f);
                }
            }
        }
    }

    public void testCentroidDpMonotonicityDeterministic() throws Exception {
        final int dimensions = 16;
        final int indexVectorPackedLengthInBytes = ESNextDiskBBQVectorsFormat.QuantEncoding.fromBits(indexBits)
            .getDocPackedLength(dimensions);
        final int queryVectorPackedLengthInBytes = indexBits == 7
            ? ESNextDiskBBQVectorsFormat.QuantEncoding.fromBits(indexBits).getQueryPackedLength(dimensions)
            : indexVectorPackedLengthInBytes * (queryBits / indexBits);

        final float[] centroid = new float[dimensions];
        Arrays.fill(centroid, 1.0f / (float) Math.sqrt(dimensions));
        final OptimizedScalarQuantizer quantizer = new OptimizedScalarQuantizer(similarityFunction);

        final float[] vector = new float[dimensions];
        final float[] query = new float[dimensions];
        Arrays.fill(vector, 0.5f);
        Arrays.fill(query, 0.25f);
        if (similarityFunction != VectorSimilarityFunction.EUCLIDEAN) {
            VectorUtil.l2normalize(vector);
            VectorUtil.l2normalize(query);
        }

        final var vectorData = createOSQIndexData(vector, centroid, quantizer, dimensions, indexBits, indexVectorPackedLengthInBytes);
        final var queryData = createOSQQueryData(query, centroid, quantizer, dimensions, queryBits, queryVectorPackedLengthInBytes);
        final float centroidDp = VectorUtil.dotProduct(centroid, centroid);
        final float largerCentroidDp = centroidDp + 0.25f;

        try (Directory dir = newParametrizedDirectory()) {
            try (IndexOutput out = dir.createOutput("centroidDpMonotonicity.bin", IOContext.DEFAULT)) {
                writeSingleOSQVectorData(out, vectorData);
                CodecUtil.writeFooter(out);
            }
            try (IndexInput in = dir.openInput("centroidDpMonotonicity.bin", IOContext.DEFAULT)) {
                final IndexInput slice = in.slice("centroidDpMonotonicity", 0, indexVectorPackedLengthInBytes + 16L);
                final var scorer = defaultProvider().newESNextOSQVectorsScorer(
                    slice,
                    queryBits,
                    indexBits,
                    dimensions,
                    indexVectorPackedLengthInBytes,
                    ESNextOSQVectorsScorer.BULK_SIZE
                );
                final long qDist = scorer.quantizeScore(queryData.quantizedVector());

                final float base = scorer.applyCorrectionsIndividually(
                    queryData.lowerInterval(),
                    queryData.upperInterval(),
                    queryData.quantizedComponentSum(),
                    queryData.additionalCorrection(),
                    similarityFunction,
                    centroidDp,
                    vectorData.lowerInterval(),
                    vectorData.upperInterval(),
                    vectorData.quantizedComponentSum(),
                    vectorData.additionalCorrection(),
                    qDist
                );
                final float shifted = scorer.applyCorrectionsIndividually(
                    queryData.lowerInterval(),
                    queryData.upperInterval(),
                    queryData.quantizedComponentSum(),
                    queryData.additionalCorrection(),
                    similarityFunction,
                    largerCentroidDp,
                    vectorData.lowerInterval(),
                    vectorData.upperInterval(),
                    vectorData.quantizedComponentSum(),
                    vectorData.additionalCorrection(),
                    qDist
                );

                if (similarityFunction == VectorSimilarityFunction.EUCLIDEAN) {
                    assertEquals(base, shifted, 1e-5f);
                } else {
                    assertTrue(shifted <= base + 1e-5f);
                }
            }
        }
    }

    private void doTestScoreBulkEquivalentToIndividual(int numVectors) throws Exception {
        final int dimensions = random().nextInt(1, random().nextInt(1, 1000) * 2);
        final int bulkSize = ESNextOSQVectorsScorer.BULK_SIZE;
        final int indexVectorPackedLengthInBytes = ESNextDiskBBQVectorsFormat.QuantEncoding.fromBits(indexBits)
            .getDocPackedLength(dimensions);
        final int perVectorBytes = indexVectorPackedLengthInBytes + 16;

        final float[] centroid = new float[dimensions];
        randomVector(random(), centroid, similarityFunction);
        final OptimizedScalarQuantizer quantizer = new OptimizedScalarQuantizer(similarityFunction);

        final VectorScorerTestUtils.OSQVectorData[] vectors = new VectorScorerTestUtils.OSQVectorData[numVectors];
        for (int i = 0; i < numVectors; i++) {
            float[] vector = new float[dimensions];
            randomVector(random(), vector, similarityFunction);
            vectors[i] = createOSQIndexData(vector, centroid, quantizer, dimensions, indexBits, indexVectorPackedLengthInBytes);
        }

        final float[] query = new float[dimensions];
        randomVector(random(), query, similarityFunction);
        final int queryVectorPackedLengthInBytes = indexBits == 7
            ? ESNextDiskBBQVectorsFormat.QuantEncoding.fromBits(indexBits).getQueryPackedLength(dimensions)
            : indexVectorPackedLengthInBytes * (queryBits / indexBits);
        final var queryData = createOSQQueryData(query, centroid, quantizer, dimensions, queryBits, queryVectorPackedLengthInBytes);
        final float centroidDp = VectorUtil.dotProduct(centroid, centroid);

        try (Directory dir = newParametrizedDirectory()) {
            try (IndexOutput bulkOut = dir.createOutput("bulk.bin", IOContext.DEFAULT)) {
                for (int i = 0; i < numVectors; i += bulkSize) {
                    int count = Math.min(bulkSize, numVectors - i);
                    writeBulkOSQVectorData(count, bulkOut, vectors, i);
                }
                CodecUtil.writeFooter(bulkOut);
            }

            final float[] bulkScores = new float[numVectors];
            final float[] individualScores = new float[numVectors];
            final float[] scoreScratch = new float[bulkSize];

            try (IndexInput bulkIn = dir.openInput("bulk.bin", IOContext.DEFAULT)) {
                final long dataLength = (long) numVectors * perVectorBytes;
                final IndexInput bulkSlice = bulkIn.slice("bulk", 0, dataLength);
                final IndexInput individualSlice = bulkIn.slice("individual", 0, dataLength);

                final var bulkScorer = defaultProvider().newESNextOSQVectorsScorer(
                    bulkSlice,
                    queryBits,
                    indexBits,
                    dimensions,
                    indexVectorPackedLengthInBytes,
                    bulkSize
                );
                final var individualScorer = defaultProvider().newESNextOSQVectorsScorer(
                    individualSlice,
                    queryBits,
                    indexBits,
                    dimensions,
                    indexVectorPackedLengthInBytes,
                    bulkSize
                );

                for (int i = 0; i < numVectors; i += bulkSize) {
                    int count = Math.min(bulkSize, numVectors - i);
                    bulkScorer.scoreBulk(
                        queryData.quantizedVector(),
                        queryData.lowerInterval(),
                        queryData.upperInterval(),
                        queryData.quantizedComponentSum(),
                        queryData.additionalCorrection(),
                        similarityFunction,
                        centroidDp,
                        scoreScratch,
                        count
                    );
                    System.arraycopy(scoreScratch, 0, bulkScores, i, count);
                }

                for (int i = 0; i < numVectors; i += bulkSize) {
                    int count = Math.min(bulkSize, numVectors - i);
                    for (int j = 0; j < count; j++) {
                        long qDist = individualScorer.quantizeScore(queryData.quantizedVector());
                        var vectorData = vectors[i + j];
                        individualScores[i + j] = individualScorer.applyCorrectionsIndividually(
                            queryData.lowerInterval(),
                            queryData.upperInterval(),
                            queryData.quantizedComponentSum(),
                            queryData.additionalCorrection(),
                            similarityFunction,
                            centroidDp,
                            vectorData.lowerInterval(),
                            vectorData.upperInterval(),
                            vectorData.quantizedComponentSum(),
                            vectorData.additionalCorrection(),
                            qDist
                        );
                    }
                    individualSlice.skipBytes(16L * count);
                }

                assertArrayEqualsPercent(individualScores, bulkScores, 0.001f, 1e-3f);
                assertEquals(dataLength, bulkSlice.getFilePointer());
                assertEquals(dataLength, individualSlice.getFilePointer());
            }
        }
    }

    private void doTestScoreBulkOffsetsEquivalentToIndividual(int[] offsets, int count) throws Exception {
        final int dimensions = random().nextInt(1, random().nextInt(1, 1000) * 2);
        final int bulkSize = ESNextOSQVectorsScorer.BULK_SIZE;
        final int numVectors = count * randomIntBetween(1, 4);
        final int offsetsCount = offsets.length;
        final int indexVectorPackedLengthInBytes = ESNextDiskBBQVectorsFormat.QuantEncoding.fromBits(indexBits)
            .getDocPackedLength(dimensions);
        final int perVectorBytes = indexVectorPackedLengthInBytes + 16;

        final float[] centroid = new float[dimensions];
        randomVector(random(), centroid, similarityFunction);
        final OptimizedScalarQuantizer quantizer = new OptimizedScalarQuantizer(similarityFunction);

        final VectorScorerTestUtils.OSQVectorData[] vectors = new VectorScorerTestUtils.OSQVectorData[numVectors];
        for (int i = 0; i < numVectors; i++) {
            float[] vector = new float[dimensions];
            randomVector(random(), vector, similarityFunction);
            vectors[i] = createOSQIndexData(vector, centroid, quantizer, dimensions, indexBits, indexVectorPackedLengthInBytes);
        }

        final float[] query = new float[dimensions];
        randomVector(random(), query, similarityFunction);
        final int queryVectorPackedLengthInBytes = indexBits == 7
            ? ESNextDiskBBQVectorsFormat.QuantEncoding.fromBits(indexBits).getQueryPackedLength(dimensions)
            : indexVectorPackedLengthInBytes * (queryBits / indexBits);
        final var queryData = createOSQQueryData(query, centroid, quantizer, dimensions, queryBits, queryVectorPackedLengthInBytes);
        final float centroidDp = VectorUtil.dotProduct(centroid, centroid);

        try (Directory dir = newParametrizedDirectory()) {
            try (IndexOutput out = dir.createOutput("bulkOffsets.bin", IOContext.DEFAULT)) {
                for (int i = 0; i < numVectors; i += count) {
                    writeBulkOSQVectorData(count, out, vectors, i);
                }
                CodecUtil.writeFooter(out);
            }

            final float[] bulkOffsetScores = new float[numVectors];
            final float[] individualScores = new float[numVectors];
            final float[] scoreScratch = new float[bulkSize];

            try (IndexInput in = dir.openInput("bulkOffsets.bin", IOContext.DEFAULT)) {
                final long dataLength = (long) numVectors * perVectorBytes;
                final IndexInput bulkSlice = in.slice("bulk-offsets", 0, dataLength);
                final IndexInput individualSlice = in.slice("individual-offsets", 0, dataLength);

                final var bulkOffsetScorer = defaultProvider().newESNextOSQVectorsScorer(
                    bulkSlice,
                    queryBits,
                    indexBits,
                    dimensions,
                    indexVectorPackedLengthInBytes,
                    bulkSize
                );
                final var individualScorer = defaultProvider().newESNextOSQVectorsScorer(
                    individualSlice,
                    queryBits,
                    indexBits,
                    dimensions,
                    indexVectorPackedLengthInBytes,
                    bulkSize
                );

                for (int i = 0; i < numVectors; i += count) {
                    bulkOffsetScorer.scoreBulkOffsets(
                        queryData.quantizedVector(),
                        queryData.lowerInterval(),
                        queryData.upperInterval(),
                        queryData.quantizedComponentSum(),
                        queryData.additionalCorrection(),
                        similarityFunction,
                        centroidDp,
                        offsets,
                        offsetsCount,
                        scoreScratch,
                        count
                    );
                    System.arraycopy(scoreScratch, 0, bulkOffsetScores, i, count);
                }

                for (int i = 0; i < numVectors; i += count) {
                    for (int j = 0; j < count; j++) {
                        long qDist = individualScorer.quantizeScore(queryData.quantizedVector());
                        if (Arrays.binarySearch(offsets, j) >= 0) {
                            var vectorData = vectors[i + j];
                            individualScores[i + j] = individualScorer.applyCorrectionsIndividually(
                                queryData.lowerInterval(),
                                queryData.upperInterval(),
                                queryData.quantizedComponentSum(),
                                queryData.additionalCorrection(),
                                similarityFunction,
                                centroidDp,
                                vectorData.lowerInterval(),
                                vectorData.upperInterval(),
                                vectorData.quantizedComponentSum(),
                                vectorData.additionalCorrection(),
                                qDist
                            );
                        } else {
                            individualScores[i + j] = 0.0f;
                        }
                    }
                    individualSlice.skipBytes(16L * count);
                }

                assertArrayEqualsPercent(individualScores, bulkOffsetScores, 0.001f, 1e-3f);
                assertEquals(dataLength, bulkSlice.getFilePointer());
                assertEquals(dataLength, individualSlice.getFilePointer());
            }
        }
    }

    private void doTestScoreBulkOffsets(int[] offsets, int offsetsCount, int count) throws Exception {
        final int bulkSize = ESNextOSQVectorsScorer.BULK_SIZE;
        final int maxDims = random().nextInt(1, 1000) * 2;
        final int dimensions = random().nextInt(1, maxDims);
        final int numVectors = count * random().nextInt(1, 10);

        final int indexVectorPackedLengthInBytes = ESNextDiskBBQVectorsFormat.QuantEncoding.fromBits(indexBits)
            .getDocPackedLength(dimensions);

        final float[] centroid = new float[dimensions];
        randomVector(random(), centroid, similarityFunction);

        OptimizedScalarQuantizer quantizer = new OptimizedScalarQuantizer(similarityFunction);
        int padding = random().nextInt(100);
        byte[] paddingBytes = new byte[padding];
        try (Directory dir = newParametrizedDirectory()) {
            try (IndexOutput out = dir.createOutput("testScore.bin", IOContext.DEFAULT)) {
                random().nextBytes(paddingBytes);
                out.writeBytes(paddingBytes, 0, padding);

                var vectors = new VectorScorerTestUtils.OSQVectorData[count];

                for (int i = 0; i < numVectors; i += count) {
                    for (int j = 0; j < count; j++) {
                        var vector = new float[dimensions];
                        randomVector(random(), vector, similarityFunction);
                        vectors[j] = createOSQIndexData(vector, centroid, quantizer, dimensions, indexBits, indexVectorPackedLengthInBytes);
                    }
                    writeBulkOSQVectorData(count, out, vectors);
                }
                CodecUtil.writeFooter(out);
            }
            final float[] query = new float[dimensions];
            randomVector(random(), query, similarityFunction);
            final int queryVectorPackedLengthInBytes = indexBits == 7
                ? ESNextDiskBBQVectorsFormat.QuantEncoding.fromBits(indexBits).getQueryPackedLength(dimensions)
                : indexVectorPackedLengthInBytes * (queryBits / indexBits);
            var queryData = createOSQQueryData(query, centroid, quantizer, dimensions, queryBits, queryVectorPackedLengthInBytes);

            final float centroidDp = VectorUtil.dotProduct(centroid, centroid);

            final float[] scoresDefault = new float[bulkSize];
            final float[] scoresPanama = new float[bulkSize];
            try (IndexInput in = dir.openInput("testScore.bin", IOContext.DEFAULT)) {
                in.seek(padding);
                final int perVectorBytes = indexVectorPackedLengthInBytes + 16;
                assertEquals(in.length(), padding + (long) numVectors * perVectorBytes + CodecUtil.footerLength());
                // Work on a slice that has just the right number of bytes to make the test fail with an
                // index-out-of-bounds in case the implementation reads more than the allowed number of
                // padding bytes.
                for (int i = 0; i < numVectors; i += count) {
                    final IndexInput slice = in.slice("test", in.getFilePointer(), (long) perVectorBytes * count);
                    final var defaultScorer = defaultProvider().newESNextOSQVectorsScorer(
                        slice,
                        queryBits,
                        indexBits,
                        dimensions,
                        indexVectorPackedLengthInBytes,
                        bulkSize
                    );
                    final var panamaScorer = maybePanamaProvider().newESNextOSQVectorsScorer(
                        in,
                        queryBits,
                        indexBits,
                        dimensions,
                        indexVectorPackedLengthInBytes,
                        bulkSize
                    );
                    float defaultMaxScore = defaultScorer.scoreBulkOffsets(
                        queryData.quantizedVector(),
                        queryData.lowerInterval(),
                        queryData.upperInterval(),
                        queryData.quantizedComponentSum(),
                        queryData.additionalCorrection(),
                        similarityFunction,
                        centroidDp,
                        offsets,
                        offsetsCount,
                        scoresDefault,
                        count
                    );
                    float panamaMaxScore = panamaScorer.scoreBulkOffsets(
                        queryData.quantizedVector(),
                        queryData.lowerInterval(),
                        queryData.upperInterval(),
                        queryData.quantizedComponentSum(),
                        queryData.additionalCorrection(),
                        similarityFunction,
                        centroidDp,
                        offsets,
                        offsetsCount,
                        scoresPanama,
                        count
                    );
                    assertEquals(defaultMaxScore, panamaMaxScore, 1e-2f);
                    assertArrayEqualsPercent(Arrays.copyOf(scoresDefault, count), Arrays.copyOf(scoresPanama, count), 0.05f, 1e-2f);
                    assertEquals(((long) count * perVectorBytes), slice.getFilePointer());
                    assertEquals(padding + ((long) (i + count) * perVectorBytes), in.getFilePointer());

                    assertFilteredNotScored(count, offsets, scoresDefault);
                }
            }
        }
    }

    private static void assertFilteredNotScored(int count, int[] offsets, float[] scoresDefault) {
        for (int j = 0; j < count; j++) {
            if (Arrays.binarySearch(offsets, j) < 0) {
                assertEquals(0.0f, scoresDefault[j], 0.0f);
            }
        }
    }

    /**
     * Regression test: verifies that the vectorized scorer correctly handles -Infinity raw scores
     * for MAXIMUM_INNER_PRODUCT. Passing Float.NEGATIVE_INFINITY as queryAdditionalCorrection
     * (with all-zero corrections) forces every element's raw score to -Infinity before
     * scaleMaxInnerProductScore is applied. The correct result is 0.0 for all elements.
     * <p>
     * This catches the AVX-512 bug where {@code _mm512_fpclass_ps_mask(res, 0x40)} (Negative Finite)
     * failed to classify -Infinity as negative, causing the positive branch ({@code 1 + res = -Infinity})
     * to be used instead of the negative branch ({@code 1/(1 - res) = 0}).
     */
    public void testScoreBulkWithNegativeInfinityScore() throws Exception {
        final int dimensions = 768;
        final int bulkSize = ESNextOSQVectorsScorer.BULK_SIZE;

        final int length = ESNextDiskBBQVectorsFormat.QuantEncoding.fromBits(indexBits).getDocPackedLength(dimensions);
        final int queryBytes = length * (queryBits / indexBits);

        try (Directory dir = newParametrizedDirectory()) {
            try (IndexOutput out = dir.createOutput("testNegInf.bin", IOContext.DEFAULT)) {
                byte[] vector = new byte[length];
                for (int i = 0; i < bulkSize; i++) {
                    random().nextBytes(vector);
                    if (indexBits == 7) clampTo7Bit(vector, dimensions);
                    out.writeBytes(vector, 0, length);
                }
                // All-zero corrections: zero bytes are interpreted identically regardless of byte order
                byte[] zeroCorrections = new byte[16 * bulkSize];
                out.writeBytes(zeroCorrections, 0, zeroCorrections.length);
                CodecUtil.writeFooter(out);
            }

            byte[] query = new byte[queryBytes];
            random().nextBytes(query);
            if (indexBits == 7) clampTo7Bit(query, dimensions);

            float[] scoresDefault = new float[bulkSize];
            float[] scoresPanama = new float[bulkSize];

            try (IndexInput in = dir.openInput("testNegInf.bin", IOContext.DEFAULT)) {
                final long dataLength = (long) bulkSize * length + 16L * bulkSize;
                final IndexInput slice = in.slice("test", 0, dataLength);
                final var defaultScorer = defaultProvider().newESNextOSQVectorsScorer(
                    slice,
                    queryBits,
                    indexBits,
                    dimensions,
                    length,
                    bulkSize
                );
                final var panamaScorer = maybePanamaProvider().newESNextOSQVectorsScorer(
                    in,
                    queryBits,
                    indexBits,
                    dimensions,
                    length,
                    bulkSize
                );

                // Pass Float.NEGATIVE_INFINITY as queryAdditionalCorrection.
                // With all-zero corrections and zero query intervals, the base score is zero,
                // and adding -Infinity makes every element's total raw score -Infinity.
                float defaultMaxScore = defaultScorer.scoreBulk(
                    query,
                    0f,
                    0f,
                    0,
                    Float.NEGATIVE_INFINITY,
                    similarityFunction,
                    0f,
                    scoresDefault
                );
                float panamaMaxScore = panamaScorer.scoreBulk(
                    query,
                    0f,
                    0f,
                    0,
                    Float.NEGATIVE_INFINITY,
                    similarityFunction,
                    0f,
                    scoresPanama
                );

                assertEquals(defaultMaxScore, panamaMaxScore, 1e-2f);
                for (int j = 0; j < bulkSize; j++) {
                    assertEquals("score mismatch at index " + j, scoresDefault[j], scoresPanama[j], 1e-2f);
                }
                assertEquals(dataLength, slice.getFilePointer());
                assertEquals(dataLength, in.getFilePointer());
            }
        }
    }

    private static void clampTo7Bit(byte[] vector, int dimensions) {
        for (int i = 0; i < dimensions; i++) {
            vector[i] = (byte) (vector[i] & 0x7F);
        }
    }

    private Directory newParametrizedDirectory() throws IOException {
        return switch (directoryType) {
            case NIOFS -> new NIOFSDirectory(createTempDir());
            case MMAP -> new MMapDirectory(createTempDir());
            case SNAP -> SearchableSnapshotDirectoryFactory.newDirectory(createTempDir());
        };
    }

    @ParametersFactory
    public static Iterable<Object[]> parametersFactory() {
        var bitCombinations = List.of(
            List.of((byte) 1, (byte) 4),
            List.of((byte) 2, (byte) 4),
            List.of((byte) 4, (byte) 4),
            List.of((byte) 7, (byte) 7)
        );
        return () -> bitCombinations.stream()
            .flatMap(bits -> Arrays.stream(DirectoryType.values()).map(d -> List.of(d, bits.get(0), bits.get(1))))
            .flatMap(p -> Arrays.stream(VectorSimilarityFunction.values()).map(f -> Stream.concat(p.stream(), Stream.of(f)).toArray()))
            .iterator();
    }
}
