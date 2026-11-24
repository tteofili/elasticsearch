/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq.next;

import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.FilterIndexInput;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.Bits;
import org.elasticsearch.common.CheckedIntFunction;
import org.elasticsearch.index.codec.vectors.GenericFlatVectorReaders;
import org.elasticsearch.index.codec.vectors.ash.AshPostingsVisitor;
import org.elasticsearch.index.codec.vectors.ash.AshProjectionMatrix;
import org.elasticsearch.index.codec.vectors.diskbbq.CentroidIndexFormat;
import org.elasticsearch.index.codec.vectors.diskbbq.IVFVectorsReader;
import org.elasticsearch.index.codec.vectors.diskbbq.IvfSegmentConfig;
import org.elasticsearch.lucene.store.MemorySegmentAccessInputAccess;
import org.elasticsearch.search.vectors.ESAcceptDocs;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.concurrent.ConcurrentHashMap;

/**
 * ASH-specific implementation of {@link IVFVectorsReader}. Scores posting lists using
 * {@link AshPostingsVisitor} with a precomputed projection matrix ({@link AshProjectionMatrix}).
 */
public class ESNextDiskASHVectorsReader extends ESNextIvfVectorsReader<ESNextDiskASHVectorsReader.ASHFieldEntry> {

    private final ConcurrentHashMap<String, AshProjectionMatrix> ashMatrixCache;

    private final int queryBitsPerDim;

    public ESNextDiskASHVectorsReader(SegmentReadState state, GenericFlatVectorReaders.LoadFlatVectorsReader getFormatReader)
        throws IOException {
        this(state, getFormatReader, IvfSegmentConfig.AshConfig.DEFAULT_QUERY_BITS_PER_DIM);
    }

    public ESNextDiskASHVectorsReader(
        SegmentReadState state,
        GenericFlatVectorReaders.LoadFlatVectorsReader getFormatReader,
        int queryBitsPerDim
    ) throws IOException {
        super(
            state,
            getFormatReader,
            ESNextDiskASHVectorsFormat.NAME,
            ESNextDiskASHVectorsFormat.CENTROID_EXTENSION,
            ESNextDiskASHVectorsFormat.CLUSTER_EXTENSION,
            ESNextDiskASHVectorsFormat.IVF_META_EXTENSION,
            ESNextDiskASHVectorsFormat.VERSION_START,
            ESNextDiskASHVectorsFormat.VERSION_CURRENT,
            ESNextDiskASHVectorsFormat.VERSION_DIRECT_IO,
            ESNextDiskASHVectorsFormat.VERSION_ON_DISK_MERGE,
            ESNextDiskASHVectorsFormat.DYNAMIC_VISIT_RATIO
        );
        this.ashMatrixCache = new ConcurrentHashMap<>();
        this.queryBitsPerDim = queryBitsPerDim;
    }

    private ESNextDiskASHVectorsReader(ESNextDiskASHVectorsReader other, GenericFlatVectorReaders genericReaders) {
        super(other, genericReaders);
        this.ashMatrixCache = other.ashMatrixCache;
        this.queryBitsPerDim = other.queryBitsPerDim;
    }

    @Override
    protected ESNextDiskASHVectorsReader mergeInstance(GenericFlatVectorReaders genericReaders) {
        return new ESNextDiskASHVectorsReader(this, genericReaders);
    }

    @Override
    protected boolean byteCentroids(ASHFieldEntry entry) {
        return false; // ASH only supports float centroids
    }

    @Override
    protected ASHFieldEntry doReadField(
        IndexInput input,
        String rawVectorFormat,
        boolean useDirectIOReads,
        VectorSimilarityFunction similarityFunction,
        VectorEncoding vectorEncoding,
        int numCentroids,
        long centroidOffset,
        long centroidLength,
        long postingListOffset,
        long postingListLength,
        float[] globalCentroid,
        float globalCentroidDp
    ) throws IOException {
        int bulkSize = input.readInt();
        CentroidIndexFormat centroidIndexFormat = CentroidIndexFormat.fromId(input.readInt());
        long preconditionerLength = input.readLong();
        long preconditionerOffset = -1;
        if (preconditionerLength > 0) {
            preconditionerOffset = input.readLong();
        }
        int numSlices = input.readInt();
        int maxSliceSize = 0;
        if (numSlices > 0) {
            maxSliceSize = input.readVInt();
        }
        int ashBitsPerDim = input.readVInt();
        return new ASHFieldEntry(
            rawVectorFormat,
            useDirectIOReads,
            similarityFunction,
            vectorEncoding,
            numCentroids,
            centroidOffset,
            centroidLength,
            postingListOffset,
            postingListLength,
            globalCentroid,
            globalCentroidDp,
            centroidIndexFormat,
            bulkSize,
            preconditionerOffset,
            preconditionerLength,
            numSlices,
            maxSliceSize,
            ashBitsPerDim
        );
    }

    private AshProjectionMatrix getAshProjectionMatrix(FieldInfo fieldInfo) {
        return ashMatrixCache.computeIfAbsent(fieldInfo.name, name -> {
            try {
                final ASHFieldEntry fieldEntry = fields.get(fieldInfo.number);
                long preconditionerOffset = fieldEntry.preconditionerOffset;
                long preconditionerLength = fieldEntry.preconditionerLength;
                if (preconditionerLength <= 0) {
                    throw new IllegalStateException("ASH segment missing projection matrix for field: " + fieldInfo.name);
                }
                IndexInput slice = ivfCentroids.slice("ash-preconditioner", preconditionerOffset, preconditionerLength);
                slice.seek(0);
                return AshProjectionMatrix.read(slice);
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        });
    }

    /**
     * Returns the trained ASH projection matrix W for the given field, or {@code null} if this
     * segment does not have one (e.g. an empty field, or a field that was not ASH-encoded).
     * <p>
     * Used at merge time to seed the merged segment's projection matrix from an existing input
     * segment instead of re-learning W from scratch. W is a global orthonormal projection of
     * centered/normalized vectors, so it is effectively centroid-independent and can be reused
     * across a merge without re-training.
     */
    AshProjectionMatrix getProjectionMatrix(FieldInfo fieldInfo) {
        final ASHFieldEntry fieldEntry = fields.get(fieldInfo.number);
        if (fieldEntry == null || fieldEntry.preconditionerLength <= 0) {
            return null;
        }
        return getAshProjectionMatrix(fieldInfo);
    }

    /** Returns the bits per projected dimension the given field was written with in this segment, for tests. */
    int getBitsPerDimForTest(FieldInfo fieldInfo) {
        return fields.get(fieldInfo.number).ashBitsPerDim();
    }

    @Override
    public PostingVisitor getPostingVisitor(
        FieldInfo fieldInfo,
        KnnVectorValues values,
        IndexInput indexInput,
        QueryTarget queryTarget,
        Bits needsScoring,
        IndexInput centroidSlice,
        ESAcceptDocs acceptDocs
    ) throws IOException {
        ASHFieldEntry entry = fields.get(fieldInfo.number);
        float[] target = queryTarget.asFloatVector();

        var ashMatrix = getAshProjectionMatrix(fieldInfo);
        int dimension = fieldInfo.getVectorDimension();
        int numCentroids = entry.numCentroids();
        // Raw float centroids are stored as the last (numCentroids * dimension * Float.BYTES) bytes
        // of the centroid file, written contiguously by FlatCentroidIndexWriter.writeCentroidData().
        long rawCentroidsOffset = centroidSlice.length() - (long) numCentroids * dimension * Float.BYTES;
        IndexInput centroidInput = centroidSlice.clone();
        float[] centroidBuf = new float[dimension];
        CheckedIntFunction<float[], IOException> centroidReader = (int ord) -> {
            centroidInput.seek(rawCentroidsOffset + (long) ord * dimension * Float.BYTES);
            centroidInput.readFloats(centroidBuf, 0, dimension);
            return centroidBuf;
        };
        // Unwrap once so the scorer and visitor share the same IndexInput object.
        // This mirrors how the BBQ MemorySegmentPostingsVisitor passes a single input
        // to both the ES940OSQVectorsScorer and the visitor's own correction reads.
        IndexInput unwrappedInput = FilterIndexInput.unwrapOnlyTest(indexInput);
        unwrappedInput = MemorySegmentAccessInputAccess.unwrap(unwrappedInput);

        if (entry.numSlices == 0) {
            // Sliced segment without per-slice centroid structure (e.g. flushed before merge).
            // Uses SlicedAshPostingsVisitor which translates ordinals to doc IDs and restricts
            // scoring to the target slice's doc range.
            int startDoc;
            int endDoc;
            if (acceptDocs == null) {
                // Plain Lucene AcceptDocs (e.g. CheckIndex) carry no slice information: search the whole segment.
                startDoc = 0;
                endDoc = values.ordToDoc(values.size() - 1) + 1;
            } else {
                // Sliced segments are only ever searched by sliced queries, which always carry a slice ordinal.
                assert acceptDocs.sliceOrd() >= 0 : "sliced segment searched without a slice ordinal";
                ESAcceptDocs.SliceAcceptDocs sliceAcceptDocs = acceptDocs.sliceAcceptDocs();
                startDoc = sliceAcceptDocs.startDoc();
                endDoc = sliceAcceptDocs.endDoc();
            }
            return AshPostingsVisitor.createSliced(
                ashMatrix,
                target,
                fieldInfo.getVectorSimilarityFunction(),
                unwrappedInput,
                needsScoring,
                entry.ashBitsPerDim(),
                queryBitsPerDim,
                centroidReader,
                values,
                startDoc,
                endDoc
            );
        }
        return AshPostingsVisitor.create(
            ashMatrix,
            target,
            fieldInfo.getVectorSimilarityFunction(),
            unwrappedInput,
            needsScoring,
            entry.ashBitsPerDim(),
            queryBitsPerDim,
            centroidReader
        );
    }

    /**
     * Calls {@link #getPostingVisitor} directly for a named float-vector field, bypassing
     * {@link #getNumberOfVectors} and its assertion. Used in tests to exercise the
     * {@code acceptDocs} handling in the {@code numSlices == 0} branch in isolation.
     */
    // package-private for testing
    PostingVisitor getPostingVisitorForTest(String field, float[] query, ESAcceptDocs acceptDocs) throws IOException {
        FieldInfo fieldInfo = fieldInfos.fieldInfo(field);
        ASHFieldEntry entry = fields.get(fieldInfo.number);
        KnnVectorValues values = getFloatVectorValues(field);
        return getPostingVisitor(
            fieldInfo,
            values,
            entry.postingListSlice(ivfClusters.clone()),
            new QueryTarget.FloatQuery(query),
            null,
            entry.centroidSlice(ivfCentroids.clone()),
            acceptDocs
        );
    }

    /**
     * ASH-specific field entry. Extends {@link FieldEntry} with the ASH bits-per-dimension
     * configuration used for document vector encoding.
     */
    public static class ASHFieldEntry extends IvfFieldEntry {
        private final int ashBitsPerDim;

        ASHFieldEntry(
            String rawVectorFormat,
            boolean doDirectIOReads,
            VectorSimilarityFunction similarityFunction,
            VectorEncoding vectorEncoding,
            int numCentroids,
            long centroidOffset,
            long centroidLength,
            long postingListOffset,
            long postingListLength,
            float[] globalCentroid,
            float globalCentroidDp,
            CentroidIndexFormat centroidIndexFormat,
            int bulkSize,
            long preconditionerOffset,
            long preconditionerLength,
            int numSlices,
            int maxSliceSize,
            int ashBitsPerDim
        ) {
            super(
                rawVectorFormat,
                doDirectIOReads,
                similarityFunction,
                vectorEncoding,
                numCentroids,
                centroidOffset,
                centroidLength,
                postingListOffset,
                postingListLength,
                globalCentroid,
                globalCentroidDp,
                centroidIndexFormat,
                bulkSize,
                preconditionerOffset,
                preconditionerLength,
                numSlices,
                maxSliceSize
            );
            this.ashBitsPerDim = ashBitsPerDim;
        }

        public int ashBitsPerDim() {
            return ashBitsPerDim;
        }
    }
}
