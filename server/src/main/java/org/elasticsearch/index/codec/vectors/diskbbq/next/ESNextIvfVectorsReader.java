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
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.LongValues;
import org.apache.lucene.util.packed.DirectReader;
import org.apache.lucene.util.packed.DirectWriter;
import org.elasticsearch.index.codec.vectors.GenericFlatVectorReaders;
import org.elasticsearch.index.codec.vectors.cluster.ClusteringVectorValues;
import org.elasticsearch.index.codec.vectors.cluster.KMeansByteVectorValues;
import org.elasticsearch.index.codec.vectors.cluster.KMeansFloatVectorValues;
import org.elasticsearch.index.codec.vectors.diskbbq.CentroidIndexFormat;
import org.elasticsearch.index.codec.vectors.diskbbq.CentroidIterator;
import org.elasticsearch.index.codec.vectors.diskbbq.FlatCentroidIndex;
import org.elasticsearch.index.codec.vectors.diskbbq.IVFVectorsReader;
import org.elasticsearch.index.codec.vectors.diskbbq.PrefetchingCentroidIterator;
import org.elasticsearch.search.vectors.ESAcceptDocs;

import java.io.IOException;

/**
 * The part of an {@code ESNext} IVF reader that does not depend on how vectors are quantized: how centroids are
 * found and prefetched, how a sliced segment limits the vectors it searches, and how the centroids of a segment are
 * read back for merging. What a quantizer adds is the extra metadata of its fields (a subclass of {@link IvfFieldEntry})
 * and the posting visitor that scores its codes.
 *
 * @param <E> the field entry of the quantizer
 */
public abstract class ESNextIvfVectorsReader<E extends ESNextIvfVectorsReader.IvfFieldEntry> extends IVFVectorsReader<E> {

    protected ESNextIvfVectorsReader(
        SegmentReadState state,
        GenericFlatVectorReaders.LoadFlatVectorsReader loadReader,
        String codecName,
        String centroidExtension,
        String clusterExtension,
        String metaExtension,
        int versionStart,
        int versionCurrent,
        int versionDirectIo,
        int versionOnDiskMerge,
        float dynamicVisitRatio
    ) throws IOException {
        super(
            state,
            loadReader,
            codecName,
            centroidExtension,
            clusterExtension,
            metaExtension,
            versionStart,
            versionCurrent,
            versionDirectIo,
            versionOnDiskMerge,
            dynamicVisitRatio
        );
    }

    protected ESNextIvfVectorsReader(ESNextIvfVectorsReader<E> other, GenericFlatVectorReaders genericReaders) {
        super(other, genericReaders);
    }

    /**
     * Whether the centroids of the field are stored as one byte per dimension, which only byte-backed fields of
     * quantizers that support them do, instead of as floats.
     */
    protected abstract boolean byteCentroids(E entry);

    CentroidIterator getPostingListPrefetchIterator(CentroidIterator centroidIterator, IndexInput postingListSlice) throws IOException {
        // TODO we may want to prefetch more than one postings list, however, we will likely want to place a limit
        // so we don't bother prefetching many lists we won't end up scoring
        return new PrefetchingCentroidIterator(centroidIterator, postingListSlice);
    }

    @Override
    protected int getNumberOfVectors(E entry, KnnVectorValues values, IndexInput centroidSlice, ESAcceptDocs esAcceptDocs)
        throws IOException {
        int size = values.size();
        assert esAcceptDocs == null
            || entry.numSlices >= 0 && esAcceptDocs.sliceOrd() >= 0
            || entry.numSlices == -1 && esAcceptDocs.sliceOrd() == -1
            : "slice ordinal [" + esAcceptDocs.sliceOrd() + "] does not match segment slice layout [" + entry.numSlices + "]";
        if (entry.numSlices > 0) {
            long fp = centroidSlice.getFilePointer();
            final int bitsRequired = DirectWriter.bitsRequired(entry.maxSliceSize);
            final long sizeLookup = DirectWriter.bytesRequired(entry.numSlices, bitsRequired);
            if (esAcceptDocs != null) {
                int sliceOrd = esAcceptDocs.sliceOrd();
                assert sliceOrd < entry.numSlices : "sliceOrd out of range for centroid slices";
                final LongValues longValues = DirectReader.getInstance(centroidSlice.randomAccessSlice(fp, sizeLookup), bitsRequired);
                size = (int) longValues.get(sliceOrd);
            }
            centroidSlice.seek(fp + sizeLookup);
        }
        return size;
    }

    @Override
    public CentroidIterator getCentroidIterator(
        FieldInfo fieldInfo,
        int numCentroids,
        IndexInput centroids,
        QueryTarget queryTarget,
        IndexInput postingListSlice,
        AcceptDocs acceptDocs,
        float approximateCost,
        KnnVectorValues values,
        float visitRatio
    ) throws IOException {
        // Extract float target for FlatCentroidIndex (byte queries are converted at the IVF search entry point)
        // TODO: For byte queries, this widens to float because FlatCentroidIndex quantizes the query against the
        // global centroid (float) via scalarQuantize(float[], ..., float[]). A mixed scalarQuantize(byte[], ..., float[])
        // overload in OptimizedScalarQuantizer + ESVectorUtil SIMD layer would eliminate this widening.
        // Low priority — only widens a single query vector, not a hot path.
        float[] targetQuery = queryTarget.asFloatVector();
        E fieldEntry = fields.get(fieldInfo.number);
        var iterator = switch (fieldEntry.centroidIndexFormat()) {
            case FLAT -> new FlatCentroidIndex(
                fieldInfo,
                fieldEntry,
                numCentroids,
                centroids,
                targetQuery,
                acceptDocs,
                approximateCost,
                values,
                visitRatio,
                byteCentroids(fieldEntry)
            ).getIterator();
        };
        return getPostingListPrefetchIterator(iterator, postingListSlice);
    }

    @Override
    protected long maxVectorsToVisit(E entry, float visitRatio, int numVectors) {
        return switch (entry.centroidIndexFormat()) {
            case FLAT -> super.maxVectorsToVisit(entry, visitRatio, numVectors);
        };
    }

    @Override
    @SuppressWarnings({ "rawtypes", "unchecked" })
    public CentroidData<?> readCentroidData(String fieldName) throws IOException {
        FieldInfo fieldInfo = fieldInfos.fieldInfo(fieldName);
        if (fieldInfo == null) {
            return null;
        }
        E entry = fields.get(fieldInfo.number);
        if (entry == null || entry.numCentroids() == 0) {
            return null;
        }
        int dimension = fieldInfo.getVectorDimension();
        int numCentroids = entry.numCentroids();
        final KnnVectorValues vectorValues;
        if (fieldInfo.getVectorEncoding().equals(VectorEncoding.BYTE)) {
            vectorValues = getByteVectorValues(fieldInfo.name);
        } else {
            vectorValues = getFloatVectorValues(fieldInfo.name);
        }
        int numVectors = vectorValues != null ? vectorValues.size() : 0;
        int[] clusterSizes = new int[numCentroids];

        // Byte-backed fields store centroids as 1 byte per dimension;
        // float-backed fields store centroids as 4 bytes (float) per dimension.
        boolean byteBacked = byteCentroids(entry);
        int bytesPerComponent = byteBacked ? Byte.BYTES : Float.BYTES;
        long rawCentroidsSize = (long) numCentroids * dimension * bytesPerComponent;

        IndexInput centroidsSlice = null;
        boolean success = false;
        try (IndexInput centroidSlice = entry.centroidSlice(ivfCentroids); IndexInput postingSlice = entry.postingListSlice(ivfClusters)) {
            long[] postingOffsets = readPostingListOffsets(centroidSlice, numVectors, numCentroids, dimension, bytesPerComponent);

            // First pass: read cluster sizes only (from the posting slice).
            for (int c = 0; c < numCentroids; c++) {
                postingSlice.seek(postingOffsets[c] + Integer.BYTES);
                clusterSizes[c] = postingSlice.readVInt();
            }

            // The raw centroids live contiguously at the end of the centroid data; slice that
            // region and hand it to the streaming view. The slice owns its own resources and
            // outlives the parent centroidSlice.
            long centroidsOffset = centroidSlice.length() - rawCentroidsSize;
            centroidsSlice = centroidSlice.slice("centroids-raw", centroidsOffset, rawCentroidsSize);
            ClusteringVectorValues centroids;
            if (byteBacked) {
                centroids = KMeansByteVectorValues.build(centroidsSlice, null, numCentroids, dimension);
            } else {
                centroids = KMeansFloatVectorValues.build(centroidsSlice, null, numCentroids, dimension);
            }
            CentroidData data = new CentroidData(centroids, clusterSizes, entry.globalCentroid(), centroidsSlice);
            success = true;
            return data;
        } finally {
            if (success == false && centroidsSlice != null) {
                centroidsSlice.close();
            }
        }
    }

    private static long[] readPostingListOffsets(
        IndexInput centroidSlice,
        int numVectors,
        int numCentroids,
        int dimension,
        int bytesPerComponent
    ) throws IOException {
        long[] offsets = new long[numCentroids];
        int bitsRequired = DirectWriter.bitsRequired(numCentroids);
        long sizeLookup = DirectWriter.bytesRequired(numVectors, bitsRequired);
        centroidSlice.seek(sizeLookup);
        int numParents = centroidSlice.readVInt();
        long rawCentroidsSize = (long) numCentroids * dimension * bytesPerComponent;
        long offsetTableEntrySize = numParents == 0 ? 2L * Long.BYTES : 2L * Long.BYTES + Integer.BYTES;
        long offsetTableStart = centroidSlice.length() - rawCentroidsSize - offsetTableEntrySize * numCentroids;

        centroidSlice.seek(offsetTableStart);
        for (int i = 0; i < numCentroids; i++) {
            offsets[i] = centroidSlice.readLong();
            centroidSlice.readLong();
            if (numParents > 0) {
                centroidSlice.readInt();
            }
        }
        return offsets;
    }

    /**
     * The metadata every {@code ESNext} field has, whatever its quantizer: how its centroids are indexed, where its
     * preconditioner or projection is stored, and whether and how its segment is sliced. Quantizers extend it with their
     * own settings.
     */
    public abstract static class IvfFieldEntry extends FieldEntry {
        private final CentroidIndexFormat centroidIndexFormat;
        protected final long preconditionerOffset;
        protected final long preconditionerLength;
        // -1 "not sliced".
        // 0 "sliced but on flush".
        // > 0 "sliced but on merge, is the number of slices".
        final int numSlices;
        final int maxSliceSize;

        protected IvfFieldEntry(
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
            int maxSliceSize
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
                bulkSize
            );
            this.centroidIndexFormat = centroidIndexFormat;
            this.preconditionerOffset = preconditionerOffset;
            this.preconditionerLength = preconditionerLength;
            this.numSlices = numSlices;
            this.maxSliceSize = maxSliceSize;
        }

        public CentroidIndexFormat centroidIndexFormat() {
            return centroidIndexFormat;
        }

        public long preconditionerOffset() {
            return preconditionerOffset;
        }

        public long preconditionerLength() {
            return preconditionerLength;
        }

        @Override
        public int numSlices() {
            return numSlices;
        }
    }
}
