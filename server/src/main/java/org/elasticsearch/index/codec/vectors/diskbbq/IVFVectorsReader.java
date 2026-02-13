/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq;

import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.CorruptIndexException;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.internal.hppc.IntObjectHashMap;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.store.ChecksumIndexInput;
import org.apache.lucene.store.DataInput;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.Bits;
import org.elasticsearch.core.IOUtils;
import org.elasticsearch.index.codec.vectors.GenericFlatVectorReaders;
import org.elasticsearch.search.vectors.ESAcceptDocs;
import org.elasticsearch.search.vectors.IVFKnnSearchStrategy;

import java.io.Closeable;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import static org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsReader.SIMILARITY_FUNCTIONS;
import static org.elasticsearch.index.codec.vectors.diskbbq.ES920DiskBBQVectorsFormat.CENTROID_EXTENSION;
import static org.elasticsearch.index.codec.vectors.diskbbq.ES920DiskBBQVectorsFormat.CLUSTER_EXTENSION;
import static org.elasticsearch.index.codec.vectors.diskbbq.ES920DiskBBQVectorsFormat.DYNAMIC_VISIT_RATIO;
import static org.elasticsearch.index.codec.vectors.diskbbq.ES920DiskBBQVectorsFormat.IVF_META_EXTENSION;
import static org.elasticsearch.index.codec.vectors.diskbbq.ES920DiskBBQVectorsFormat.VERSION_DIRECT_IO;

/**
 * Reader for IVF vectors. This reader is used to read the IVF vectors from the index.
 */
public abstract class IVFVectorsReader extends KnnVectorsReader {

    protected final IndexInput ivfCentroids, ivfClusters;
    private final SegmentReadState state;
    private final int versionMeta;
    private final FieldInfos fieldInfos;
    protected final IntObjectHashMap<FieldEntry> fields;
    private final GenericFlatVectorReaders genericReaders;

    @SuppressWarnings("this-escape")
    protected IVFVectorsReader(SegmentReadState state, GenericFlatVectorReaders.LoadFlatVectorsReader loadReader) throws IOException {
        this.state = state;
        this.fieldInfos = state.fieldInfos;
        this.fields = new IntObjectHashMap<>();
        this.genericReaders = new GenericFlatVectorReaders();
        String meta = IndexFileNames.segmentFileName(state.segmentInfo.name, state.segmentSuffix, IVF_META_EXTENSION);

        int versionMeta = -1;
        try (ChecksumIndexInput ivfMeta = state.directory.openChecksumInput(meta)) {
            Throwable priorE = null;
            try {
                versionMeta = CodecUtil.checkIndexHeader(
                    ivfMeta,
                    ES920DiskBBQVectorsFormat.NAME,
                    ES920DiskBBQVectorsFormat.VERSION_START,
                    ES920DiskBBQVectorsFormat.VERSION_CURRENT,
                    state.segmentInfo.getId(),
                    state.segmentSuffix
                );
                readFields(ivfMeta, versionMeta, genericReaders, loadReader);
            } catch (Throwable exception) {
                priorE = exception;
            } finally {
                CodecUtil.checkFooter(ivfMeta, priorE);
            }
            this.versionMeta = versionMeta;
            ivfCentroids = openDataInput(state, versionMeta, CENTROID_EXTENSION, ES920DiskBBQVectorsFormat.NAME, state.context);
            ivfClusters = openDataInput(state, versionMeta, CLUSTER_EXTENSION, ES920DiskBBQVectorsFormat.NAME, state.context);
        } catch (Throwable t) {
            IOUtils.closeWhileHandlingException(this);
            throw t;
        }
    }

    public abstract CentroidIterator getCentroidIterator(
        FieldInfo fieldInfo,
        int numCentroids,
        IndexInput centroids,
        float[] target,
        IndexInput postingListSlice,
        AcceptDocs acceptDocs,
        float approximateCost,
        FloatVectorValues values,
        float visitRatio
    ) throws IOException;

    protected static IndexInput openDataInput(
        SegmentReadState state,
        int versionMeta,
        String fileExtension,
        String codecName,
        IOContext context
    ) throws IOException {
        final String fileName = IndexFileNames.segmentFileName(state.segmentInfo.name, state.segmentSuffix, fileExtension);
        final IndexInput in = state.directory.openInput(fileName, context);
        try {
            final int versionVectorData = CodecUtil.checkIndexHeader(
                in,
                codecName,
                ES920DiskBBQVectorsFormat.VERSION_START,
                ES920DiskBBQVectorsFormat.VERSION_CURRENT,
                state.segmentInfo.getId(),
                state.segmentSuffix
            );
            if (versionMeta != versionVectorData) {
                throw new CorruptIndexException(
                    "Format versions mismatch: meta=" + versionMeta + ", " + codecName + "=" + versionVectorData,
                    in
                );
            }
            CodecUtil.retrieveChecksum(in);
            return in;
        } catch (Throwable t) {
            IOUtils.closeWhileHandlingException(in);
            throw t;
        }
    }

    private void readFields(
        ChecksumIndexInput meta,
        int versionMeta,
        GenericFlatVectorReaders genericFields,
        GenericFlatVectorReaders.LoadFlatVectorsReader loadReader
    ) throws IOException {
        for (int fieldNumber = meta.readInt(); fieldNumber != -1; fieldNumber = meta.readInt()) {
            final FieldInfo info = fieldInfos.fieldInfo(fieldNumber);
            if (info == null) {
                throw new CorruptIndexException("Invalid field number: " + fieldNumber, meta);
            }

            FieldEntry fieldEntry = readField(meta, info, versionMeta);
            // attach the field entry to the corresponding flat vectors reader
            genericFields.loadField(fieldNumber, fieldEntry, loadReader);

            fields.put(info.number, fieldEntry);
        }
    }

    private FieldEntry readField(IndexInput input, FieldInfo info, int versionMeta) throws IOException {
        final String rawVectorFormat = input.readString();
        final boolean useDirectIOReads = versionMeta >= VERSION_DIRECT_IO && input.readByte() == 1;
        final VectorEncoding vectorEncoding = readVectorEncoding(input);
        final VectorSimilarityFunction similarityFunction = readSimilarityFunction(input);
        if (similarityFunction != info.getVectorSimilarityFunction()) {
            throw new IllegalStateException(
                "Inconsistent vector similarity function for field=\""
                    + info.name
                    + "\"; "
                    + similarityFunction
                    + " != "
                    + info.getVectorSimilarityFunction()
            );
        }
        final int numCentroids = input.readInt();
        final long centroidOffset = input.readLong();
        final long centroidLength = input.readLong();
        final float[] globalCentroid = new float[info.getVectorDimension()];
        long postingListOffset = -1;
        long postingListLength = -1;
        float globalCentroidDp = 0;
        if (centroidLength > 0) {
            postingListOffset = input.readLong();
            postingListLength = input.readLong();
            input.readFloats(globalCentroid, 0, globalCentroid.length);
            globalCentroidDp = Float.intBitsToFloat(input.readInt());
        }
        return doReadField(
            input,
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
            globalCentroidDp
        );
    }

    protected abstract FieldEntry doReadField(
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
    ) throws IOException;

    private static VectorSimilarityFunction readSimilarityFunction(DataInput input) throws IOException {
        final int i = input.readInt();
        if (i < 0 || i >= SIMILARITY_FUNCTIONS.size()) {
            throw new IllegalArgumentException("invalid distance function: " + i);
        }
        return SIMILARITY_FUNCTIONS.get(i);
    }

    private static VectorEncoding readVectorEncoding(DataInput input) throws IOException {
        final int encodingId = input.readInt();
        if (encodingId < 0 || encodingId >= VectorEncoding.values().length) {
            throw new CorruptIndexException("Invalid vector encoding id: " + encodingId, input);
        }
        return VectorEncoding.values()[encodingId];
    }

    @Override
    public final void checkIntegrity() throws IOException {
        for (var reader : genericReaders.allReaders()) {
            reader.checkIntegrity();
        }
        CodecUtil.checksumEntireFile(ivfCentroids);
        CodecUtil.checksumEntireFile(ivfClusters);
    }

    protected FlatVectorsReader getReaderForField(String field) {
        FieldInfo info = fieldInfos.fieldInfo(field);
        if (info == null) throw new IllegalArgumentException("Could not find field [" + field + "]");
        return genericReaders.getReaderForField(info.number);
    }

    @Override
    public final FloatVectorValues getFloatVectorValues(String field) throws IOException {
        return getReaderForField(field).getFloatVectorValues(field);
    }

    @Override
    public final ByteVectorValues getByteVectorValues(String field) throws IOException {
        return getReaderForField(field).getByteVectorValues(field);
    }

    @Override
    public final void search(String field, float[] target, KnnCollector knnCollector, AcceptDocs acceptDocs) throws IOException {
        final FieldInfo fieldInfo = state.fieldInfos.fieldInfo(field);
        if (fieldInfo.getVectorEncoding().equals(VectorEncoding.FLOAT32) == false) {
            getReaderForField(field).search(field, target, knnCollector, acceptDocs);
            return;
        }
        if (fieldInfo.getVectorDimension() != target.length) {
            throw new IllegalArgumentException(
                "vector query dimension: " + target.length + " differs from field dimension: " + fieldInfo.getVectorDimension()
            );
        }
        final ESAcceptDocs esAcceptDocs;
        if (acceptDocs instanceof ESAcceptDocs) {
            esAcceptDocs = (ESAcceptDocs) acceptDocs;
        } else {
            esAcceptDocs = null;
        }
        // IVF search is only used for float vectors; byte vectors are handled by the flat reader.
        FloatVectorValues values = getReaderForField(field).getFloatVectorValues(field);
        int numVectors = values.size();
        // TODO returning cost 0 in ESAcceptDocs.ESAcceptDocsAll feels wrong? cost is related to the number of matching documents?
        float approximateCost = (float) (esAcceptDocs == null ? acceptDocs.cost()
            : esAcceptDocs instanceof ESAcceptDocs.ESAcceptDocsAll ? numVectors
            : esAcceptDocs.approximateCost());
        float percentFiltered = Math.max(0f, Math.min(1f, approximateCost / numVectors));
        float visitRatio = DYNAMIC_VISIT_RATIO;
        // Search strategy may be null if this is being called from checkIndex (e.g. from a test)
        if (knnCollector.getSearchStrategy() instanceof IVFKnnSearchStrategy ivfSearchStrategy) {
            visitRatio = ivfSearchStrategy.getVisitRatio();
        }

        FieldEntry entry = fields.get(fieldInfo.number);
        if (visitRatio == DYNAMIC_VISIT_RATIO) {
            // empirically based, and a good dynamic to get decent recall while scaling a la "efSearch"
            // scaling by the number of vectors vs. the nearest neighbors requested
            // not perfect, but a comparative heuristic.
            // TODO: we might want to consider the density of the centroids as experiments shows that for fewer vectors per centroid,
            // the least vectors we need to score to get a good recall.
            float estimated = Math.round(Math.log10(numVectors) * Math.log10(numVectors) * (knnCollector.k()));
            // clip so we visit at least one vector
            visitRatio = estimated / numVectors;
        }
        // we account for soar vectors here. We can potentially visit a vector twice so we multiply by 2 here.
        long maxVectorVisited = (long) (2.0 * visitRatio * numVectors);
        IndexInput postListSlice = entry.postingListSlice(ivfClusters);
        // order centroids by proximity and exposes their posting list metadata
        CentroidIterator centroidPrefetchingIterator = getCentroidIterator(
            fieldInfo,
            entry.numCentroids,
            entry.centroidSlice(ivfCentroids),
            target,
            postListSlice,
            acceptDocs,
            approximateCost,
            values,
            visitRatio
        );
        Bits acceptDocsBits = acceptDocs.bits();
        // score vectors inside each posting list until we have visited enough vectors or found good enough matches
        // we may end up visiting more than maxVectorVisited ?
        PostingVisitor scorer = getPostingVisitor(fieldInfo, postListSlice, target, acceptDocsBits, entry.centroidSlice(ivfCentroids));
        long expectedDocs = 0;
        long actualDocs = 0;
        // initially we visit only the "centroids to search"
        // Note, numCollected is doing the bare minimum here.
        // TODO do we need to handle nested doc counts similarly to how we handle
        // filtering? E.g. keep exploring until we hit an expected number of parent documents vs. child vectors?
        while (centroidPrefetchingIterator.hasNext()
            && (maxVectorVisited > expectedDocs || knnCollector.minCompetitiveSimilarity() == Float.NEGATIVE_INFINITY)) {
            PostingMetadata postingMetadata = centroidPrefetchingIterator.nextPosting();
            expectedDocs += scorer.resetPostingsScorer(postingMetadata);
            actualDocs += scorer.visit(knnCollector);
            if (knnCollector.getSearchStrategy() != null) {
                knnCollector.getSearchStrategy().nextVectorsBlock();
            }
        }
        if (acceptDocsBits != null) {
            // TODO Adjust the value here when using centroid filtering
            float unfilteredRatioVisited = (float) expectedDocs / numVectors;
            int filteredVectors = (int) Math.ceil(numVectors * percentFiltered);
            float expectedScored = Math.min(2 * filteredVectors * unfilteredRatioVisited, expectedDocs / 2f);
            while (centroidPrefetchingIterator.hasNext() && (actualDocs < expectedScored || actualDocs < knnCollector.k())) {
                PostingMetadata postingMetadata = centroidPrefetchingIterator.nextPosting();
                scorer.resetPostingsScorer(postingMetadata);
                actualDocs += scorer.visit(knnCollector);
                if (knnCollector.getSearchStrategy() != null) {
                    knnCollector.getSearchStrategy().nextVectorsBlock();
                }
            }
        }
    }

    @Override
    public final void search(String field, byte[] target, KnnCollector knnCollector, AcceptDocs acceptDocs) throws IOException {
        final FieldInfo fieldInfo = state.fieldInfos.fieldInfo(field);
        final ByteVectorValues values = getReaderForField(field).getByteVectorValues(field);
        for (int i = 0; i < values.size(); i++) {
            final float score = fieldInfo.getVectorSimilarityFunction().compare(target, values.vectorValue(i));
            knnCollector.collect(values.ordToDoc(i), score);
            if (knnCollector.earlyTerminated()) {
                return;
            }
        }
    }

    /**
     * Returns the ordered list of posting list metadata (centroid phase only, no vector scoring).
     * Used by the query layer to partition work for parallel segment search. Caller must pass the
     * same visitRatio that would be used for a full search.
     *
     * @return list of postings in visit order, or empty list if field is not IVF float
     */
    public final List<PostingMetadata> getOrderedPostingMetadataList(
        String field,
        float[] target,
        AcceptDocs acceptDocs,
        float visitRatio
    ) throws IOException {
        final FieldInfo fieldInfo = state.fieldInfos.fieldInfo(field);
        if (fieldInfo == null || fieldInfo.getVectorEncoding().equals(VectorEncoding.FLOAT32) == false) {
            return List.of();
        }
        if (fieldInfo.getVectorDimension() != target.length) {
            throw new IllegalArgumentException(
                "vector query dimension: " + target.length + " differs from field dimension: " + fieldInfo.getVectorDimension()
            );
        }
        FieldEntry entry = fields.get(fieldInfo.number);
        if (entry == null) {
            return List.of();
        }
        FloatVectorValues values = getReaderForField(field).getFloatVectorValues(field);
        if (values == null || values.size() == 0) {
            return List.of();
        }
        int numVectors = values.size();
        final ESAcceptDocs esAcceptDocs = acceptDocs instanceof ESAcceptDocs ? (ESAcceptDocs) acceptDocs : null;
        float approximateCost = (float) (esAcceptDocs == null ? acceptDocs.cost()
            : esAcceptDocs instanceof ESAcceptDocs.ESAcceptDocsAll ? numVectors
            : esAcceptDocs.approximateCost());
        if (visitRatio == DYNAMIC_VISIT_RATIO) {
            // Fallback when caller does not pass a concrete ratio (e.g. chunked search should pass ratio from query)
            float estimated = (float) Math.round(
                Math.log10(numVectors) * Math.log10(numVectors) * Math.min(1000, Math.max(10, numVectors / 100))
            );
            visitRatio = Math.max(0f, Math.min(1f, estimated / numVectors));
        }
        // Same cap as full search: only include postings that would be visited under visitRatio
        long maxVectorVisited = Math.max(1, (long) (2.0 * visitRatio * numVectors));
        IndexInput postListSlice = entry.postingListSlice(ivfClusters);
        CentroidIterator it = getCentroidIterator(
            fieldInfo,
            entry.numCentroids,
            entry.centroidSlice(ivfCentroids),
            target,
            postListSlice,
            acceptDocs,
            approximateCost,
            values,
            visitRatio
        );
        Bits acceptDocsBits = acceptDocs.bits();
        PostingVisitor scorer = getPostingVisitor(
            fieldInfo,
            postListSlice,
            target,
            acceptDocsBits,
            entry.centroidSlice(ivfCentroids)
        );
        List<PostingMetadata> list = new ArrayList<>();
        long cumulativeExpected = 0;
        while (it.hasNext() && cumulativeExpected < maxVectorVisited) {
            PostingMetadata pm = it.nextPosting();
            cumulativeExpected += scorer.resetPostingsScorer(pm);
            list.add(pm);
        }
        return list;
    }

    /**
     * Scores a chunk of posting lists using a dedicated IndexInput. Thread-safe when each thread
     * calls with its own chunk: this method opens and closes its own cluster and centroid
     * IndexInputs (the directory must support multiple opens of the same file, e.g. MMapDirectory).
     * Caller is responsible for merging chunk results and applying docBase to doc IDs.
     *
     * @param postingsChunk ordered list of postings to score (subset from getOrderedPostingMetadataList)
     */
    public final void searchPostingListChunk(
        String field,
        float[] target,
        AcceptDocs acceptDocs,
        List<PostingMetadata> postingsChunk,
        KnnCollector knnCollector
    ) throws IOException {
        if (postingsChunk.isEmpty()) {
            return;
        }
        final FieldInfo fieldInfo = state.fieldInfos.fieldInfo(field);
        if (fieldInfo.getVectorEncoding().equals(VectorEncoding.FLOAT32) == false) {
            getReaderForField(field).search(field, target, knnCollector, acceptDocs);
            return;
        }
        if (fieldInfo.getVectorDimension() != target.length) {
            throw new IllegalArgumentException(
                "vector query dimension: " + target.length + " differs from field dimension: " + fieldInfo.getVectorDimension()
            );
        }
        FieldEntry entry = fields.get(fieldInfo.number);
        if (entry == null) {
            return;
        }
        IndexInput clustersInput = null;
        IndexInput centroidsInput = null;
        try {
            clustersInput = openNewClustersInput();
            centroidsInput = openNewCentroidsInput();
            IndexInput postListSlice = entry.postingListSlice(clustersInput);
            IndexInput centroidSlice = entry.centroidSlice(centroidsInput);
            Bits acceptDocsBits = acceptDocs.bits();
            PostingVisitor scorer = getPostingVisitor(fieldInfo, postListSlice, target, acceptDocsBits, centroidSlice);
            for (PostingMetadata pm : postingsChunk) {
                scorer.resetPostingsScorer(pm);
                scorer.visit(knnCollector);
                if (knnCollector.getSearchStrategy() != null) {
                    knnCollector.getSearchStrategy().nextVectorsBlock();
                }
            }
        } finally {
            IOUtils.close(clustersInput, centroidsInput);
        }
    }

    /**
     * Opens a new IndexInput over the segment's cluster file. Each parallel chunk task must use
     * its own input; the directory must support multiple opens of the same file (e.g. MMapDirectory).
     */
    protected final IndexInput openNewClustersInput() throws IOException {
        return openDataInput(state, versionMeta, CLUSTER_EXTENSION, ES920DiskBBQVectorsFormat.NAME, state.context);
    }

    /**
     * Opens a new IndexInput over the segment's centroid file. Each parallel chunk task must use
     * its own input; the directory must support multiple opens of the same file (e.g. MMapDirectory).
     */
    protected final IndexInput openNewCentroidsInput() throws IOException {
        return openDataInput(state, versionMeta, CENTROID_EXTENSION, ES920DiskBBQVectorsFormat.NAME, state.context);
    }

    @Override
    public Map<String, Long> getOffHeapByteSize(FieldInfo fieldInfo) {
        var raw = getReaderForField(fieldInfo.name).getOffHeapByteSize(fieldInfo);
        FieldEntry fe = fields.get(fieldInfo.number);
        if (fe == null) {
            assert fieldInfo.getVectorEncoding() == VectorEncoding.BYTE;
            return raw;
        }

        var centroidsClusters = Map.of(CENTROID_EXTENSION, fe.centroidLength, CLUSTER_EXTENSION, fe.postingListLength);
        return KnnVectorsReader.mergeOffHeapByteSizeMaps(raw, centroidsClusters);
    }

    @Override
    public void close() throws IOException {
        List<Closeable> closeables = new ArrayList<>(genericReaders.allReaders());
        Collections.addAll(closeables, ivfCentroids, ivfClusters);
        IOUtils.close(closeables);
    }

    protected static class FieldEntry implements GenericFlatVectorReaders.Field {
        protected final String rawVectorFormatName;
        protected final boolean useDirectIOReads;
        protected final VectorSimilarityFunction similarityFunction;
        protected final VectorEncoding vectorEncoding;
        protected final int numCentroids;
        protected final long centroidOffset;
        protected final long centroidLength;
        protected final long postingListOffset;
        protected final long postingListLength;
        protected final float[] globalCentroid;
        protected final float globalCentroidDp;
        protected final int bulkSize;

        protected FieldEntry(
            String rawVectorFormatName,
            boolean useDirectIOReads,
            VectorSimilarityFunction similarityFunction,
            VectorEncoding vectorEncoding,
            int numCentroids,
            long centroidOffset,
            long centroidLength,
            long postingListOffset,
            long postingListLength,
            float[] globalCentroid,
            float globalCentroidDp,
            int bulkSize
        ) {
            this.rawVectorFormatName = rawVectorFormatName;
            this.useDirectIOReads = useDirectIOReads;
            this.similarityFunction = similarityFunction;
            this.vectorEncoding = vectorEncoding;
            this.numCentroids = numCentroids;
            this.centroidOffset = centroidOffset;
            this.centroidLength = centroidLength;
            this.postingListOffset = postingListOffset;
            this.postingListLength = postingListLength;
            this.globalCentroid = globalCentroid;
            this.globalCentroidDp = globalCentroidDp;
            this.bulkSize = bulkSize;
        }

        @Override
        public String rawVectorFormatName() {
            return rawVectorFormatName;
        }

        @Override
        public boolean useDirectIOReads() {
            return useDirectIOReads;
        }

        public int numCentroids() {
            return numCentroids;
        }

        public float[] globalCentroid() {
            return globalCentroid;
        }

        public float globalCentroidDp() {
            return globalCentroidDp;
        }

        public VectorSimilarityFunction similarityFunction() {
            return similarityFunction;
        }

        public IndexInput centroidSlice(IndexInput centroidFile) throws IOException {
            return centroidFile.slice("centroids", centroidOffset, centroidLength);
        }

        public IndexInput postingListSlice(IndexInput postingListFile) throws IOException {
            return postingListFile.slice("postingLists", postingListOffset, postingListLength);
        }

        public int getBulkSize() {
            return bulkSize;
        }
    }

    public abstract PostingVisitor getPostingVisitor(
        FieldInfo fieldInfo,
        IndexInput postingsLists,
        float[] target,
        Bits needsScoring,
        IndexInput centroidSlice
    ) throws IOException;

    public interface PostingVisitor {
        /** returns the number of documents in the posting list */
        int resetPostingsScorer(PostingMetadata metadata) throws IOException;

        /** returns the number of scored documents */
        int visit(KnnCollector collector) throws IOException;
    }
}
