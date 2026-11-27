/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq.next;

import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.SortedDocValues;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.TaskExecutor;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.hnsw.IntToIntFunction;
import org.apache.lucene.util.packed.PackedInts;
import org.apache.lucene.util.packed.PackedLongValues;
import org.elasticsearch.core.SuppressForbidden;
import org.elasticsearch.core.WelfordVariance;
import org.elasticsearch.index.codec.vectors.OptimizedScalarQuantizer;
import org.elasticsearch.index.codec.vectors.cluster.CentroidOps;
import org.elasticsearch.index.codec.vectors.cluster.ClusteringByteVectorValues;
import org.elasticsearch.index.codec.vectors.cluster.ClusteringByteVectorValuesSlice;
import org.elasticsearch.index.codec.vectors.cluster.ClusteringFloatVectorValues;
import org.elasticsearch.index.codec.vectors.cluster.ClusteringFloatVectorValuesSlice;
import org.elasticsearch.index.codec.vectors.cluster.ClusteringVectorValues;
import org.elasticsearch.index.codec.vectors.cluster.HierarchicalKMeans;
import org.elasticsearch.index.codec.vectors.cluster.KMeansFloatVectorValues;
import org.elasticsearch.index.codec.vectors.cluster.KMeansNeighbors;
import org.elasticsearch.index.codec.vectors.cluster.KMeansResult;
import org.elasticsearch.index.codec.vectors.cluster.KMeansWithOverspill;
import org.elasticsearch.index.codec.vectors.diskbbq.CentroidIndexFormat;
import org.elasticsearch.index.codec.vectors.diskbbq.CentroidInformation;
import org.elasticsearch.index.codec.vectors.diskbbq.CentroidSlices;
import org.elasticsearch.index.codec.vectors.diskbbq.CentroidSupplier;
import org.elasticsearch.index.codec.vectors.diskbbq.ClusterAssignmentBuilder;
import org.elasticsearch.index.codec.vectors.diskbbq.DiskBBQBulkWriter;
import org.elasticsearch.index.codec.vectors.diskbbq.DocIdsWriter;
import org.elasticsearch.index.codec.vectors.diskbbq.FlatCentroidClusters;
import org.elasticsearch.index.codec.vectors.diskbbq.IVFVectorsReader;
import org.elasticsearch.index.codec.vectors.diskbbq.IVFVectorsWriter;
import org.elasticsearch.index.codec.vectors.diskbbq.IntSorter;
import org.elasticsearch.index.codec.vectors.diskbbq.IvfFlushConfigSource;
import org.elasticsearch.index.codec.vectors.diskbbq.IvfMergeConfigResolver;
import org.elasticsearch.index.codec.vectors.diskbbq.IvfSegmentConfig;
import org.elasticsearch.index.codec.vectors.diskbbq.OverspillAssignments;
import org.elasticsearch.index.codec.vectors.diskbbq.Preconditioner;
import org.elasticsearch.index.codec.vectors.diskbbq.QuantEncoding;
import org.elasticsearch.index.codec.vectors.diskbbq.QuantizedVectorValues;
import org.elasticsearch.index.codec.vectors.diskbbq.TieredMergeStrategy;
import org.elasticsearch.index.codec.vectors.diskbbq.VectorPreconditioner;
import org.elasticsearch.index.codec.vectors.diskbbq.quantize.OsqVectorQuantizer;
import org.elasticsearch.logging.LogManager;
import org.elasticsearch.logging.Logger;
import org.elasticsearch.simdvec.ES940OSQVectorsScorer;
import org.elasticsearch.simdvec.ESVectorUtil;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

import static org.elasticsearch.simdvec.ES940OSQVectorsScorer.BULK_SIZE;

/**
 * Default implementation of {@link IVFVectorsWriter}. It uses {@link HierarchicalKMeans} algorithm to
 * partition the vector space, and then stores the centroids and posting list in a sequential
 * fashion.
 */
public class ESNextDiskBBQVectorsWriter extends ESNextIvfVectorsWriter {
    private static final Logger logger = LogManager.getLogger(ESNextDiskBBQVectorsWriter.class);

    private final QuantEncoding quantEncoding;
    private final int blockDimension;
    private final boolean doPrecondition;

    @Override
    protected boolean supportsByteNative() {
        return true;
    }

    public ESNextDiskBBQVectorsWriter(
        SegmentWriteState state,
        String rawVectorFormatName,
        boolean useDirectIOReads,
        boolean onDiskMerge,
        FlatVectorsWriter rawVectorDelegate,
        CentroidIndexFormat centroidIndexFormat,
        QuantEncoding encoding,
        int vectorPerCluster,
        int centroidsPerParentCluster,
        TaskExecutor mergeExec,
        int numMergeWorkers,
        int blockDimension,
        boolean doPrecondition,
        int flatVectorThreshold,
        String sliceField,
        IvfFlushConfigSource flushConfigSource,
        IvfMergeConfigResolver mergeConfigResolver
    ) throws IOException {
        super(
            state,
            rawVectorFormatName,
            useDirectIOReads,
            rawVectorDelegate,
            ESNextDiskBBQVectorsFormat.VERSION_CURRENT,
            ESNextDiskBBQVectorsFormat.NAME,
            ESNextDiskBBQVectorsFormat.IVF_META_EXTENSION,
            ESNextDiskBBQVectorsFormat.CENTROID_EXTENSION,
            ESNextDiskBBQVectorsFormat.CLUSTER_EXTENSION,
            flatVectorThreshold,
            onDiskMerge,
            centroidIndexFormat,
            vectorPerCluster,
            centroidsPerParentCluster,
            mergeExec,
            numMergeWorkers,
            sliceField,
            flushConfigSource,
            mergeConfigResolver
        );
        this.quantEncoding = encoding;
        this.blockDimension = blockDimension;
        this.doPrecondition = doPrecondition;
    }

    @Override
    protected IvfSegmentConfig beginIvfFieldFlush(FieldInfo fieldInfo) throws IOException {
        IvfSegmentConfig codec = IvfSegmentConfig.fromCodecDefaults(
            centroidIndexFormat,
            new IvfSegmentConfig.OsqConfig(quantEncoding),
            doPrecondition
        );
        return flushConfigSource.load(segmentWriteState, fieldInfo).orElse(codec);
    }

    @Override
    protected IvfSegmentConfig resolveMergeConfig(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
        return mergeConfigResolver.resolve(
            fieldInfo,
            mergeState,
            IvfSegmentConfig.fromCodecDefaults(centroidIndexFormat, new IvfSegmentConfig.OsqConfig(quantEncoding), doPrecondition)
        );
    }

    private static IvfSegmentConfig requireSegmentConfig(IvfSegmentConfig cfg) {
        return Objects.requireNonNull(cfg, "ivf segment config must not be null");
    }

    @Override
    protected Preconditioner inheritPreconditioner(FieldInfo fieldInfo, MergeState mergeState, IvfSegmentConfig fieldWritingContext)
        throws IOException {
        if (requireSegmentConfig(fieldWritingContext).usePrecondition()) {
            for (KnnVectorsReader reader : mergeState.knnVectorsReaders) {
                if (reader instanceof VectorPreconditioner vp) {
                    Preconditioner preconditioner = vp.getPreconditioner(fieldInfo);
                    if (preconditioner != null) {
                        return preconditioner;
                    }
                }
            }
            // else
            return createPreconditioner(fieldInfo.getVectorDimension(), fieldWritingContext);
        }
        return null;
    }

    @Override
    protected Preconditioner createPreconditioner(int dimension, IvfSegmentConfig ivfSegmentConfig) {
        if (requireSegmentConfig(ivfSegmentConfig).usePrecondition()) {
            return Preconditioner.createPreconditioner(dimension, blockDimension);
        } else {
            return null;
        }
    }

    @Override
    protected void writePreconditioner(Preconditioner preconditioner, IndexOutput out) throws IOException {
        if (preconditioner != null) {
            preconditioner.write(out);
        }
    }

    @Override
    public CentroidOffsetAndLength buildAndWritePostingsLists(
        FieldInfo fieldInfo,
        CentroidSupplier centroidSupplier,
        ClusteringVectorValues<?> vectorValues,
        IndexOutput postingsOutput,
        long fileOffset,
        int[] assignments,
        OverspillAssignments overspillAssignments,
        IvfSegmentConfig fieldWritingContext
    ) throws IOException {
        final IvfSegmentConfig segmentConfig = requireSegmentConfig(fieldWritingContext);

        final QuantEncoding effectiveQuantEncoding = segmentConfig.osqEncoding();
        FlatCentroidClusters centroidClusters = (FlatCentroidClusters) centroidSupplier.centroidIndex();
        final boolean isByte = vectorValues instanceof ByteVectorValues;

        ClusterAssignmentBuilder clusterAssignments = ClusterAssignmentBuilder.build(
            assignments,
            overspillAssignments,
            centroidSupplier.size()
        );
        int[][] assignmentsByCluster = clusterAssignments.assignmentsByCluster();
        int maxPostingListSize = clusterAssignments.maxPostingListSize();
        // write the posting lists
        final PackedLongValues.Builder offsets = PackedLongValues.monotonicBuilder(PackedInts.COMPACT);
        final PackedLongValues.Builder lengths = PackedLongValues.monotonicBuilder(PackedInts.COMPACT);
        DiskBBQBulkWriter bulkWriter = DiskBBQBulkWriter.fromBitSize(effectiveQuantEncoding.bits(), BULK_SIZE, postingsOutput, true, true);
        final QuantizedVectorValues quantizedVectors;
        if (isByte) {
            quantizedVectors = new OnHeapQuantizedByteVectors(
                (ByteVectorValues) vectorValues,
                effectiveQuantEncoding,
                fieldInfo.getVectorDimension(),
                new OsqVectorQuantizer(
                    fieldInfo.getVectorSimilarityFunction(),
                    fieldInfo.getVectorDimension(),
                    effectiveQuantEncoding.bits()
                )
            );
        } else {
            quantizedVectors = new OnHeapQuantizedVectors(
                (FloatVectorValues) vectorValues,
                effectiveQuantEncoding,
                fieldInfo.getVectorDimension(),
                new OsqVectorQuantizer(
                    fieldInfo.getVectorSimilarityFunction(),
                    fieldInfo.getVectorDimension(),
                    effectiveQuantEncoding.bits()
                )
            );
        }
        final int[] docIdsScratch = new int[maxPostingListSize];
        final int[] docDeltas = new int[maxPostingListSize];
        final int[] clusterOrds = new int[maxPostingListSize];
        DocIdsWriter idsWriter = new DocIdsWriter();

        for (int c = 0; c < centroidSupplier.size(); c++) {
            float[] parentCentroid = centroidClusters.getCentroid(c);
            int[] cluster = assignmentsByCluster[c];
            long offset = postingsOutput.alignFilePointer(Float.BYTES) - fileOffset;
            offsets.add(offset);

            if (isByte) {
                byte[] centroid = centroidSupplier.byteCentroid(c);
                postingsOutput.writeInt(Float.floatToIntBits(ESVectorUtil.squareDistance(centroid, parentCentroid)));
            } else {
                float[] centroid = centroidSupplier.centroid(c);
                postingsOutput.writeInt(Float.floatToIntBits(ESVectorUtil.squareDistance(centroid, parentCentroid)));
            }

            int size = cluster.length;
            // write docIds
            postingsOutput.writeVInt(size);
            for (int j = 0; j < size; j++) {
                docIdsScratch[j] = vectorValues.ordToDoc(cluster[j]);
                clusterOrds[j] = j;
            }
            // sort cluster.buffer by docIds values, this way cluster ordinals are sorted by docIds
            new IntSorter(clusterOrds, i -> docIdsScratch[i]).sort(0, size);
            // encode doc deltas
            for (int j = 0; j < size; j++) {
                docDeltas[j] = j == 0 ? docIdsScratch[clusterOrds[j]] : docIdsScratch[clusterOrds[j]] - docIdsScratch[clusterOrds[j - 1]];
            }

            if (isByte) {
                byte[] centroid = centroidSupplier.byteCentroid(c);
                ((OnHeapQuantizedByteVectors) quantizedVectors).reset(centroid, parentCentroid, size, ord -> cluster[clusterOrds[ord]]);
            } else {
                float[] centroid = centroidSupplier.centroid(c);
                ((OnHeapQuantizedVectors) quantizedVectors).reset(centroid, parentCentroid, size, ord -> cluster[clusterOrds[ord]]);
            }

            byte encoding = idsWriter.calculateBlockEncoding(i -> docDeltas[i], size, BULK_SIZE);
            postingsOutput.writeByte(encoding);
            if (sliceField != null) {
                // We are not writing the docIds as we know they are writing in vector ord order.
                // we will use the delegated instance on read to do the translation for us.
                assert centroidSupplier.size() == 1;
                bulkWriter.writeVectors(quantizedVectors, null);
            } else {
                bulkWriter.writeVectors(quantizedVectors, i -> {
                    // for vector i we write `bulk` size docs or the remaining docs
                    idsWriter.writeDocIds(d -> docDeltas[i + d], Math.min(BULK_SIZE, size - i), encoding, postingsOutput);
                });
            }
            lengths.add(postingsOutput.getFilePointer() - fileOffset - offset);
        }

        if (logger.isDebugEnabled()) {
            printClusterQualityStatistics(assignmentsByCluster);
        }

        return new CentroidOffsetAndLength(offsets.build(), lengths.build());
    }

    @Override
    @SuppressForbidden(reason = "require usage of Lucene's IOUtils#deleteFilesIgnoringExceptions(...)")
    public CentroidOffsetAndLength buildAndWritePostingsLists(
        FieldInfo fieldInfo,
        CentroidSupplier centroidSupplier,
        ClusteringVectorValues<?> vectorValues,
        IndexOutput postingsOutput,
        long fileOffset,
        MergeState mergeState,
        int[] assignments,
        OverspillAssignments overspillAssignments,
        IvfSegmentConfig fieldWritingContext
    ) throws IOException {
        final IvfSegmentConfig segmentConfig = requireSegmentConfig(fieldWritingContext);

        final QuantEncoding effectiveQuantEncoding = segmentConfig.osqEncoding();
        // first, quantize all the vectors into a temporary file
        var vectorSimilarityFunction = fieldInfo.getVectorSimilarityFunction();
        FlatCentroidClusters centroidClusters = (FlatCentroidClusters) centroidSupplier.centroidIndex();
        PackedLongValues.Builder vectorCentroidOffsets = PackedLongValues.monotonicBuilder(PackedInts.COMPACT);
        String quantizedVectorsTempName = null;
        try (
            IndexOutput quantizedVectorsTemp = mergeState.segmentInfo.dir.createTempOutput(
                mergeState.segmentInfo.name,
                "qvec_",
                IOContext.DEFAULT
            )
        ) {
            quantizedVectorsTempName = quantizedVectorsTemp.getName();
            OsqVectorQuantizer quantizer = new OsqVectorQuantizer(
                vectorSimilarityFunction,
                fieldInfo.getVectorDimension(),
                effectiveQuantEncoding.bits()
            );
            int[] quantized = new int[effectiveQuantEncoding.discretizedDimensions(fieldInfo.getVectorDimension())];
            byte[] binary = new byte[effectiveQuantEncoding.getDocPackedLength(fieldInfo.getVectorDimension())];
            final boolean isByte = vectorValues instanceof ByteVectorValues;
            for (int i = 0; i < assignments.length; i++) {
                // record where this vector's centroid data starts
                vectorCentroidOffsets.add(quantizedVectorsTemp.getFilePointer());
                int c = assignments[i];
                float[] parentCentroid = centroidClusters.getCentroid(c);
                OptimizedScalarQuantizer.QuantizationResult result;
                if (isByte) {
                    result = quantizer.quantize(
                        ((ByteVectorValues) vectorValues).vectorValue(i),
                        centroidSupplier.byteCentroid(c),
                        parentCentroid,
                        quantized
                    );
                } else {
                    result = quantizer.quantize(
                        ((FloatVectorValues) vectorValues).vectorValue(i),
                        centroidSupplier.centroid(c),
                        parentCentroid,
                        quantized
                    );
                }
                effectiveQuantEncoding.pack(quantized, binary);
                writeQuantizedValue(quantizedVectorsTemp, binary, result);

                // write the overspill vectors immediately afterwards
                for (var it = overspillAssignments.getAssignmentsFor(i); it.hasNext();) {
                    int s = it.nextInt();
                    float[] overspillParentCentroid = centroidClusters.getCentroid(s);
                    if (isByte) {
                        result = quantizer.quantize(
                            ((ByteVectorValues) vectorValues).vectorValue(i),
                            centroidSupplier.byteCentroid(s),
                            overspillParentCentroid,
                            quantized
                        );
                    } else {
                        result = quantizer.quantize(
                            ((FloatVectorValues) vectorValues).vectorValue(i),
                            centroidSupplier.centroid(s),
                            overspillParentCentroid,
                            quantized
                        );
                    }
                    effectiveQuantEncoding.pack(quantized, binary);
                    writeQuantizedValue(quantizedVectorsTemp, binary, result);
                }
            }
        } catch (Throwable t) {
            if (quantizedVectorsTempName != null) {
                org.apache.lucene.util.IOUtils.deleteFilesIgnoringExceptions(mergeState.segmentInfo.dir, quantizedVectorsTempName);
            }
            throw t;
        }
        int[] centroidVectorCount = new int[centroidSupplier.size()];
        for (int i = 0; i < assignments.length; i++) {
            centroidVectorCount[assignments[i]]++;

            // if overspill assignments are present, count them as well
            for (var it = overspillAssignments.getAssignmentsFor(i); it.hasNext();) {
                centroidVectorCount[it.nextInt()]++;
            }
        }

        int maxPostingListSize = 0;
        // centroid id -> array of vector ids for that centroid
        int[][] assignmentsByCluster = new int[centroidSupplier.size()][];
        // centroid id -> index of centroid in the vector's centroid data in the temporary file for vector i in assignmentsByCluster
        int[][] overspillVectorIdx = new int[centroidSupplier.size()][];
        for (int c = 0; c < centroidSupplier.size(); c++) {
            int size = centroidVectorCount[c];
            maxPostingListSize = Math.max(maxPostingListSize, size);
            assignmentsByCluster[c] = new int[size];
            overspillVectorIdx[c] = new int[size];
        }
        Arrays.fill(centroidVectorCount, 0);

        for (int i = 0; i < assignments.length; i++) {
            int c = assignments[i];
            assignmentsByCluster[c][centroidVectorCount[c]++] = i;

            // if overspill assignments are present, add them to the cluster as well
            int vIdx = 1;  // don't need to set for the main centroid, it's initialized to 0 anyway
            for (var it = overspillAssignments.getAssignmentsFor(i); it.hasNext(); vIdx++) {
                int s = it.nextInt();
                assignmentsByCluster[s][centroidVectorCount[s]] = i;
                overspillVectorIdx[s][centroidVectorCount[s]] = vIdx;
                centroidVectorCount[s]++;
            }
        }
        // now we can read the quantized vectors from the temporary file
        try (IndexInput quantizedVectorsInput = mergeState.segmentInfo.dir.openInput(quantizedVectorsTempName, IOContext.DEFAULT)) {
            final PackedLongValues.Builder offsets = PackedLongValues.monotonicBuilder(PackedInts.COMPACT);
            final PackedLongValues.Builder lengths = PackedLongValues.monotonicBuilder(PackedInts.COMPACT);
            OffHeapQuantizedVectors offHeapQuantizedVectors = new OffHeapQuantizedVectors(
                quantizedVectorsInput,
                effectiveQuantEncoding,
                fieldInfo.getVectorDimension(),
                vectorCentroidOffsets.build()
            );
            DiskBBQBulkWriter bulkWriter = DiskBBQBulkWriter.fromBitSize(
                effectiveQuantEncoding.bits(),
                BULK_SIZE,
                postingsOutput,
                true,
                true
            );
            // write the posting lists
            final int[] docIdsScratch = new int[maxPostingListSize];
            final int[] docDeltas = new int[maxPostingListSize];
            final int[] clusterOrds = new int[maxPostingListSize];
            DocIdsWriter idsWriter = new DocIdsWriter();
            for (int c = 0; c < centroidSupplier.size(); c++) {
                int[] cluster = assignmentsByCluster[c];
                int[] vectorCentroidIdx = overspillVectorIdx[c];
                long offset = postingsOutput.alignFilePointer(Float.BYTES) - fileOffset;
                offsets.add(offset);
                if (vectorValues instanceof ByteVectorValues) {
                    byte[] centroid = centroidSupplier.byteCentroid(c);
                    postingsOutput.writeInt(Float.floatToIntBits(ESVectorUtil.squareDistance(centroid, centroidClusters.getCentroid(c))));
                } else {
                    float[] centroid = centroidSupplier.centroid(c);
                    postingsOutput.writeInt(Float.floatToIntBits(ESVectorUtil.squareDistance(centroid, centroidClusters.getCentroid(c))));
                }
                // write docIds
                int size = cluster.length;
                postingsOutput.writeVInt(size);
                for (int j = 0; j < size; j++) {
                    docIdsScratch[j] = vectorValues.ordToDoc(cluster[j]);
                    clusterOrds[j] = j;
                }
                // sort cluster.buffer by docIds values, this way cluster ordinals are sorted by docIds
                new IntSorter(clusterOrds, i -> docIdsScratch[i]).sort(0, size);
                // encode doc deltas
                for (int j = 0; j < size; j++) {
                    docDeltas[j] = j == 0
                        ? docIdsScratch[clusterOrds[j]]
                        : docIdsScratch[clusterOrds[j]] - docIdsScratch[clusterOrds[j - 1]];
                }
                byte encoding = idsWriter.calculateBlockEncoding(i -> docDeltas[i], size, BULK_SIZE);
                postingsOutput.writeByte(encoding);
                offHeapQuantizedVectors.reset(size, ord -> vectorCentroidIdx[clusterOrds[ord]], ord -> cluster[clusterOrds[ord]]);
                // write vectors
                if (sliceField != null && centroidSupplier.slices() == null) {
                    // Small-slices case: all vectors are in one centroid, no doc IDs needed.
                    // The reader uses vector ord order for doc translation.
                    assert centroidSupplier.size() == 1;
                    bulkWriter.writeVectors(offHeapQuantizedVectors, null);
                } else {
                    bulkWriter.writeVectors(offHeapQuantizedVectors, i -> {
                        // for vector i we write `bulk` size docs or the remaining docs
                        idsWriter.writeDocIds(d -> docDeltas[d + i], Math.min(BULK_SIZE, size - i), encoding, postingsOutput);
                    });
                }
                lengths.add(postingsOutput.getFilePointer() - fileOffset - offset);
            }

            if (logger.isDebugEnabled()) {
                printClusterQualityStatistics(assignmentsByCluster);
            }
            return new CentroidOffsetAndLength(offsets.build(), lengths.build());
        } finally {
            org.apache.lucene.util.IOUtils.deleteFilesIgnoringExceptions(mergeState.segmentInfo.dir, quantizedVectorsTempName);
        }
    }

    private static void printClusterQualityStatistics(int[][] clusters) {
        int[] sizes = new int[clusters.length];
        for (int i = 0; i < clusters.length; i++) {
            sizes[i] = clusters[i] == null ? 0 : clusters[i].length;
        }
        printClusterQualityStatistics(sizes);
    }

    private static void printClusterQualityStatistics(int[] clusterSizes) {
        float min = Float.MAX_VALUE;
        float max = Float.MIN_VALUE;
        WelfordVariance clusterSizeStats = new WelfordVariance();
        for (int size : clusterSizes) {
            clusterSizeStats.add(size);
            min = Math.min(min, size);
            max = Math.max(max, size);
        }
        double variance = clusterSizeStats.m2() / (clusterSizes.length - 1);
        logger.debug(
            "Centroid count: {} min: {} max: {} mean: {} stdDev: {} variance: {}",
            clusterSizes.length,
            min,
            max,
            clusterSizeStats.mean(),
            Math.sqrt(variance),
            variance
        );
    }

    @Override
    protected CentroidSupplier createCentroidSupplier(FieldInfo info, byte[][] centroids, float[] globalCentroid) throws IOException {
        CentroidSupplier centroidSupplier = CentroidSupplier.fromByteArray(
            centroids,
            new FlatCentroidClusters(KMeansResult.singleCluster(globalCentroid, centroids.length)),
            info.getVectorDimension()
        );
        if (centroidSupplier.size() > centroidsPerParentCluster * centroidsPerParentCluster) {
            // For second-level clustering, widen byte centroids to float
            ClusteringFloatVectorValues floatVectorValues = centroidSupplier.asKmeansFloatVectorValues();
            KMeansResult<float[]> centroidClusters = buildSecondLevelClusters(info, floatVectorValues, false);
            return CentroidSupplier.fromByteArray(centroids, new FlatCentroidClusters(centroidClusters), info.getVectorDimension());
        }
        return centroidSupplier;
    }

    @Override
    @SuppressForbidden(reason = "require usage of Lucene's IOUtils#closeWhileHandlingException(...)")
    @SuppressWarnings({ "rawtypes", "unchecked" })
    public CentroidInformation<?> calculateCentroids(FieldInfo fieldInfo, ClusteringVectorValues<?> vectorValues, MergeState mergeState)
        throws IOException {
        // Sliced indices treat each slice as an independent partition that must be clustered on its
        // own. The tiered merge strategy operates on the merged segment as a flat whole, which would
        // silently collapse slice boundaries, so always fall back to the sliced full rebuild here.
        // TODO: teach the tiered strategy about slices and reuse per-slice priors.
        if (sliceField != null) {
            return calculateCentroidsFullRebuildSliced(vectorValues, mergeState);
        }

        // For byte fields, cluster natively in byte space using TieredMergeStrategy<byte[]>
        if (vectorValues instanceof ClusteringByteVectorValues byteVectorValues) {
            int numSegments = mergeState.knnVectorsReaders.length;
            int[] segmentSizes = new int[numSegments];
            int[] segmentCentroidCounts = new int[numSegments];
            IVFVectorsReader.CentroidData[] segmentCentroidData = new IVFVectorsReader.CentroidData[numSegments];

            try {
                for (int i = 0; i < numSegments; i++) {
                    KnnVectorsReader reader = mergeState.knnVectorsReaders[i];
                    if (reader instanceof PerFieldKnnVectorsFormat.FieldsReader perFieldReader) {
                        reader = perFieldReader.getFieldReader(fieldInfo.name);
                    }
                    if (reader instanceof IVFVectorsReader<?> ivfReader && mergeState.fieldInfos[i].fieldInfo(fieldInfo.name) != null) {
                        ByteVectorValues bvv = ivfReader.getByteVectorValues(fieldInfo.name);
                        segmentSizes[i] = bvv != null ? bvv.size() : 0;
                        segmentCentroidData[i] = ivfReader.readCentroidData(fieldInfo.name);
                        segmentCentroidCounts[i] = segmentCentroidData[i] != null ? segmentCentroidData[i].numCentroids() : 0;
                    } else {
                        segmentSizes[i] = 0;
                        segmentCentroidCounts[i] = 0;
                    }
                }

                TieredMergeStrategy<byte[]> tieredStrategy = new TieredMergeStrategy<>(vectorPerCluster, CentroidOps.BYTE);
                @SuppressWarnings({ "unchecked", "rawtypes" })
                TieredMergeStrategy.MergeAction<byte[]> action = tieredStrategy.selectAction(
                    segmentSizes,
                    segmentCentroidCounts,
                    segmentCentroidData
                );

                HierarchicalKMeans<byte[]> byteKMeans;
                if (mergeExec != null) {
                    byteKMeans = HierarchicalKMeans.ofConcurrent(
                        CentroidOps.BYTE,
                        byteVectorValues.dimension(),
                        mergeExec,
                        numMergeWorkers
                    );
                } else {
                    byteKMeans = HierarchicalKMeans.ofSerial(CentroidOps.BYTE, byteVectorValues.dimension());
                }
                @SuppressWarnings({ "unchecked", "rawtypes" })
                KMeansWithOverspill<byte[]> kMeansResult = action.execute(byteKMeans, byteVectorValues, vectorPerCluster);
                return CentroidInformation.ofBytes(
                    fieldInfo.getVectorDimension(),
                    kMeansResult.centroids(),
                    kMeansResult.assignments(),
                    kMeansResult.overspill()
                );
            } finally {
                org.apache.lucene.util.IOUtils.closeWhileHandlingException(segmentCentroidData);
            }
        }

        // Float clustering path
        KMeansFloatVectorValues floatVectorValues = asFloatVectorValues(fieldInfo, vectorValues);

        // Gather prior segment statistics for tiered merge strategy selection
        int numSegments = mergeState.knnVectorsReaders.length;
        int[] segmentSizes = new int[numSegments];
        int[] segmentCentroidCounts = new int[numSegments];
        IVFVectorsReader.CentroidData[] segmentCentroidData = new IVFVectorsReader.CentroidData[numSegments];

        try {
            for (int i = 0; i < numSegments; i++) {
                KnnVectorsReader reader = mergeState.knnVectorsReaders[i];
                if (reader instanceof PerFieldKnnVectorsFormat.FieldsReader perFieldReader) {
                    reader = perFieldReader.getFieldReader(fieldInfo.name);
                }
                if (reader instanceof IVFVectorsReader<?> ivfReader && mergeState.fieldInfos[i].fieldInfo(fieldInfo.name) != null) {
                    // Get segment size — use the appropriate vector values accessor based on encoding
                    if (fieldInfo.getVectorEncoding() == VectorEncoding.BYTE) {
                        ByteVectorValues bvv = ivfReader.getByteVectorValues(fieldInfo.name);
                        segmentSizes[i] = bvv != null ? bvv.size() : 0;
                    } else {
                        segmentSizes[i] = ivfReader.getFloatVectorValues(fieldInfo.name).size();
                    }
                    segmentCentroidData[i] = ivfReader.readCentroidData(fieldInfo.name);
                    segmentCentroidCounts[i] = segmentCentroidData[i] != null ? segmentCentroidData[i].numCentroids() : 0;
                } else {
                    segmentSizes[i] = 0;
                    segmentCentroidCounts[i] = 0;
                }
            }

            // Select merge strategy
            TieredMergeStrategy<float[]> tieredStrategy = new TieredMergeStrategy<>(vectorPerCluster, CentroidOps.FLOAT);
            TieredMergeStrategy.MergeAction<float[]> action = tieredStrategy.selectAction(
                segmentSizes,
                segmentCentroidCounts,
                segmentCentroidData
            );

            if (logger.isDebugEnabled()) {
                int totalVectors = 0;
                int totalCentroids = 0;
                for (int s : segmentSizes) {
                    totalVectors += s;
                }
                for (int c : segmentCentroidCounts) {
                    totalCentroids += c;
                }
                logger.debug(
                    "DiskBBQ merge for field [{}]: selected strategy [{}], segments={}, totalVectors={}, totalCentroids={}",
                    fieldInfo.name,
                    action.strategy(),
                    numSegments,
                    totalVectors,
                    totalCentroids
                );
            }

            HierarchicalKMeans<float[]> hierarchicalKMeans;
            if (mergeExec != null) {
                hierarchicalKMeans = HierarchicalKMeans.ofConcurrent(
                    CentroidOps.FLOAT,
                    floatVectorValues.dimension(),
                    mergeExec,
                    numMergeWorkers
                );
            } else {
                hierarchicalKMeans = HierarchicalKMeans.ofSerial(CentroidOps.FLOAT, floatVectorValues.dimension());
            }
            KMeansWithOverspill<float[]> kMeansResult = action.execute(hierarchicalKMeans, floatVectorValues, vectorPerCluster);
            if (logger.isDebugEnabled()) {
                int[] clusterSizes = new int[kMeansResult.centroids().length];
                for (int a : kMeansResult.assignments()) {
                    clusterSizes[a]++;
                }
                printClusterQualityStatistics(clusterSizes);
            }

            // TODO: swap out SOAR for SRAIR when HNSW graphs are used for the centroids
            return CentroidInformation.ofFloat(
                fieldInfo.getVectorDimension(),
                kMeansResult.centroids(),
                kMeansResult.assignments(),
                kMeansResult.overspill()
            );
        } finally {
            // CentroidData owns the IndexInput backing the streaming centroid view; close once
            // the clustering pass has consumed it (and on any failure mid-way).
            org.apache.lucene.util.IOUtils.closeWhileHandlingException(segmentCentroidData);
        }
    }

    private CentroidInformation<?> calculateCentroidsFullRebuildSliced(ClusteringVectorValues<?> vectorValues, MergeState mergeState)
        throws IOException {
        final FieldInfo slicedFieldInfo = mergeState.mergeFieldInfos.fieldInfo(sliceField);
        assert slicedFieldInfo != null;
        assert slicedFieldInfo.getDocValuesType() == DocValuesType.SORTED : "sliceField must be SortedDocValues";
        final SortedDocValues values = DocValueConsumerHelper.INSTANCE.getMergeSortedField(slicedFieldInfo, mergeState);
        final int numSlices = values.getValueCount();

        // Dispatch to generic typed implementation
        if (vectorValues instanceof ClusteringByteVectorValues byteVectorValues) {
            return calculateCentroidsFullRebuildSlicedTyped(byteVectorValues, CentroidOps.BYTE, values, numSlices);
        } else {
            return calculateCentroidsFullRebuildSlicedTyped((KMeansFloatVectorValues) vectorValues, CentroidOps.FLOAT, values, numSlices);
        }
    }

    private <V> CentroidInformation<V> calculateCentroidsFullRebuildSlicedTyped(
        ClusteringVectorValues<V> vectorValues,
        CentroidOps<V> ops,
        SortedDocValues values,
        int numSlices
    ) throws IOException {
        // For small slices, skip clustering and use a single flat centroid computed in the native type.
        if (vectorValues.size() / numSlices <= 4 * flatVectorThreshold) {
            final int dim = vectorValues.dimension();
            V nativeCentroid = ops.computeMeanCentroid(vectorValues, dim);
            V[] centroids = ops.newCentroidArrayShallow(1);
            centroids[0] = nativeCentroid;
            int[] assignments = new int[vectorValues.size()];
            return CentroidInformation.of(dim, centroids, assignments, OverspillAssignments.NONE, null, ops);
        }

        HierarchicalKMeans<V> hierarchicalKMeans;
        if (mergeExec != null) {
            hierarchicalKMeans = HierarchicalKMeans.ofConcurrent(ops, vectorValues.dimension(), mergeExec, numMergeWorkers);
        } else {
            hierarchicalKMeans = HierarchicalKMeans.ofSerial(ops, vectorValues.dimension());
        }
        final KnnVectorValues.DocIndexIterator iterator = ((KnnVectorValues) vectorValues).iterator();
        iterator.advance(0);
        values.nextDoc();
        // slice field must be dense populated, but we might have documents without a vector.
        final int[] sliceOffsets = new int[numSlices];
        final int[] sliceLengths = new int[numSlices];
        List<KMeansWithOverspill<V>> kmeansResults = new ArrayList<>();
        for (int i = 0; i < numSlices; i++) {
            if (iterator.docID() == DocIdSetIterator.NO_MORE_DOCS) {
                // no more vectors, we are done
                sliceLengths[i] = 0;
                sliceOffsets[i] = i == 0 ? 0 : sliceOffsets[i - 1];
                continue;
            }
            // get start and end of a slice
            int sliceDocStart = values.docID();
            while (values.docID() != DocIdSetIterator.NO_MORE_DOCS && values.ordValue() == i) {
                values.nextDoc();
            }
            final int sliceDocEnd = values.docID();
            // get the vector ordinals for the slice
            int vectorDocStart = iterator.docID();
            if (vectorDocStart < sliceDocStart) {
                // advance iterator to the beginning of the slice
                vectorDocStart = iterator.advance(sliceDocStart);
            }
            if (vectorDocStart > sliceDocEnd) {
                // no vectors in this slice
                sliceLengths[i] = 0;
                sliceOffsets[i] = i == 0 ? 0 : sliceOffsets[i - 1];
                continue;
            }
            final int vectorOrdStart = iterator.index();
            final int docEnd = vectorDocStart == sliceDocEnd ? sliceDocEnd : iterator.advance(sliceDocEnd);
            final int vectorOrdEnd = docEnd == KnnVectorValues.DocIndexIterator.NO_MORE_DOCS ? vectorValues.size() : iterator.index();
            final int sliceNumVectors = vectorOrdEnd - vectorOrdStart;
            final ClusteringVectorValues<V> slice = createSlice(vectorValues, j -> vectorOrdStart + j, sliceNumVectors);
            final KMeansNeighbors<V> kMeansResult = hierarchicalKMeans.cluster(slice, vectorPerCluster);
            final OverspillAssignments overspill = hierarchicalKMeans.computeSoar(
                slice,
                kMeansResult.result(),
                kMeansResult.neighborHoods()
            );
            kmeansResults.add(new KMeansWithOverspill<>(kMeansResult.result(), overspill));
            sliceLengths[i] = sliceNumVectors;
            sliceOffsets[i] = i == 0 ? kMeansResult.centroids().length : sliceOffsets[i - 1] + kMeansResult.centroids().length;
        }
        final KMeansWithOverspill<V> merged = KMeansWithOverspill.merge(kmeansResults, ops);
        if (logger.isDebugEnabled()) {
            logger.debug("final centroid count: {}", merged.centroids().length);
        }
        final CentroidSlices centroidSlices = new CentroidSlices(sliceOffsets, sliceLengths);
        return CentroidInformation.of(
            vectorValues.dimension(),
            merged.centroids(),
            merged.assignments(),
            merged.overspill(),
            centroidSlices,
            ops
        );
    }

    @SuppressWarnings("unchecked")
    private static <V> ClusteringVectorValues<V> createSlice(ClusteringVectorValues<V> values, IntToIntFunction ordTranslator, int size) {
        if (values instanceof ClusteringByteVectorValues byteValues) {
            return (ClusteringVectorValues<V>) new ClusteringByteVectorValuesSlice(byteValues, ordTranslator, size);
        } else {
            return (ClusteringVectorValues<V>) new ClusteringFloatVectorValuesSlice(
                (ClusteringFloatVectorValues) values,
                ordTranslator,
                size
            );
        }
    }

    // This class helps to access the merged view of a slice.
    @Override
    protected void writeSchemeMetaHead(IndexOutput metaOutput, IvfSegmentConfig ivfSegmentConfig) throws IOException {
        final IvfSegmentConfig segmentConfig = requireSegmentConfig(ivfSegmentConfig);
        metaOutput.writeInt(ES940OSQVectorsScorer.BULK_SIZE);
        metaOutput.writeInt(segmentConfig.centroidIndexFormat().id());
        metaOutput.writeInt(segmentConfig.osqEncoding().id());
    }

    @Override
    protected void writeSchemeMetaTail(IndexOutput metaOutput, FieldInfo field, IvfSegmentConfig ivfSegmentConfig, boolean byteCentroids)
        throws IOException {
        final IvfSegmentConfig segmentConfig = requireSegmentConfig(ivfSegmentConfig);
        metaOutput.writeInt(Float.floatToIntBits(segmentConfig.rescoreOversample()));
        // ESNext format extension: byte centroid flag — indicates whether raw centroids
        // are stored as 1 byte/dim (byte fields) or 4 bytes/dim (float fields).
        metaOutput.writeByte(byteCentroids ? (byte) 1 : (byte) 0);
    }

    @Override
    public CentroidInformation<?> calculateCentroids(FieldInfo fieldInfo, ClusteringVectorValues<?> vectorValues) throws IOException {
        if (sliceField != null) {
            // for sliced indexed, we don't cluster the data during flush so we can search our vectors by docId range
            return buildFlatCentroidAssignments(fieldInfo, vectorValues);
        }
        // For BYTE non-COSINE fields, cluster natively in byte space using SIMD byte distances.
        if (vectorValues instanceof ClusteringByteVectorValues byteVectorValues) {
            int dim = fieldInfo.getVectorDimension();
            HierarchicalKMeans<byte[]> byteKMeans = HierarchicalKMeans.ofSerial(CentroidOps.BYTE, dim);
            KMeansNeighbors<byte[]> kMeansResult = byteKMeans.cluster(byteVectorValues, vectorPerCluster);
            OverspillAssignments soarOverspill = byteKMeans.computeSoar(
                byteVectorValues,
                kMeansResult.result(),
                kMeansResult.neighborHoods()
            );
            if (logger.isDebugEnabled()) {
                logger.debug("native byte clustering: final centroid count: {}", kMeansResult.centroids().length);
            }
            return CentroidInformation.ofBytes(dim, kMeansResult.centroids(), kMeansResult.assignments(), soarOverspill);
        }
        ClusteringFloatVectorValues floatVectorValues = (ClusteringFloatVectorValues) vectorValues;
        HierarchicalKMeans<float[]> hierarchicalKMeans = HierarchicalKMeans.ofSerial(CentroidOps.FLOAT, floatVectorValues.dimension());
        KMeansNeighbors<float[]> kMeansResult = hierarchicalKMeans.cluster(floatVectorValues, vectorPerCluster);
        OverspillAssignments soarOverspill = hierarchicalKMeans.computeSoar(
            floatVectorValues,
            kMeansResult.result(),
            kMeansResult.neighborHoods()
        );
        if (logger.isDebugEnabled()) {
            logger.debug("final centroid count: {}", kMeansResult.centroids().length);
        }

        // TODO: swap out SOAR for SRAIR when HNSW graphs are used for the centroids
        return CentroidInformation.ofFloat(
            fieldInfo.getVectorDimension(),
            kMeansResult.centroids(),
            kMeansResult.assignments(),
            soarOverspill
        );
    }

    static void writeQuantizedValue(IndexOutput indexOutput, byte[] binaryValue, OptimizedScalarQuantizer.QuantizationResult corrections)
        throws IOException {
        indexOutput.writeBytes(binaryValue, binaryValue.length);
        indexOutput.writeInt(Float.floatToIntBits(corrections.lowerInterval()));
        indexOutput.writeInt(Float.floatToIntBits(corrections.upperInterval()));
        indexOutput.writeInt(Float.floatToIntBits(corrections.additionalCorrection()));
        indexOutput.writeInt(corrections.quantizedComponentSum());
    }

    /**
     * On-heap quantized vector values for float-encoded fields. Quantizes float vectors against
     * float centroids using the {@link OptimizedScalarQuantizer}.
     */
    static class OnHeapQuantizedVectors implements QuantizedVectorValues {
        private final FloatVectorValues vectorValues;
        private final OsqVectorQuantizer quantizer;
        private final byte[] quantizedVector;
        private final int[] quantizedVectorScratch;
        private final QuantEncoding encoding;
        private OptimizedScalarQuantizer.QuantizationResult corrections;
        private float[] currentCentroid, currentParentCentroid;
        private IntToIntFunction ordTransformer = null;
        private int currOrd = -1;
        private int count;

        OnHeapQuantizedVectors(FloatVectorValues vectorValues, QuantEncoding encoding, int dimension, OsqVectorQuantizer quantizer) {
            this.vectorValues = vectorValues;
            this.encoding = encoding;
            this.quantizer = quantizer;
            this.quantizedVector = new byte[encoding.getDocPackedLength(dimension)];
            this.quantizedVectorScratch = new int[encoding.discretizedDimensions(dimension)];
            this.corrections = null;
            this.currentParentCentroid = null;
        }

        private void reset(float[] centroid, float[] currentParentCentroid, int count, IntToIntFunction ordTransformer) {
            this.currentCentroid = centroid;
            this.ordTransformer = ordTransformer;
            this.currOrd = -1;
            this.count = count;
            this.currentParentCentroid = currentParentCentroid;
        }

        @Override
        public int count() {
            return count;
        }

        @Override
        public byte[] next() throws IOException {
            if (currOrd >= count() - 1) {
                throw new IllegalStateException("No more vectors to read, current ord: " + currOrd + ", count: " + count());
            }
            currOrd++;
            int ord = ordTransformer.apply(currOrd);
            // note, with a parent centroid, our correction needs to take it into account
            corrections = quantizer.quantize(vectorValues.vectorValue(ord), currentCentroid, currentParentCentroid, quantizedVectorScratch);
            encoding.pack(quantizedVectorScratch, quantizedVector);
            return quantizedVector;
        }

        @Override
        public OptimizedScalarQuantizer.QuantizationResult getCorrections() {
            if (currOrd == -1) {
                throw new IllegalStateException("No vector read yet, call next first");
            }
            return corrections;
        }
    }

    /**
     * On-heap quantized vector values for byte-encoded fields. Quantizes byte vectors against
     * byte centroids using the native byte {@link OptimizedScalarQuantizer} overload, avoiding
     * byte-to-float widening during quantization.
     */
    static class OnHeapQuantizedByteVectors implements QuantizedVectorValues {
        private final ByteVectorValues vectorValues;
        private final OsqVectorQuantizer quantizer;
        private final byte[] quantizedVector;
        private final int[] quantizedVectorScratch;
        private final QuantEncoding encoding;
        private OptimizedScalarQuantizer.QuantizationResult corrections;
        private byte[] currentCentroid;
        private float[] currentParentCentroid;
        private IntToIntFunction ordTransformer = null;
        private int currOrd = -1;
        private int count;

        OnHeapQuantizedByteVectors(ByteVectorValues vectorValues, QuantEncoding encoding, int dimension, OsqVectorQuantizer quantizer) {
            this.vectorValues = vectorValues;
            this.encoding = encoding;
            this.quantizer = quantizer;
            this.quantizedVector = new byte[encoding.getDocPackedLength(dimension)];
            this.quantizedVectorScratch = new int[encoding.discretizedDimensions(dimension)];
            this.corrections = null;
            this.currentParentCentroid = null;
        }

        private void reset(byte[] centroid, float[] currentParentCentroid, int count, IntToIntFunction ordTransformer) {
            this.currentCentroid = centroid;
            this.ordTransformer = ordTransformer;
            this.currOrd = -1;
            this.count = count;
            this.currentParentCentroid = currentParentCentroid;
        }

        @Override
        public int count() {
            return count;
        }

        @Override
        public byte[] next() throws IOException {
            if (currOrd >= count() - 1) {
                throw new IllegalStateException("No more vectors to read, current ord: " + currOrd + ", count: " + count());
            }
            currOrd++;
            int ord = ordTransformer.apply(currOrd);
            // Use the byte overload: quantize a byte vector against a byte centroid, without widening
            corrections = quantizer.quantize(vectorValues.vectorValue(ord), currentCentroid, currentParentCentroid, quantizedVectorScratch);
            encoding.pack(quantizedVectorScratch, quantizedVector);
            return quantizedVector;
        }

        @Override
        public OptimizedScalarQuantizer.QuantizationResult getCorrections() {
            if (currOrd == -1) {
                throw new IllegalStateException("No vector read yet, call next first");
            }
            return corrections;
        }
    }

    static class OffHeapQuantizedVectors implements QuantizedVectorValues {
        private final IndexInput quantizedVectorsInput;
        private final PackedLongValues vectorCentroidOffsets;
        private final byte[] binaryScratch;
        private final float[] corrections = new float[3];

        private final long vectorByteSize;
        private int bitSum;
        private int currOrd = -1;
        private int count;
        private IntToIntFunction overspillIdx = null;
        private IntToIntFunction ordTransformer = null;

        OffHeapQuantizedVectors(
            IndexInput quantizedVectorsInput,
            QuantEncoding encoding,
            int dimension,
            PackedLongValues vectorCentroidOffsets
        ) {
            this.quantizedVectorsInput = quantizedVectorsInput;
            this.vectorCentroidOffsets = vectorCentroidOffsets;
            this.binaryScratch = new byte[encoding.getDocPackedLength(dimension)];
            this.vectorByteSize = (binaryScratch.length + 3 * Float.BYTES + Integer.BYTES);
        }

        private void reset(int count, IntToIntFunction isOverspill, IntToIntFunction ordTransformer) {
            this.count = count;
            this.overspillIdx = isOverspill;
            this.ordTransformer = ordTransformer;
            this.currOrd = -1;
        }

        @Override
        public int count() {
            return count;
        }

        @Override
        public byte[] next() throws IOException {
            if (currOrd >= count - 1) {
                throw new IllegalStateException("No more vectors to read, current ord: " + currOrd + ", count: " + count);
            }
            currOrd++;
            int ord = ordTransformer.apply(currOrd);
            int centroidIdx = this.overspillIdx.apply(currOrd);
            return getVector(ord, centroidIdx);
        }

        @Override
        public OptimizedScalarQuantizer.QuantizationResult getCorrections() {
            if (currOrd == -1) {
                throw new IllegalStateException("No vector read yet, call readQuantizedVector first");
            }
            return new OptimizedScalarQuantizer.QuantizationResult(corrections[0], corrections[1], corrections[2], bitSum);
        }

        byte[] getVector(int ord, int centroidIdx) throws IOException {
            readQuantizedVector(ord, centroidIdx);
            return binaryScratch;
        }

        public void readQuantizedVector(int ord, int centroidIdx) throws IOException {
            long offset = vectorCentroidOffsets.get(ord) + centroidIdx * vectorByteSize;
            quantizedVectorsInput.seek(offset);
            quantizedVectorsInput.readBytes(binaryScratch, 0, binaryScratch.length);
            quantizedVectorsInput.readFloats(corrections, 0, 3);
            bitSum = quantizedVectorsInput.readInt();
        }
    }
}
