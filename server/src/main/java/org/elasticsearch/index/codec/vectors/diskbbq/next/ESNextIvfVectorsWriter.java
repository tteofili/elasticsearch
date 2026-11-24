/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq.next;

import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.search.TaskExecutor;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.elasticsearch.index.codec.vectors.cluster.CentroidOps;
import org.elasticsearch.index.codec.vectors.cluster.ClusteringFloatVectorValues;
import org.elasticsearch.index.codec.vectors.cluster.ClusteringFloatVectorValuesSlice;
import org.elasticsearch.index.codec.vectors.cluster.HierarchicalKMeans;
import org.elasticsearch.index.codec.vectors.cluster.KMeansResult;
import org.elasticsearch.index.codec.vectors.diskbbq.CentroidAssignments;
import org.elasticsearch.index.codec.vectors.diskbbq.CentroidIndexFormat;
import org.elasticsearch.index.codec.vectors.diskbbq.CentroidSlices;
import org.elasticsearch.index.codec.vectors.diskbbq.CentroidSupplier;
import org.elasticsearch.index.codec.vectors.diskbbq.FlatCentroidClusters;
import org.elasticsearch.index.codec.vectors.diskbbq.FlatCentroidIndexWriter;
import org.elasticsearch.index.codec.vectors.diskbbq.IVFVectorsWriter;
import org.elasticsearch.index.codec.vectors.diskbbq.IvfFlushConfigSource;
import org.elasticsearch.index.codec.vectors.diskbbq.IvfMergeConfigResolver;
import org.elasticsearch.index.codec.vectors.diskbbq.IvfSegmentConfig;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * The part of an {@code ESNext} IVF writer that does not depend on how vectors are quantized: how the centroids of a
 * segment are indexed and how many vectors a cluster holds, how merges are parallelized, and how the second level of
 * clusters is built. What a quantizer adds is the encoding of the vectors of each posting list and the metadata of its
 * fields.
 */
public abstract class ESNextIvfVectorsWriter extends IVFVectorsWriter<FlatCentroidIndexWriter.CentroidGroups> {

    protected final int vectorPerCluster;
    protected final CentroidIndexFormat centroidIndexFormat;
    protected final int centroidsPerParentCluster;
    protected final TaskExecutor mergeExec;
    protected final int numMergeWorkers;
    // field for slicing, null for no slicing
    protected final String sliceField;
    protected final IvfFlushConfigSource flushConfigSource;
    protected final IvfMergeConfigResolver mergeConfigResolver;

    protected ESNextIvfVectorsWriter(
        SegmentWriteState state,
        String rawVectorFormatName,
        boolean useDirectIOReads,
        FlatVectorsWriter rawVectorDelegate,
        int writeVersion,
        String codecName,
        String metaExtension,
        String centroidExtension,
        String clusterExtension,
        int flatVectorThreshold,
        boolean onDiskMerge,
        CentroidIndexFormat centroidIndexFormat,
        int vectorPerCluster,
        int centroidsPerParentCluster,
        TaskExecutor mergeExec,
        int numMergeWorkers,
        String sliceField,
        IvfFlushConfigSource flushConfigSource,
        IvfMergeConfigResolver mergeConfigResolver
    ) throws IOException {
        super(
            state,
            rawVectorFormatName,
            useDirectIOReads,
            rawVectorDelegate,
            writeVersion,
            codecName,
            metaExtension,
            centroidExtension,
            clusterExtension,
            true,
            flatVectorThreshold,
            onDiskMerge,
            true
        );
        this.vectorPerCluster = vectorPerCluster;
        this.centroidIndexFormat = centroidIndexFormat;
        this.centroidsPerParentCluster = centroidsPerParentCluster;
        this.mergeExec = mergeExec;
        this.numMergeWorkers = numMergeWorkers;
        this.sliceField = sliceField;
        this.flushConfigSource = flushConfigSource != null ? flushConfigSource : IvfFlushConfigSource.empty();
        this.mergeConfigResolver = mergeConfigResolver != null ? mergeConfigResolver : IvfMergeConfigResolver.useCodecDefault();
    }

    protected KMeansResult<float[]> buildSecondLevelClusters(
        FieldInfo fieldInfo,
        ClusteringFloatVectorValues floatVectorValues,
        boolean isMerge
    ) throws IOException {
        // we use the HierarchicalKMeans to partition the space of all vectors across merging segments
        // this are small numbers so we run it wih all the centroids.
        HierarchicalKMeans<float[]> hierarchicalKMeans;
        if (isMerge && mergeExec != null) {
            hierarchicalKMeans = HierarchicalKMeans.ofConcurrent(
                CentroidOps.FLOAT,
                fieldInfo.getVectorDimension(),
                mergeExec,
                numMergeWorkers,
                HierarchicalKMeans.MAX_ITERATIONS_DEFAULT,
                HierarchicalKMeans.SAMPLES_PER_CLUSTER_DEFAULT,
                HierarchicalKMeans.MAXK
            );
        } else {
            hierarchicalKMeans = HierarchicalKMeans.ofSerial(
                CentroidOps.FLOAT,
                fieldInfo.getVectorDimension(),
                HierarchicalKMeans.MAX_ITERATIONS_DEFAULT,
                HierarchicalKMeans.SAMPLES_PER_CLUSTER_DEFAULT,
                HierarchicalKMeans.MAXK
            );
        }
        return hierarchicalKMeans.cluster(floatVectorValues, centroidsPerParentCluster).result();
    }

    @Override
    public CentroidSupplier createCentroidSupplier(IndexInput centroidsInput, CentroidAssignments centroidAssignments, FieldInfo fieldInfo)
        throws IOException {
        int numCentroids = centroidAssignments.numCentroids();
        float[] globalCentroid = centroidAssignments.globalCentroid();
        CentroidSlices centroidSlices = centroidAssignments.centroidSlices();

        CentroidSupplier centroidSupplier = new OffHeapCentroidSupplier(
            centroidsInput,
            numCentroids,
            fieldInfo,
            KMeansResult.singleCluster(globalCentroid, numCentroids),
            centroidSlices
        );
        if (centroidSupplier.size() > centroidsPerParentCluster * centroidsPerParentCluster) {
            ClusteringFloatVectorValues floatVectorValues = centroidSupplier.asKmeansFloatVectorValues();
            if (centroidSlices == null) {
                KMeansResult<float[]> centroidClusters = buildSecondLevelClusters(fieldInfo, floatVectorValues, true);
                return new OffHeapCentroidSupplier(centroidsInput, numCentroids, fieldInfo, centroidClusters, null);
            } else {
                List<KMeansResult<float[]>> centroidClusters = new ArrayList<>(centroidSlices.sliceOffsets().length);
                int start = 0;
                for (int i = 0; i < centroidSlices.sliceOffsets().length; i++) {
                    final int offset = start;
                    start = centroidSlices.sliceOffsets()[i];
                    int count = start - offset;
                    ClusteringFloatVectorValues slice = new ClusteringFloatVectorValuesSlice(floatVectorValues, j -> offset + j, count);
                    KMeansResult<float[]> result = buildSecondLevelClusters(fieldInfo, slice, true);
                    centroidClusters.add(result);
                    if (i == 0) {
                        centroidSlices.sliceOffsets()[i] = result.centroids().length;
                    } else {
                        centroidSlices.sliceOffsets()[i] = centroidSlices.sliceOffsets()[i - 1] + result.centroids().length;
                    }
                }
                KMeansResult<float[]> result = KMeansResult.merge(centroidClusters, CentroidOps.FLOAT);
                assert CentroidSlices.assertSliceOffsets(centroidSlices.sliceOffsets(), result.centroids().length);
                return new OffHeapCentroidSupplier(centroidsInput, numCentroids, fieldInfo, result, centroidSlices);
            }
        }
        return centroidSupplier;
    }

    @Override
    public CentroidSupplier createCentroidSupplier(FieldInfo info, float[][] centroids, float[] globalCentroid) throws IOException {
        CentroidSupplier centroidSupplier = CentroidSupplier.fromArray(
            centroids,
            new FlatCentroidClusters(KMeansResult.singleCluster(globalCentroid, centroids.length)),
            info.getVectorDimension()
        );
        if (centroidSupplier.size() > centroidsPerParentCluster * centroidsPerParentCluster) {
            KMeansResult<float[]> centroidClusters = buildSecondLevelClusters(info, centroidSupplier.asKmeansFloatVectorValues(), false);
            return CentroidSupplier.fromArray(centroids, new FlatCentroidClusters(centroidClusters), info.getVectorDimension());
        }
        return centroidSupplier;
    }

    @Override
    protected void writeCentroidData(
        FieldInfo fieldInfo,
        CentroidSupplier centroidSupplier,
        float[] globalCentroid,
        CentroidOffsetAndLength centroidOffsetAndLength,
        FlatCentroidIndexWriter.CentroidGroups centroidGroups,
        IndexOutput centroidOutput
    ) throws IOException {
        switch (centroidIndexFormat) {
            case FLAT -> FlatCentroidIndexWriter.writeCentroidData(
                fieldInfo,
                centroidSupplier,
                globalCentroid,
                centroidOffsetAndLength,
                centroidGroups,
                centroidOutput
            );
        }
    }

    /**
     * Writes the metadata of a field. The layout is part of the format and interleaves what every quantizer writes with
     * what only one does, in this order: the scheme's head ({@link #writeSchemeMetaHead}), the preconditioner or
     * projection slot, the slices, and the scheme's tail ({@link #writeSchemeMetaTail}).
     */
    @Override
    protected void doWriteMeta(
        IndexOutput metaOutput,
        FieldInfo field,
        int numCentroids,
        long preconditionerOffset,
        long preconditionerLength,
        int numberOfSlices,
        int maxSliceSize,
        IvfSegmentConfig ivfSegmentConfig,
        boolean byteCentroids
    ) throws IOException {
        writeSchemeMetaHead(metaOutput, ivfSegmentConfig);
        metaOutput.writeLong(preconditionerLength);
        if (preconditionerLength > 0) {
            metaOutput.writeLong(preconditionerOffset);
        }
        if (sliceField == null) {
            assert numberOfSlices == 0;
            metaOutput.writeInt(-1);
        } else {
            metaOutput.writeInt(numberOfSlices);
            if (numberOfSlices > 0) {
                metaOutput.writeVInt(maxSliceSize);
            }
        }
        writeSchemeMetaTail(metaOutput, field, ivfSegmentConfig, byteCentroids);
    }

    /** Writes what the quantizer puts at the start of the metadata of a field: the bulk size and what it needs to read its codes. */
    protected abstract void writeSchemeMetaHead(IndexOutput metaOutput, IvfSegmentConfig ivfSegmentConfig) throws IOException;

    /** Writes what the quantizer puts at the end of the metadata of a field, after the slices. */
    protected abstract void writeSchemeMetaTail(
        IndexOutput metaOutput,
        FieldInfo field,
        IvfSegmentConfig ivfSegmentConfig,
        boolean byteCentroids
    ) throws IOException;

    @Override
    protected FlatCentroidIndexWriter.CentroidGroups writeCentroidIndex(
        CentroidSupplier centroidSupplier,
        int[] centroidAssignments,
        IndexOutput centroidOutput
    ) throws IOException {
        return switch (centroidIndexFormat) {
            case FLAT -> FlatCentroidIndexWriter.writeCentroidIndex(centroidSupplier, centroidAssignments, centroidOutput);
        };
    }
}
