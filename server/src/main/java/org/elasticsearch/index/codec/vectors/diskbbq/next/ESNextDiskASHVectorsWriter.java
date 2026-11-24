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
import org.apache.lucene.store.IndexOutput;
import org.elasticsearch.core.SuppressForbidden;
import org.elasticsearch.core.WelfordVariance;
import org.elasticsearch.index.codec.vectors.ash.AshPostingsListWriter;
import org.elasticsearch.index.codec.vectors.ash.AshProjectionMatrix;
import org.elasticsearch.index.codec.vectors.cluster.CentroidOps;
import org.elasticsearch.index.codec.vectors.cluster.ClusteringFloatVectorValues;
import org.elasticsearch.index.codec.vectors.cluster.ClusteringFloatVectorValuesSlice;
import org.elasticsearch.index.codec.vectors.cluster.ClusteringVectorValues;
import org.elasticsearch.index.codec.vectors.cluster.HierarchicalKMeans;
import org.elasticsearch.index.codec.vectors.cluster.KMeansFloatVectorValues;
import org.elasticsearch.index.codec.vectors.cluster.KMeansNeighbors;
import org.elasticsearch.index.codec.vectors.cluster.KMeansWithOverspill;
import org.elasticsearch.index.codec.vectors.diskbbq.AshConfigResolver;
import org.elasticsearch.index.codec.vectors.diskbbq.CentroidIndexFormat;
import org.elasticsearch.index.codec.vectors.diskbbq.CentroidInformation;
import org.elasticsearch.index.codec.vectors.diskbbq.CentroidSlices;
import org.elasticsearch.index.codec.vectors.diskbbq.CentroidSupplier;
import org.elasticsearch.index.codec.vectors.diskbbq.IVFVectorsReader;
import org.elasticsearch.index.codec.vectors.diskbbq.IVFVectorsWriter;
import org.elasticsearch.index.codec.vectors.diskbbq.IvfFlushConfigSource;
import org.elasticsearch.index.codec.vectors.diskbbq.IvfMergeConfigResolver;
import org.elasticsearch.index.codec.vectors.diskbbq.IvfSegmentConfig;
import org.elasticsearch.index.codec.vectors.diskbbq.OverspillAssignments;
import org.elasticsearch.index.codec.vectors.diskbbq.Preconditioner;
import org.elasticsearch.index.codec.vectors.diskbbq.TieredMergeStrategy;
import org.elasticsearch.logging.LogManager;
import org.elasticsearch.logging.Logger;
import org.elasticsearch.simdvec.ES940OSQVectorsScorer;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * ASH-specific implementation of {@link IVFVectorsWriter}. Delegates quantized posting list
 * writing to {@link AshPostingsListWriter} and stores the learned projection matrix in the
 * preconditioner slot of the centroid file.
 */
public class ESNextDiskASHVectorsWriter extends ESNextIvfVectorsWriter {
    private static final Logger logger = LogManager.getLogger(ESNextDiskASHVectorsWriter.class);

    private final TaskExecutor quantizerExec;
    private final AshConfigResolver ashConfigResolver;

    // Temporary storage for ASH projection matrix between buildAndWritePostingsLists and writePreconditioner
    private AshProjectionMatrix pendingAshMatrix;

    public ESNextDiskASHVectorsWriter(
        SegmentWriteState state,
        String rawVectorFormatName,
        boolean useDirectIOReads,
        boolean onDiskMerge,
        FlatVectorsWriter rawVectorDelegate,
        int vectorPerCluster,
        int centroidsPerParentCluster,
        TaskExecutor mergeExec,
        int numMergeWorkers,
        TaskExecutor quantizerExec,
        int flatVectorThreshold,
        String sliceField,
        IvfFlushConfigSource flushConfigSource,
        IvfMergeConfigResolver mergeConfigResolver,
        AshConfigResolver ashConfigResolver
    ) throws IOException {
        super(
            state,
            rawVectorFormatName,
            useDirectIOReads,
            rawVectorDelegate,
            ESNextDiskASHVectorsFormat.VERSION_CURRENT,
            ESNextDiskASHVectorsFormat.NAME,
            ESNextDiskASHVectorsFormat.IVF_META_EXTENSION,
            ESNextDiskASHVectorsFormat.CENTROID_EXTENSION,
            ESNextDiskASHVectorsFormat.CLUSTER_EXTENSION,
            flatVectorThreshold,
            onDiskMerge,
            CentroidIndexFormat.FLAT,
            vectorPerCluster,
            centroidsPerParentCluster,
            mergeExec,
            numMergeWorkers,
            sliceField,
            flushConfigSource,
            mergeConfigResolver
        );
        this.quantizerExec = quantizerExec;
        this.ashConfigResolver = ashConfigResolver;
    }

    private IvfSegmentConfig.AshConfig ashConfig(FieldInfo fieldInfo) {
        return ashConfigResolver.resolve(fieldInfo.getVectorDimension());
    }

    @Override
    protected IvfSegmentConfig beginIvfFieldFlush(FieldInfo fieldInfo) throws IOException {
        return IvfSegmentConfig.fromCodecDefaults(CentroidIndexFormat.FLAT, ashConfig(fieldInfo), false);
    }

    @Override
    protected IvfSegmentConfig resolveMergeConfig(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
        return IvfSegmentConfig.fromCodecDefaults(CentroidIndexFormat.FLAT, ashConfig(fieldInfo), false);
    }

    @Override
    protected Preconditioner inheritPreconditioner(FieldInfo fieldInfo, MergeState mergeState, IvfSegmentConfig ivfSegmentConfig)
        throws IOException {
        // ASH trains a new W matrix on each merge; no inheritance needed
        return null;
    }

    @Override
    protected Preconditioner createPreconditioner(int dimension, IvfSegmentConfig ivfSegmentConfig) {
        // ASH writes its own projection matrix; no standard preconditioner needed
        return null;
    }

    @Override
    protected void writePreconditioner(Preconditioner preconditioner, IndexOutput out) throws IOException {
        if (pendingAshMatrix != null) {
            pendingAshMatrix.write(out);
            pendingAshMatrix = null;
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
        IvfSegmentConfig ivfSegmentConfig
    ) throws IOException {
        return buildAndWriteAshPostingsLists(
            fieldInfo,
            centroidSupplier,
            vectorValues,
            postingsOutput,
            fileOffset,
            assignments,
            overspillAssignments,
            ivfSegmentConfig,
            quantizerExec,
            null,
            false // flush: the flush training factor; the vectors are on-heap
        );
    }

    @Override
    public CentroidOffsetAndLength buildAndWritePostingsLists(
        FieldInfo fieldInfo,
        CentroidSupplier centroidSupplier,
        ClusteringVectorValues<?> vectorValues,
        IndexOutput postingsOutput,
        long fileOffset,
        MergeState mergeState,
        int[] assignments,
        OverspillAssignments overspillAssignments,
        IvfSegmentConfig ivfSegmentConfig
    ) throws IOException {
        // Reuse an already-trained projection matrix from one of the segments being merged instead of
        // re-learning W from scratch. W is a global orthonormal projection of centered/normalized
        // vectors and is effectively centroid-independent, so it can be carried across a merge.
        final AshProjectionMatrix inherited = inheritProjectionMatrix(fieldInfo, mergeState);
        return buildAndWriteAshPostingsLists(
            fieldInfo,
            centroidSupplier,
            vectorValues,
            postingsOutput,
            fileOffset,
            assignments,
            overspillAssignments,
            ivfSegmentConfig,
            quantizerExec,
            inherited,
            true // merge: the merge training factor; the vectors are streamed off-heap by ordinal
        );
    }

    /**
     * Finds a trained ASH projection matrix W among the segments being merged, returning it or {@code null} if none is
     * available. Prefers the input segment with the most vectors so W is seeded from the best-trained
     * source. Any reader that is not an {@link ESNextDiskASHVectorsReader} (or has no matrix for the
     * field) is skipped, and a {@code null} result causes the caller to fall back to training W.
     */
    private AshProjectionMatrix inheritProjectionMatrix(FieldInfo fieldInfo, MergeState mergeState) {
        int originalDim = fieldInfo.getVectorDimension();
        IvfSegmentConfig.AshConfig config = ashConfig(fieldInfo);
        if (inheritsProjection(config, originalDim) == false) {
            return null;
        }
        List<SizedProjectionMatrix> candidates = new ArrayList<>();
        for (int i = 0; i < mergeState.knnVectorsReaders.length; i++) {
            KnnVectorsReader reader = mergeState.knnVectorsReaders[i];
            if (reader == null) {
                continue;
            }
            if (reader instanceof PerFieldKnnVectorsFormat.FieldsReader fieldsReader) {
                reader = fieldsReader.getFieldReader(fieldInfo.name);
            }
            if (reader instanceof ESNextDiskASHVectorsReader ashReader && mergeState.fieldInfos[i].fieldInfo(fieldInfo.name) != null) {
                AshProjectionMatrix matrix = ashReader.getProjectionMatrix(fieldInfo);
                if (matrix == null) {
                    continue;
                }
                int size = 0;
                try {
                    FloatVectorValues values = ashReader.getFloatVectorValues(fieldInfo.name);
                    if (values != null) {
                        size = values.size();
                    }
                } catch (IOException e) {
                    // Fall back to using the matrix without a size preference if the count is unavailable.
                    size = 0;
                }
                candidates.add(new SizedProjectionMatrix(matrix, size));
            }
        }
        return selectInheritedProjectionMatrix(originalDim, config.projectedDims(originalDim), candidates);
    }

    /**
     * Returns whether merging a field under {@code config} should seed its projection from an input segment. Without
     * dimensionality reduction and without Procrustes iterations the projection is a seeded random rotation, the same
     * in every segment, so there is nothing to inherit and the input segments' matrices are not loaded at all.
     */
    static boolean inheritsProjection(IvfSegmentConfig.AshConfig config, int originalDim) {
        return config.projectedDims(originalDim) != originalDim || config.trainingIterations() > 0;
    }

    /**
     * A candidate projection matrix for merge inheritance together with the number of vectors in the
     * segment it came from (used to prefer the best-trained source).
     */
    record SizedProjectionMatrix(AshProjectionMatrix matrix, int size) {}

    /**
     * Selects the projection matrix to inherit at merge from the given candidates, or {@code null}
     * if none is suitable. Prefers the candidate from the segment with the most vectors so W is seeded
     * from the best-trained source. Skips random (non-learned) projections from transient flush segments
     * — inheriting a random rotation would carry it into the merged segment — and skips matrices whose
     * shape does not match {@code expectedDim} and {@code expectedProjectedDims}.
     * <p>
     * A shape mismatch is a valid situation rather than an error: the projected dimensions follow from the vector
     * dimension and the configuration, and segments written with another configuration can meet in a merge, such
     * as segments from snapshot builds before projected dimensions were derived from the vector dimension, or from a
     * fixed {@link IvfSegmentConfig.AshConfig}. Such a merge trains a new projection instead.
     */
    static AshProjectionMatrix selectInheritedProjectionMatrix(
        int expectedDim,
        int expectedProjectedDims,
        List<SizedProjectionMatrix> candidates
    ) {
        AshProjectionMatrix best = null;
        int bestSize = -1;
        for (SizedProjectionMatrix candidate : candidates) {
            AshProjectionMatrix matrix = candidate.matrix();
            if (matrix == null
                || matrix.isLearned() == false
                || matrix.originalDim() != expectedDim
                || matrix.nDims() != expectedProjectedDims) {
                continue;
            }
            if (candidate.size() > bestSize) {
                bestSize = candidate.size();
                best = matrix;
            }
        }
        return best;
    }

    private CentroidOffsetAndLength buildAndWriteAshPostingsLists(
        FieldInfo fieldInfo,
        CentroidSupplier centroidSupplier,
        ClusteringVectorValues<?> vectorValues,
        IndexOutput postingsOutput,
        long fileOffset,
        int[] assignments,
        OverspillAssignments overspillAssignments,
        IvfSegmentConfig segmentConfig,
        TaskExecutor quantizerExec,
        AshProjectionMatrix inherited,
        boolean useMergeTrainingFactor
    ) throws IOException {
        if (vectorValues instanceof FloatVectorValues == false) {
            throw new IllegalStateException("ASH requires float vectors, got: " + vectorValues.getClass().getSimpleName());
        }
        // In the sliced flush case (single centroid), skip writing per-block doc IDs.
        // The reader uses vector ordinal order for doc translation via ordToDoc().
        boolean skipDocIds = sliceField != null && centroidSupplier.size() == 1;
        var ashWriter = new AshPostingsListWriter();
        var result = ashWriter.buildAndWrite(
            fieldInfo,
            centroidSupplier,
            (FloatVectorValues) vectorValues,
            postingsOutput,
            fileOffset,
            assignments,
            overspillAssignments,
            segmentConfig.ashConfig(),
            quantizerExec,
            fieldInfo.getVectorSimilarityFunction(),
            skipDocIds,
            inherited,
            useMergeTrainingFactor
        );
        pendingAshMatrix = ashWriter.getAshProjectionMatrix();
        return new CentroidOffsetAndLength(result.offsets(), result.lengths());
    }

    @Override
    protected void writeSchemeMetaHead(IndexOutput metaOutput, IvfSegmentConfig ivfSegmentConfig) throws IOException {
        metaOutput.writeInt(ES940OSQVectorsScorer.BULK_SIZE);
        metaOutput.writeInt(CentroidIndexFormat.FLAT.id());
    }

    @Override
    protected void writeSchemeMetaTail(IndexOutput metaOutput, FieldInfo field, IvfSegmentConfig ivfSegmentConfig, boolean byteCentroids)
        throws IOException {
        // ASH-specific: bits per dimension — use the writer's own config rather than the segment
        // config, because the segment config may be IvfSegmentConfig.NONE for unsupported byte fields.
        metaOutput.writeVInt(ashConfig(field).bitsPerDim());
    }

    @Override
    public CentroidInformation<?> calculateCentroids(FieldInfo fieldInfo, ClusteringVectorValues<?> vectorValues) throws IOException {
        if (sliceField != null) {
            return buildFlatCentroidAssignments(fieldInfo, vectorValues);
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
            logger.debug("ASH flush: final centroid count: {}", kMeansResult.centroids().length);
        }
        return CentroidInformation.ofFloat(
            fieldInfo.getVectorDimension(),
            kMeansResult.centroids(),
            kMeansResult.assignments(),
            soarOverspill
        );
    }

    @Override
    @SuppressForbidden(reason = "require usage of Lucene's IOUtils#closeWhileHandlingException(...)")
    @SuppressWarnings({ "rawtypes", "unchecked" })
    public CentroidInformation<?> calculateCentroids(FieldInfo fieldInfo, ClusteringVectorValues<?> vectorValues, MergeState mergeState)
        throws IOException {
        if (sliceField != null) {
            return calculateCentroidsFullRebuildSliced(vectorValues, mergeState);
        }

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
                    "ASH merge for field [{}]: selected strategy [{}], segments={}, totalVectors={}, totalCentroids={}",
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

            return CentroidInformation.ofFloat(
                fieldInfo.getVectorDimension(),
                kMeansResult.centroids(),
                kMeansResult.assignments(),
                kMeansResult.overspill()
            );
        } finally {
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

        // ASH only supports float vectors
        return calculateCentroidsFullRebuildSlicedFloat((ClusteringFloatVectorValues) vectorValues, values, numSlices);
    }

    private CentroidInformation<float[]> calculateCentroidsFullRebuildSlicedFloat(
        ClusteringFloatVectorValues vectorValues,
        SortedDocValues values,
        int numSlices
    ) throws IOException {
        if (vectorValues.size() / numSlices <= 4 * flatVectorThreshold) {
            final int dim = vectorValues.dimension();
            float[] centroid = new float[dim];
            CentroidOps.FLOAT.accumulateAll(vectorValues, centroid);
            for (int d = 0; d < dim; d++) {
                centroid[d] /= vectorValues.size();
            }
            return CentroidInformation.ofFloat(dim, new float[][] { centroid }, new int[vectorValues.size()], OverspillAssignments.NONE);
        }

        HierarchicalKMeans<float[]> hierarchicalKMeans;
        if (mergeExec != null) {
            hierarchicalKMeans = HierarchicalKMeans.ofConcurrent(CentroidOps.FLOAT, vectorValues.dimension(), mergeExec, numMergeWorkers);
        } else {
            hierarchicalKMeans = HierarchicalKMeans.ofSerial(CentroidOps.FLOAT, vectorValues.dimension());
        }
        final KnnVectorValues.DocIndexIterator iterator = vectorValues.iterator();
        iterator.advance(0);
        values.nextDoc();
        final int[] sliceOffsets = new int[numSlices];
        final int[] sliceLengths = new int[numSlices];
        List<KMeansWithOverspill<float[]>> kmeansResults = new ArrayList<>();
        for (int i = 0; i < numSlices; i++) {
            if (iterator.docID() == DocIdSetIterator.NO_MORE_DOCS) {
                sliceLengths[i] = 0;
                sliceOffsets[i] = i == 0 ? 0 : sliceOffsets[i - 1];
                continue;
            }
            int sliceDocStart = values.docID();
            while (values.docID() != DocIdSetIterator.NO_MORE_DOCS && values.ordValue() == i) {
                values.nextDoc();
            }
            final int sliceDocEnd = values.docID();
            int vectorDocStart = iterator.docID();
            if (vectorDocStart < sliceDocStart) {
                vectorDocStart = iterator.advance(sliceDocStart);
            }
            if (vectorDocStart > sliceDocEnd) {
                sliceLengths[i] = 0;
                sliceOffsets[i] = i == 0 ? 0 : sliceOffsets[i - 1];
                continue;
            }
            final int vectorOrdStart = iterator.index();
            final int docEnd = vectorDocStart == sliceDocEnd ? sliceDocEnd : iterator.advance(sliceDocEnd);
            final int vectorOrdEnd = docEnd == KnnVectorValues.DocIndexIterator.NO_MORE_DOCS ? vectorValues.size() : iterator.index();
            final int sliceNumVectors = vectorOrdEnd - vectorOrdStart;
            final ClusteringFloatVectorValues slice = new ClusteringFloatVectorValuesSlice(
                vectorValues,
                j -> vectorOrdStart + j,
                sliceNumVectors
            );
            final KMeansNeighbors<float[]> kMeansResult = hierarchicalKMeans.cluster(slice, vectorPerCluster);
            final OverspillAssignments overspill = hierarchicalKMeans.computeSoar(
                slice,
                kMeansResult.result(),
                kMeansResult.neighborHoods()
            );
            kmeansResults.add(new KMeansWithOverspill<>(kMeansResult.result(), overspill));
            sliceLengths[i] = sliceNumVectors;
            sliceOffsets[i] = i == 0 ? kMeansResult.centroids().length : sliceOffsets[i - 1] + kMeansResult.centroids().length;
        }
        final KMeansWithOverspill<float[]> merged = KMeansWithOverspill.merge(kmeansResults, CentroidOps.FLOAT);
        if (logger.isDebugEnabled()) {
            logger.debug("ASH sliced merge: final centroid count: {}", merged.centroids().length);
        }
        final CentroidSlices centroidSlices = new CentroidSlices(sliceOffsets, sliceLengths);
        return CentroidInformation.ofFloat(
            vectorValues.dimension(),
            merged.centroids(),
            merged.assignments(),
            merged.overspill(),
            centroidSlices
        );
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

}
