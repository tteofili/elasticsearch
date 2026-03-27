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
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.TaskExecutor;
import org.apache.lucene.store.ByteBuffersDataOutput;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.hnsw.IntToIntFunction;
import org.apache.lucene.util.packed.DirectWriter;
import org.apache.lucene.util.packed.PackedInts;
import org.apache.lucene.util.packed.PackedLongValues;
import org.elasticsearch.core.SuppressForbidden;
import org.elasticsearch.core.TimeValue;
import org.elasticsearch.index.codec.vectors.OptimizedScalarQuantizer;
import org.elasticsearch.index.codec.vectors.cluster.HierarchicalKMeans;
import org.elasticsearch.index.codec.vectors.cluster.KMeansFloatVectorValues;
import org.elasticsearch.index.codec.vectors.cluster.KMeansResult;
import org.elasticsearch.index.codec.vectors.diskbbq.CentroidAssignments;
import org.elasticsearch.index.codec.vectors.diskbbq.CentroidSupplier;
import org.elasticsearch.index.codec.vectors.diskbbq.DiskBBQBulkWriter;
import org.elasticsearch.index.codec.vectors.diskbbq.DocIdsWriter;
import org.elasticsearch.index.codec.vectors.diskbbq.IVFVectorsWriter;
import org.elasticsearch.index.codec.vectors.diskbbq.IntSorter;
import org.elasticsearch.index.codec.vectors.diskbbq.IntToBooleanFunction;
import org.elasticsearch.index.codec.vectors.diskbbq.Preconditioner;
import org.elasticsearch.index.codec.vectors.diskbbq.QuantizedVectorValues;
import org.elasticsearch.index.codec.vectors.diskbbq.VectorPreconditioner;
import org.elasticsearch.logging.LogManager;
import org.elasticsearch.logging.Logger;
import org.elasticsearch.simdvec.ESNextOSQVectorsScorer;
import org.elasticsearch.simdvec.ESVectorUtil;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.function.Consumer;
import java.util.function.IntUnaryOperator;

import static org.elasticsearch.index.codec.vectors.cluster.HierarchicalKMeans.NO_SOAR_ASSIGNMENT;
import static org.elasticsearch.simdvec.ESNextOSQVectorsScorer.BULK_SIZE;

/**
 * Default implementation of {@link IVFVectorsWriter}. It uses {@link HierarchicalKMeans} algorithm to
 * partition the vector space, and then stores the centroids and posting list in a sequential
 * fashion.
 */
public class ESNextDiskBBQVectorsWriter extends IVFVectorsWriter {
    private static final Logger logger = LogManager.getLogger(ESNextDiskBBQVectorsWriter.class);

    private final int vectorPerCluster;
    private final int centroidsPerParentCluster;
    private final ESNextDiskBBQVectorsFormat.QuantEncoding quantEncoding;
    private final boolean quantizationAuto;
    private final AutoQuantizationSelector autoQuantizationSelector;
    private final TaskExecutor mergeExec;
    private final int numMergeWorkers;
    private final int blockDimension;
    private final boolean doPrecondition;

    /**
     * Resolved calibration result for the current field during buildAndWritePostingsLists; used when quantizationAuto is true.
     */
    private AutoQuantizationSelector.CalibrationResult calibrationResultForCurrentField;

    /**
     * Preconditioner stored for deferred application when quantizationAuto is true.
     * Created eagerly but only applied after calibration decides whether preconditioning helps.
     */
    private Preconditioner savedPreconditioner;

    /**
     * Reference to the global centroid array for the current field, stored so that deferred
     * preconditioning can modify it in-place before writeMeta writes it to disk.
     */
    private float[] savedGlobalCentroid;
    private long autoQuantizationCalibrationNanosThisMergeField;

    public ESNextDiskBBQVectorsWriter(
        SegmentWriteState state,
        String rawVectorFormatName,
        boolean useDirectIOReads,
        FlatVectorsWriter rawVectorDelegate,
        ESNextDiskBBQVectorsFormat.QuantEncoding encoding,
        boolean quantizationAuto,
        AutoQuantizationSelector autoQuantizationSelector,
        int vectorPerCluster,
        int centroidsPerParentCluster,
        TaskExecutor mergeExec,
        int numMergeWorkers,
        int blockDimension,
        boolean doPrecondition,
        int flatVectorThreshold
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
            true,
            flatVectorThreshold
        );
        this.vectorPerCluster = vectorPerCluster;
        this.centroidsPerParentCluster = centroidsPerParentCluster;
        this.quantEncoding = encoding;
        this.quantizationAuto = quantizationAuto;
        this.autoQuantizationSelector = autoQuantizationSelector != null
            ? autoQuantizationSelector
            : NoOpAutomaticQuantizationSelector.INSTANCE;
        this.mergeExec = mergeExec;
        this.numMergeWorkers = numMergeWorkers;
        this.blockDimension = blockDimension;
        this.doPrecondition = doPrecondition;
    }

    @Override
    protected void mergeIvfStarted(MergeState mergeState) {
        autoQuantizationCalibrationNanosThisMergeField = 0L;
    }

    @Override
    protected void mergeIvfCompleted(MergeState mergeState, FieldInfo fieldInfo, long totalMergeNanos) {
        if (quantizationAuto) {
            long calibrationNanos = autoQuantizationCalibrationNanosThisMergeField;
            double calibrationPercent = totalMergeNanos > 0 ? (100.0 * calibrationNanos / totalMergeNanos) : 0.0;
            logger.info(
                "diskbbq IVF merge timing segment=[{}] field=[{}] total=[{}] calibration=[{}] calibrationPercent=[{}%]",
                mergeState.segmentInfo.name,
                fieldInfo.name,
                TimeValue.timeValueNanos(totalMergeNanos),
                TimeValue.timeValueNanos(calibrationNanos),
                String.format(Locale.ROOT, "%.1f", calibrationPercent)
            );
        }
    }

    @Override
    protected Preconditioner inheritPreconditioner(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
        if (doPrecondition) {
            for (KnnVectorsReader reader : mergeState.knnVectorsReaders) {
                if (reader instanceof VectorPreconditioner) {
                    Preconditioner preconditioner = ((VectorPreconditioner) reader).getPreconditioner(fieldInfo);
                    if (preconditioner != null) {
                        if (quantizationAuto) {
                            savedPreconditioner = preconditioner;
                        }
                        return preconditioner;
                    }
                }
            }
            return createPreconditioner(fieldInfo.getVectorDimension());
        }
        return null;
    }

    @Override
    protected Preconditioner createPreconditioner(int dimension) {
        if (doPrecondition) {
            Preconditioner p = Preconditioner.createPreconditioner(dimension, blockDimension);
            if (quantizationAuto) {
                savedPreconditioner = p;
            }
            return p;
        }
        return null;
    }

    @Override
    protected void writePreconditioner(Preconditioner preconditioner, IndexOutput out) throws IOException {
        if (quantizationAuto) {
            if (calibrationResultForCurrentField != null && calibrationResultForCurrentField.doPrecondition() && preconditioner != null) {
                preconditioner.write(out);
            }
            return;
        }
        if (preconditioner != null) {
            preconditioner.write(out);
        }
    }

    @Override
    protected Consumer<List<float[]>> preconditionVectors(Preconditioner preconditioner) {
        if (quantizationAuto) {
            return (vectors) -> {};
        }
        return (vectors) -> {
            if (doPrecondition == false || vectors.isEmpty()) {
                return;
            }
            if (preconditioner == null) {
                throw new IllegalStateException("preconditioner was not created but should be first");
            }
            float[] out = new float[vectors.getFirst().length];
            for (int i = 0; i < vectors.size(); i++) {
                float[] vector = vectors.get(i);
                preconditioner.applyTransform(vector, out);
                System.arraycopy(out, 0, vector, 0, vector.length);
            }
        };
    }

    @Override
    protected FloatVectorValues preconditionVectors(Preconditioner preconditioner, FloatVectorValues vectors) {
        if (quantizationAuto) {
            if (preconditioner != null) {
                savedPreconditioner = preconditioner;
            }
            return vectors;
        }
        if (doPrecondition == false) {
            return vectors;
        }
        if (preconditioner == null) {
            throw new IllegalStateException("preconditioner was not created but should be first");
        }

        // TODO: batch apply preconditioner for better performance and keep a batch on heap at a time
        return new FloatVectorValues() {
            final float[] preconditionedVectorValue = new float[vectors.dimension()];
            int cachedOrd = -1;

            @Override
            public float[] vectorValue(int ord) throws IOException {
                assert ord != -1;
                if (ord != cachedOrd) {
                    float[] vectorValue = vectors.vectorValue(ord);
                    preconditioner.applyTransform(vectorValue, this.preconditionedVectorValue);
                    cachedOrd = ord;
                }
                return this.preconditionedVectorValue;
            }

            @Override
            public FloatVectorValues copy() throws IOException {
                return vectors.copy();
            }

            @Override
            public int dimension() {
                return vectors.dimension();
            }

            @Override
            public int size() {
                return vectors.size();
            }

            @Override
            public DocIndexIterator iterator() {
                return vectors.iterator();
            }
        };
    }

    @Override
    public CentroidOffsetAndLength buildAndWritePostingsLists(
        FieldInfo fieldInfo,
        CentroidSupplier centroidSupplier,
        FloatVectorValues floatVectorValues,
        IndexOutput postingsOutput,
        long fileOffset,
        int[] assignments,
        int[] overspillAssignments
    ) throws IOException {
        final long calibrationStartNanos = System.nanoTime();
        final AutoQuantizationSelector.CalibrationResult calibrationResult = resolveCalibrationResult(
            fieldInfo,
            floatVectorValues,
            centroidSupplier,
            assignments,
            overspillAssignments,
            null
        );
        if (quantizationAuto) {
            logAutoQuantizationCalibration(
                fieldInfo,
                floatVectorValues,
                null,
                null,
                System.nanoTime() - calibrationStartNanos,
                calibrationResult
            );
        }
        calibrationResultForCurrentField = calibrationResult;
        final ESNextDiskBBQVectorsFormat.QuantEncoding effectiveEncoding = calibrationResult.encoding();

        CentroidSupplier effectiveCS = centroidSupplier;
        FloatVectorValues effectiveFvv = floatVectorValues;
        if (shouldApplyDeferredPreconditioning()) {
            preconditionClusterCentroidsInPlace(centroidSupplier, fieldInfo.getVectorDimension());
            effectiveCS = wrapCentroidSupplierWithPreconditioner(centroidSupplier, fieldInfo.getVectorDimension());
            effectiveFvv = wrapFvvWithPreconditioner(floatVectorValues);
        }

        KMeansResult centroidClusters = effectiveCS.secondLevelClusters();
        int[] centroidVectorCount = new int[effectiveCS.size()];
        for (int i = 0; i < assignments.length; i++) {
            centroidVectorCount[assignments[i]]++;
            if (overspillAssignments.length > i && overspillAssignments[i] != NO_SOAR_ASSIGNMENT) {
                centroidVectorCount[overspillAssignments[i]]++;
            }
        }

        int maxPostingListSize = 0;
        int[][] assignmentsByCluster = new int[effectiveCS.size()][];
        for (int c = 0; c < effectiveCS.size(); c++) {
            int size = centroidVectorCount[c];
            maxPostingListSize = Math.max(maxPostingListSize, size);
            assignmentsByCluster[c] = new int[size];
        }
        Arrays.fill(centroidVectorCount, 0);

        for (int i = 0; i < assignments.length; i++) {
            int c = assignments[i];
            assignmentsByCluster[c][centroidVectorCount[c]++] = i;
            if (overspillAssignments.length > i) {
                int s = overspillAssignments[i];
                if (s != NO_SOAR_ASSIGNMENT) {
                    assignmentsByCluster[s][centroidVectorCount[s]++] = i;
                }
            }
        }
        final PackedLongValues.Builder offsets = PackedLongValues.monotonicBuilder(PackedInts.COMPACT);
        final PackedLongValues.Builder lengths = PackedLongValues.monotonicBuilder(PackedInts.COMPACT);
        DiskBBQBulkWriter bulkWriter = DiskBBQBulkWriter.fromBitSize(effectiveEncoding.bits(), BULK_SIZE, postingsOutput, true, true);
        OnHeapQuantizedVectors onHeapQuantizedVectors = new OnHeapQuantizedVectors(
            effectiveFvv,
            fieldInfo.getVectorSimilarityFunction(),
            effectiveEncoding,
            fieldInfo.getVectorDimension(),
            new OptimizedScalarQuantizer(fieldInfo.getVectorSimilarityFunction())
        );
        final int[] docIds = new int[maxPostingListSize];
        final int[] docDeltas = new int[maxPostingListSize];
        final int[] clusterOrds = new int[maxPostingListSize];
        DocIdsWriter idsWriter = new DocIdsWriter();
        for (int c = 0; c < effectiveCS.size(); c++) {
            float[] centroid = effectiveCS.centroid(c);
            int[] cluster = assignmentsByCluster[c];
            long offset = postingsOutput.alignFilePointer(Float.BYTES) - fileOffset;
            offsets.add(offset);
            postingsOutput.writeInt(Float.floatToIntBits(ESVectorUtil.squareDistance(centroid, centroidClusters.getCentroid(c))));
            int size = cluster.length;
            postingsOutput.writeVInt(size);
            for (int j = 0; j < size; j++) {
                docIds[j] = effectiveFvv.ordToDoc(cluster[j]);
                clusterOrds[j] = j;
            }
            new IntSorter(clusterOrds, i -> docIds[i]).sort(0, size);
            for (int j = 0; j < size; j++) {
                docDeltas[j] = j == 0 ? docIds[clusterOrds[j]] : docIds[clusterOrds[j]] - docIds[clusterOrds[j - 1]];
            }
            onHeapQuantizedVectors.reset(centroid, centroidClusters.getCentroid(c), size, ord -> cluster[clusterOrds[ord]]);
            byte encoding = idsWriter.calculateBlockEncoding(i -> docDeltas[i], size, BULK_SIZE);
            postingsOutput.writeByte(encoding);
            bulkWriter.writeVectors(onHeapQuantizedVectors, i -> {
                idsWriter.writeDocIds(d -> docDeltas[i + d], Math.min(BULK_SIZE, size - i), encoding, postingsOutput);
            });
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
        FloatVectorValues floatVectorValues,
        IndexOutput postingsOutput,
        long fileOffset,
        MergeState mergeState,
        int[] assignments,
        int[] overspillAssignments
    ) throws IOException {
        final long calibrationStartNanos = System.nanoTime();
        final MergeCalibrationContext mergeCalibrationContext = MergeCalibrationContext.from(mergeState);
        final AutoQuantizationSelector.CalibrationResult calibrationResult = resolveCalibrationResult(
            fieldInfo,
            floatVectorValues,
            centroidSupplier,
            assignments,
            overspillAssignments,
            mergeState
        );
        if (quantizationAuto) {
            final long calibrationNanos = System.nanoTime() - calibrationStartNanos;
            autoQuantizationCalibrationNanosThisMergeField += calibrationNanos;
            logAutoQuantizationCalibration(
                fieldInfo,
                floatVectorValues,
                mergeState,
                mergeCalibrationContext,
                calibrationNanos,
                calibrationResult
            );
        }
        calibrationResultForCurrentField = calibrationResult;
        final ESNextDiskBBQVectorsFormat.QuantEncoding effectiveEncoding = calibrationResult.encoding();

        CentroidSupplier effectiveCS = centroidSupplier;
        FloatVectorValues effectiveFvv = floatVectorValues;
        if (shouldApplyDeferredPreconditioning()) {
            preconditionClusterCentroidsInPlace(centroidSupplier, fieldInfo.getVectorDimension());
            effectiveCS = wrapCentroidSupplierWithPreconditioner(centroidSupplier, fieldInfo.getVectorDimension());
            effectiveFvv = wrapFvvWithPreconditioner(floatVectorValues);
        }

        var vectorSimilarityFunction = fieldInfo.getVectorSimilarityFunction();
        KMeansResult centroidClusters = effectiveCS.secondLevelClusters();
        String quantizedVectorsTempName = null;
        try (
            IndexOutput quantizedVectorsTemp = mergeState.segmentInfo.dir.createTempOutput(
                mergeState.segmentInfo.name,
                "qvec_",
                IOContext.DEFAULT
            )
        ) {
            quantizedVectorsTempName = quantizedVectorsTemp.getName();
            OptimizedScalarQuantizer quantizer = new OptimizedScalarQuantizer(vectorSimilarityFunction);
            int[] quantized = new int[effectiveEncoding.discretizedDimensions(fieldInfo.getVectorDimension())];
            byte[] binary = new byte[effectiveEncoding.getDocPackedLength(fieldInfo.getVectorDimension())];
            float[] scratch = new float[fieldInfo.getVectorDimension()];
            for (int i = 0; i < assignments.length; i++) {
                int c = assignments[i];
                float[] centroid = effectiveCS.centroid(c);
                float[] parentCentroid = centroidClusters.getCentroid(c);
                float[] vector = effectiveFvv.vectorValue(i);
                boolean overspill = overspillAssignments.length > i && overspillAssignments[i] != NO_SOAR_ASSIGNMENT;
                OptimizedScalarQuantizer.QuantizationResult result = quantizer.scalarQuantize(
                    vector,
                    scratch,
                    quantized,
                    effectiveEncoding.bits(),
                    centroid
                );
                if (parentCentroid != null) {
                    float additionalCorrection = vectorSimilarityFunction == VectorSimilarityFunction.EUCLIDEAN
                        ? ESVectorUtil.squareDistance(vector, parentCentroid)
                        : ESVectorUtil.dotProduct(scratch, parentCentroid);
                    result = new OptimizedScalarQuantizer.QuantizationResult(
                        result.lowerInterval(),
                        result.upperInterval(),
                        additionalCorrection,
                        result.quantizedComponentSum()
                    );
                }
                effectiveEncoding.pack(quantized, binary);
                writeQuantizedValue(quantizedVectorsTemp, binary, result);
                if (overspill) {
                    int s = overspillAssignments[i];
                    float[] overspillCentroid = effectiveCS.centroid(s);
                    float[] overspillParentCentroid = centroidClusters.getCentroid(s);
                    // write the overspill vector as well
                    result = quantizer.scalarQuantize(vector, scratch, quantized, effectiveEncoding.bits(), overspillCentroid);
                    if (overspillParentCentroid != null) {
                        float additionalCorrection = vectorSimilarityFunction == VectorSimilarityFunction.EUCLIDEAN
                            ? ESVectorUtil.squareDistance(vector, overspillParentCentroid)
                            : ESVectorUtil.dotProduct(scratch, overspillParentCentroid);
                        result = new OptimizedScalarQuantizer.QuantizationResult(
                            result.lowerInterval(),
                            result.upperInterval(),
                            additionalCorrection,
                            result.quantizedComponentSum()
                        );
                    }
                    effectiveEncoding.pack(quantized, binary);
                    writeQuantizedValue(quantizedVectorsTemp, binary, result);
                } else {
                    // write a zero vector for the overspill
                    Arrays.fill(binary, (byte) 0);
                    OptimizedScalarQuantizer.QuantizationResult zeroResult = new OptimizedScalarQuantizer.QuantizationResult(0f, 0f, 0f, 0);
                    writeQuantizedValue(quantizedVectorsTemp, binary, zeroResult);
                }
            }
        } catch (Throwable t) {
            if (quantizedVectorsTempName != null) {
                org.apache.lucene.util.IOUtils.deleteFilesIgnoringExceptions(mergeState.segmentInfo.dir, quantizedVectorsTempName);
            }
            throw t;
        }
        int[] centroidVectorCount = new int[effectiveCS.size()];
        for (int i = 0; i < assignments.length; i++) {
            centroidVectorCount[assignments[i]]++;
            if (overspillAssignments.length > i && overspillAssignments[i] != NO_SOAR_ASSIGNMENT) {
                centroidVectorCount[overspillAssignments[i]]++;
            }
        }

        int maxPostingListSize = 0;
        int[][] assignmentsByCluster = new int[effectiveCS.size()][];
        boolean[][] isOverspillByCluster = new boolean[effectiveCS.size()][];
        for (int c = 0; c < effectiveCS.size(); c++) {
            int size = centroidVectorCount[c];
            maxPostingListSize = Math.max(maxPostingListSize, size);
            assignmentsByCluster[c] = new int[size];
            isOverspillByCluster[c] = new boolean[size];
        }
        Arrays.fill(centroidVectorCount, 0);

        for (int i = 0; i < assignments.length; i++) {
            int c = assignments[i];
            assignmentsByCluster[c][centroidVectorCount[c]++] = i;
            // if soar assignments are present, add them to the cluster as well
            if (overspillAssignments.length > i) {
                int s = overspillAssignments[i];
                if (s != NO_SOAR_ASSIGNMENT) {
                    assignmentsByCluster[s][centroidVectorCount[s]] = i;
                    isOverspillByCluster[s][centroidVectorCount[s]++] = true;
                }
            }
        }
        // now we can read the quantized vectors from the temporary file
        try (IndexInput quantizedVectorsInput = mergeState.segmentInfo.dir.openInput(quantizedVectorsTempName, IOContext.DEFAULT)) {
            final PackedLongValues.Builder offsets = PackedLongValues.monotonicBuilder(PackedInts.COMPACT);
            final PackedLongValues.Builder lengths = PackedLongValues.monotonicBuilder(PackedInts.COMPACT);
            OffHeapQuantizedVectors offHeapQuantizedVectors = new OffHeapQuantizedVectors(
                quantizedVectorsInput,
                effectiveEncoding,
                fieldInfo.getVectorDimension()
            );
            DiskBBQBulkWriter bulkWriter = DiskBBQBulkWriter.fromBitSize(effectiveEncoding.bits(), BULK_SIZE, postingsOutput, true, true);
            // write the posting lists
            final int[] docIds = new int[maxPostingListSize];
            final int[] docDeltas = new int[maxPostingListSize];
            final int[] clusterOrds = new int[maxPostingListSize];
            DocIdsWriter idsWriter = new DocIdsWriter();
            for (int c = 0; c < effectiveCS.size(); c++) {
                float[] centroid = effectiveCS.centroid(c);
                int[] cluster = assignmentsByCluster[c];
                boolean[] isOverspill = isOverspillByCluster[c];
                long offset = postingsOutput.alignFilePointer(Float.BYTES) - fileOffset;
                offsets.add(offset);
                postingsOutput.writeInt(Float.floatToIntBits(ESVectorUtil.squareDistance(centroid, centroidClusters.getCentroid(c))));
                int size = cluster.length;
                postingsOutput.writeVInt(size);
                for (int j = 0; j < size; j++) {
                    docIds[j] = effectiveFvv.ordToDoc(cluster[j]);
                    clusterOrds[j] = j;
                }
                // sort cluster.buffer by docIds values, this way cluster ordinals are sorted by docIds
                new IntSorter(clusterOrds, i -> docIds[i]).sort(0, size);
                // encode doc deltas
                for (int j = 0; j < size; j++) {
                    docDeltas[j] = j == 0 ? docIds[clusterOrds[j]] : docIds[clusterOrds[j]] - docIds[clusterOrds[j - 1]];
                }
                byte encoding = idsWriter.calculateBlockEncoding(i -> docDeltas[i], size, BULK_SIZE);
                postingsOutput.writeByte(encoding);
                offHeapQuantizedVectors.reset(size, ord -> isOverspill[clusterOrds[ord]], ord -> cluster[clusterOrds[ord]]);
                // write vectors
                bulkWriter.writeVectors(offHeapQuantizedVectors, i -> {
                    // for vector i we write `bulk` size docs or the remaining docs
                    idsWriter.writeDocIds(d -> docDeltas[d + i], Math.min(BULK_SIZE, size - i), encoding, postingsOutput);
                });
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
        float min = Float.MAX_VALUE;
        float max = Float.MIN_VALUE;
        float mean = 0;
        float m2 = 0;
        // iteratively compute the variance & mean
        int count = 0;
        for (int[] cluster : clusters) {
            count += 1;
            if (cluster == null) {
                continue;
            }
            float delta = cluster.length - mean;
            mean += delta / count;
            m2 += delta * (cluster.length - mean);
            min = Math.min(min, cluster.length);
            max = Math.max(max, cluster.length);
        }
        float variance = m2 / (clusters.length - 1);
        logger.debug(
            "Centroid count: {} min: {} max: {} mean: {} stdDev: {} variance: {}",
            clusters.length,
            min,
            max,
            mean,
            Math.sqrt(variance),
            variance
        );
    }

    @Override
    public CentroidSupplier createCentroidSupplier(IndexInput centroidsInput, int numCentroids, FieldInfo fieldInfo, float[] globalCentroid)
        throws IOException {
        if (quantizationAuto) {
            savedGlobalCentroid = globalCentroid;
        }
        CentroidSupplier centroidSupplier = new OffHeapCentroidSupplier(
            centroidsInput,
            numCentroids,
            fieldInfo,
            KMeansResult.singleCluster(globalCentroid, numCentroids)
        );
        if (centroidSupplier.size() > centroidsPerParentCluster * centroidsPerParentCluster) {
            KMeansResult centroidClusters = buildSecondLevelClusters(fieldInfo, centroidSupplier, true);
            return new OffHeapCentroidSupplier(centroidsInput, numCentroids, fieldInfo, centroidClusters);
        }
        return centroidSupplier;
    }

    @Override
    public CentroidSupplier createCentroidSupplier(FieldInfo info, float[][] centroids, float[] globalCentroid) throws IOException {
        if (quantizationAuto) {
            savedGlobalCentroid = globalCentroid;
        }
        CentroidSupplier centroidSupplier = CentroidSupplier.fromArray(
            centroids,
            KMeansResult.singleCluster(globalCentroid, centroids.length),
            info.getVectorDimension()
        );
        if (centroidSupplier.size() > centroidsPerParentCluster * centroidsPerParentCluster) {
            KMeansResult centroidClusters = buildSecondLevelClusters(info, centroidSupplier, false);
            return CentroidSupplier.fromArray(centroids, centroidClusters, info.getVectorDimension());
        }
        return centroidSupplier;
    }

    @Override
    protected void doWriteMeta(
        IndexOutput metaOutput,
        FieldInfo field,
        int numCentroids,
        long preconditionerOffset,
        long preconditionerLength
    ) throws IOException {
        metaOutput.writeInt(ESNextOSQVectorsScorer.BULK_SIZE);
        ESNextDiskBBQVectorsFormat.QuantEncoding encodingToWrite;
        float calibratedOversample;
        if (quantizationAuto && calibrationResultForCurrentField != null) {
            encodingToWrite = calibrationResultForCurrentField.encoding();
            calibratedOversample = calibrationResultForCurrentField.oversample();
        } else {
            encodingToWrite = quantEncoding;
            calibratedOversample = AutoQuantizationSelector.NO_CALIBRATED_OVERSAMPLE;
        }
        boolean calibratedDoPrecondition = quantizationAuto
            && calibrationResultForCurrentField != null
            && calibrationResultForCurrentField.doPrecondition();
        metaOutput.writeInt(encodingToWrite.id());
        metaOutput.writeInt(Float.floatToIntBits(calibratedOversample));
        metaOutput.writeByte(calibratedDoPrecondition ? (byte) 1 : (byte) 0);
        metaOutput.writeLong(preconditionerLength);
        if (preconditionerLength > 0) {
            metaOutput.writeLong(preconditionerOffset);
        }
    }

    private void logAutoQuantizationCalibration(
        FieldInfo fieldInfo,
        FloatVectorValues floatVectorValues,
        MergeState mergeState,
        MergeCalibrationContext mergeCalibrationContext,
        long calibrationNanos,
        AutoQuantizationSelector.CalibrationResult result
    ) {
        int vectorCount = floatVectorValues.size();
        if (mergeState != null && mergeCalibrationContext != null) {
            logger.info(
                "diskbbq auto-quantization calibration field=[{}] segment=[{}] maxDoc={} vectorCount={} mergeKind={} inputSegments={} mergeMaxNumSegments={} took=[{}] result={}",
                fieldInfo.name,
                mergeState.segmentInfo.name,
                mergeState.segmentInfo.maxDoc(),
                vectorCount,
                mergeCalibrationContext.mergeKind(),
                mergeCalibrationContext.inputSegments(),
                mergeCalibrationContext.mergeMaxNumSegmentsForLog(),
                TimeValue.timeValueNanos(calibrationNanos),
                result
            );
        } else {
            logger.info(
                "diskbbq auto-quantization calibration field=[{}] vectorCount={} mergeKind=[flush] took=[{}] result={}",
                fieldInfo.name,
                vectorCount,
                TimeValue.timeValueNanos(calibrationNanos),
                result
            );
        }
    }

    private AutoQuantizationSelector.CalibrationResult resolveCalibrationResult(
        FieldInfo fieldInfo,
        FloatVectorValues floatVectorValues,
        CentroidSupplier centroidSupplier,
        int[] assignments,
        int[] overspillAssignments,
        MergeState mergeState
    ) {
        if (quantizationAuto) {
            return autoQuantizationSelector.select(
                fieldInfo,
                floatVectorValues,
                centroidSupplier,
                assignments,
                overspillAssignments != null ? overspillAssignments : new int[0],
                mergeState
            );
        }
        return new AutoQuantizationSelector.CalibrationResult(quantEncoding, AutoQuantizationSelector.NO_CALIBRATED_OVERSAMPLE, false);
    }

    /**
     * When quantizationAuto is true and calibration determined preconditioning helps,
     * applies the saved preconditioner to second-level cluster centroids (in-place) and the
     * global centroid (in-place). First-level centroids are handled via a wrapping supplier.
     */
    private void preconditionClusterCentroidsInPlace(CentroidSupplier centroidSupplier, int dim) throws IOException {
        if (savedPreconditioner == null) {
            return;
        }
        float[] out = new float[dim];
        KMeansResult clusters = centroidSupplier.secondLevelClusters();
        if (clusters != null && clusters != KMeansResult.EMPTY && clusters.centroids() != null) {
            for (float[] parentCentroid : clusters.centroids()) {
                if (parentCentroid != null) {
                    savedPreconditioner.applyTransform(parentCentroid, out);
                    System.arraycopy(out, 0, parentCentroid, 0, dim);
                }
            }
        }
        if (savedGlobalCentroid != null) {
            savedPreconditioner.applyTransform(savedGlobalCentroid, out);
            System.arraycopy(out, 0, savedGlobalCentroid, 0, dim);
        }
    }

    private FloatVectorValues wrapFvvWithPreconditioner(FloatVectorValues vectors) {
        return new FloatVectorValues() {
            final float[] preconditionedVectorValue = new float[vectors.dimension()];
            int cachedOrd = -1;

            @Override
            public float[] vectorValue(int ord) throws IOException {
                assert ord != -1;
                if (ord != cachedOrd) {
                    float[] vectorValue = vectors.vectorValue(ord);
                    savedPreconditioner.applyTransform(vectorValue, this.preconditionedVectorValue);
                    cachedOrd = ord;
                }
                return this.preconditionedVectorValue;
            }

            @Override
            public FloatVectorValues copy() throws IOException {
                return vectors.copy();
            }

            @Override
            public int dimension() {
                return vectors.dimension();
            }

            @Override
            public int size() {
                return vectors.size();
            }

            @Override
            public DocIndexIterator iterator() {
                return vectors.iterator();
            }
        };
    }

    /**
     * Wraps a CentroidSupplier to apply the saved preconditioner on-the-fly to first-level centroids.
     * Second-level cluster centroids must be preconditioned in-place before using this wrapper.
     */
    private CentroidSupplier wrapCentroidSupplierWithPreconditioner(CentroidSupplier delegate, int dim) {
        return new CentroidSupplier() {
            private final float[] scratch = new float[dim];

            @Override
            public int size() {
                return delegate.size();
            }

            @Override
            public float[] centroid(int centroidOrdinal) throws IOException {
                float[] raw = delegate.centroid(centroidOrdinal);
                savedPreconditioner.applyTransform(raw, scratch);
                return scratch;
            }

            @Override
            public KMeansResult secondLevelClusters() throws IOException {
                return delegate.secondLevelClusters();
            }

            @Override
            public KMeansFloatVectorValues asKmeansFloatVectorValues() throws IOException {
                return delegate.asKmeansFloatVectorValues();
            }
        };
    }

    private boolean shouldApplyDeferredPreconditioning() {
        return quantizationAuto
            && calibrationResultForCurrentField != null
            && calibrationResultForCurrentField.doPrecondition()
            && savedPreconditioner != null;
    }

    @Override
    public void writeCentroids(
        FieldInfo fieldInfo,
        CentroidSupplier centroidSupplier,
        int[] centroidAssignments,
        float[] globalCentroid,
        CentroidOffsetAndLength centroidOffsetAndLength,
        IndexOutput centroidOutput
    ) throws IOException {
        doWriteCentroids(fieldInfo, centroidSupplier, centroidAssignments, globalCentroid, centroidOffsetAndLength, centroidOutput);
    }

    @Override
    public void writeCentroids(
        FieldInfo fieldInfo,
        CentroidSupplier centroidSupplier,
        int[] centroidAssignments,
        float[] globalCentroid,
        CentroidOffsetAndLength centroidOffsetAndLength,
        IndexOutput centroidOutput,
        MergeState mergeState
    ) throws IOException {
        doWriteCentroids(fieldInfo, centroidSupplier, centroidAssignments, globalCentroid, centroidOffsetAndLength, centroidOutput);
    }

    private record CentroidGroups(float[][] centroids, int[][] vectors, int maxVectorsPerCentroidLength) {}

    private void doWriteCentroids(
        FieldInfo fieldInfo,
        CentroidSupplier centroidSupplier,
        int[] centroidAssignments,
        float[] globalCentroid,
        CentroidOffsetAndLength centroidOffsetAndLength,
        IndexOutput centroidOutput
    ) throws IOException {
        CentroidSupplier effectiveCS = centroidSupplier;
        if (shouldApplyDeferredPreconditioning()) {
            effectiveCS = wrapCentroidSupplierWithPreconditioner(centroidSupplier, fieldInfo.getVectorDimension());
        }
        if (effectiveCS.secondLevelClusters().centroidsSupplier().size() > 1) {
            final CentroidGroups centroidGroups = buildCentroidGroups(effectiveCS.secondLevelClusters());
            {
                final int[] centroidOrdinalMap = new int[effectiveCS.size()];
                int idx = 0;
                for (int[] centroidVectors : centroidGroups.vectors()) {
                    for (int assignment : centroidVectors) {
                        centroidOrdinalMap[assignment] = idx++;
                    }
                }
                assert idx == effectiveCS.size() : "Expected [" + effectiveCS.size() + "], got [" + idx + "]";
                writeCentroidLookup(centroidOutput, centroidAssignments, i -> centroidOrdinalMap[i], effectiveCS.size());
            }
            writeCentroidsWithParents(fieldInfo, effectiveCS, globalCentroid, centroidOffsetAndLength, centroidOutput, centroidGroups);
        } else {
            writeCentroidLookup(centroidOutput, centroidAssignments, IntUnaryOperator.identity(), effectiveCS.size());
            writeCentroidsWithoutParents(fieldInfo, effectiveCS, globalCentroid, centroidOffsetAndLength, centroidOutput);
        }
    }

    private void writeCentroidLookup(IndexOutput out, int[] centroidAssignments, IntUnaryOperator OrdinalMap, int numberCentroids)
        throws IOException {
        final int bitsRequired = DirectWriter.bitsRequired(numberCentroids);
        final long bytesRequired = ESNextDiskBBQVectorsReader.directWriterSizeOnDisk(centroidAssignments.length, bitsRequired);
        final ByteBuffersDataOutput memory = new ByteBuffersDataOutput(bytesRequired);
        final DirectWriter writer = DirectWriter.getInstance(memory, centroidAssignments.length, bitsRequired);
        for (int centroidAssignment : centroidAssignments) {
            writer.add(OrdinalMap.applyAsInt(centroidAssignment));
        }
        writer.finish();
        out.copyBytes(memory.toDataInput(), memory.size());
    }

    private void writeCentroidsWithParents(
        FieldInfo fieldInfo,
        CentroidSupplier centroidSupplier,
        float[] globalCentroid,
        CentroidOffsetAndLength centroidOffsetAndLength,
        IndexOutput centroidOutput,
        CentroidGroups centroidGroups
    ) throws IOException {
        DiskBBQBulkWriter bulkWriter = DiskBBQBulkWriter.fromBitSize(7, BULK_SIZE, centroidOutput, true, true);
        final OptimizedScalarQuantizer osq = new OptimizedScalarQuantizer(fieldInfo.getVectorSimilarityFunction());
        centroidOutput.writeVInt(centroidGroups.centroids().length);
        centroidOutput.writeVInt(centroidGroups.maxVectorsPerCentroidLength());
        // let's also write the raw parent centroids
        final ByteBuffer buffer = ByteBuffer.allocate(fieldInfo.getVectorDimension() * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < centroidGroups.centroids().length; i++) {
            float[] centroid = centroidGroups.centroids()[i];
            buffer.asFloatBuffer().put(centroid);
            centroidOutput.writeBytes(buffer.array(), buffer.array().length);
        }
        QuantizedCentroids parentQuantizeCentroid = new QuantizedCentroids(
            CentroidSupplier.fromArray(centroidGroups.centroids, KMeansResult.EMPTY, fieldInfo.getVectorDimension()),
            fieldInfo.getVectorDimension(),
            osq,
            globalCentroid
        );
        bulkWriter.writeVectors(parentQuantizeCentroid, null);
        int offset = 0;
        for (int[] centroidVectors : centroidGroups.vectors()) {
            centroidOutput.writeInt(offset);
            centroidOutput.writeInt(centroidVectors.length);
            offset += centroidVectors.length;
        }

        QuantizedCentroids childrenQuantizeCentroid = new QuantizedCentroids(
            centroidSupplier,
            fieldInfo.getVectorDimension(),
            osq,
            globalCentroid
        );
        for (int[] centroidVectors : centroidGroups.vectors()) {
            childrenQuantizeCentroid.reset(idx -> centroidVectors[idx], centroidVectors.length);
            bulkWriter.writeVectors(childrenQuantizeCentroid, null);
        }
        // write the centroid offsets at the end of the file
        int parentOrd = 0;
        for (int[] centroidVectors : centroidGroups.vectors()) {
            for (int assignment : centroidVectors) {
                centroidOutput.writeLong(centroidOffsetAndLength.offsets().get(assignment));
                centroidOutput.writeLong(centroidOffsetAndLength.lengths().get(assignment));
                centroidOutput.writeInt(parentOrd);
            }
            parentOrd++;
        }
    }

    private void writeCentroidsWithoutParents(
        FieldInfo fieldInfo,
        CentroidSupplier centroidSupplier,
        float[] globalCentroid,
        CentroidOffsetAndLength centroidOffsetAndLength,
        IndexOutput centroidOutput
    ) throws IOException {
        centroidOutput.writeVInt(0);
        DiskBBQBulkWriter bulkWriter = DiskBBQBulkWriter.fromBitSize(7, BULK_SIZE, centroidOutput, true, true);
        final OptimizedScalarQuantizer osq = new OptimizedScalarQuantizer(fieldInfo.getVectorSimilarityFunction());
        QuantizedCentroids quantizedCentroids = new QuantizedCentroids(
            centroidSupplier,
            fieldInfo.getVectorDimension(),
            osq,
            globalCentroid
        );
        bulkWriter.writeVectors(quantizedCentroids, null);
        // write the centroid offsets at the end of the file
        for (int i = 0; i < centroidSupplier.size(); i++) {
            centroidOutput.writeLong(centroidOffsetAndLength.offsets().get(i));
            centroidOutput.writeLong(centroidOffsetAndLength.lengths().get(i));
        }
    }

    private KMeansResult buildSecondLevelClusters(FieldInfo fieldInfo, CentroidSupplier centroidSupplier, boolean isMerge)
        throws IOException {
        final KMeansFloatVectorValues floatVectorValues = centroidSupplier.asKmeansFloatVectorValues();
        // we use the HierarchicalKMeans to partition the space of all vectors across merging segments
        // this are small numbers so we run it wih all the centroids.
        HierarchicalKMeans hierarchicalKMeans;
        if (isMerge && mergeExec != null) {
            hierarchicalKMeans = HierarchicalKMeans.ofConcurrent(
                fieldInfo.getVectorDimension(),
                mergeExec,
                numMergeWorkers,
                HierarchicalKMeans.MAX_ITERATIONS_DEFAULT,
                HierarchicalKMeans.SAMPLES_PER_CLUSTER_DEFAULT,
                HierarchicalKMeans.MAXK,
                -1 // disable SOAR assignments
            );
        } else {
            hierarchicalKMeans = HierarchicalKMeans.ofSerial(
                fieldInfo.getVectorDimension(),
                HierarchicalKMeans.MAX_ITERATIONS_DEFAULT,
                HierarchicalKMeans.SAMPLES_PER_CLUSTER_DEFAULT,
                HierarchicalKMeans.MAXK,
                -1 // disable SOAR assignments
            );
        }
        return hierarchicalKMeans.cluster(floatVectorValues, centroidsPerParentCluster);
    }

    private CentroidGroups buildCentroidGroups(KMeansResult kMeansResult) {
        final int[] centroidVectorCount = new int[kMeansResult.centroids().length];
        for (int i = 0; i < kMeansResult.assignments().length; i++) {
            centroidVectorCount[kMeansResult.assignments()[i]]++;
        }
        final int[][] vectorsPerCentroid = new int[kMeansResult.centroids().length][];
        int maxVectorsPerCentroidLength = 0;
        for (int i = 0; i < kMeansResult.centroidsSupplier().size(); i++) {
            vectorsPerCentroid[i] = new int[centroidVectorCount[i]];
            maxVectorsPerCentroidLength = Math.max(maxVectorsPerCentroidLength, centroidVectorCount[i]);
        }
        Arrays.fill(centroidVectorCount, 0);
        for (int i = 0; i < kMeansResult.assignments().length; i++) {
            final int c = kMeansResult.assignments()[i];
            vectorsPerCentroid[c][centroidVectorCount[c]++] = i;
        }
        return new CentroidGroups(kMeansResult.centroids(), vectorsPerCentroid, maxVectorsPerCentroidLength);
    }

    @Override
    public CentroidAssignments calculateCentroids(FieldInfo fieldInfo, KMeansFloatVectorValues floatVectorValues, MergeState mergeState)
        throws IOException {
        // TODO: consider hinting / bootstrapping hierarchical kmeans with the prior segments centroids
        // TODO: for flush we are doing this over the vectors and here centroids which seems duplicative
        // preliminary tests suggest recall is good using only centroids but need to do further evaluation
        HierarchicalKMeans hierarchicalKMeans;
        if (mergeExec != null) {
            hierarchicalKMeans = HierarchicalKMeans.ofConcurrent(floatVectorValues.dimension(), mergeExec, numMergeWorkers);
        } else {
            hierarchicalKMeans = HierarchicalKMeans.ofSerial(floatVectorValues.dimension());
        }
        return calculateCentroids(hierarchicalKMeans, floatVectorValues, fieldInfo);
    }

    /**
     * Calculate the centroids for the given field.
     * We use the {@link HierarchicalKMeans} algorithm to partition the space of all vectors across merging segments
     *
     * @param fieldInfo merging field info
     * @param floatVectorValues the float vector values to merge
     * @return the vector assignments, soar assignments, and if asked the centroids themselves that were computed
     * @throws IOException if an I/O error occurs
     */
    @Override
    public CentroidAssignments calculateCentroids(FieldInfo fieldInfo, KMeansFloatVectorValues floatVectorValues) throws IOException {
        HierarchicalKMeans hierarchicalKMeans = HierarchicalKMeans.ofSerial(floatVectorValues.dimension());
        return calculateCentroids(hierarchicalKMeans, floatVectorValues, fieldInfo);
    }

    private CentroidAssignments calculateCentroids(
        HierarchicalKMeans hierarchicalKMeans,
        KMeansFloatVectorValues floatVectorValues,
        FieldInfo fieldInfo
    ) throws IOException {
        KMeansResult kMeansResult = hierarchicalKMeans.cluster(floatVectorValues, vectorPerCluster);
        float[][] centroids = kMeansResult.centroids();
        if (logger.isDebugEnabled()) {
            logger.debug("final centroid count: {}", centroids.length);
        }
        int[] assignments = kMeansResult.assignments();
        int[] soarAssignments = kMeansResult.soarAssignments();
        return new CentroidAssignments(fieldInfo.getVectorDimension(), centroids, assignments, soarAssignments);
    }

    static void writeQuantizedValue(IndexOutput indexOutput, byte[] binaryValue, OptimizedScalarQuantizer.QuantizationResult corrections)
        throws IOException {
        indexOutput.writeBytes(binaryValue, binaryValue.length);
        indexOutput.writeInt(Float.floatToIntBits(corrections.lowerInterval()));
        indexOutput.writeInt(Float.floatToIntBits(corrections.upperInterval()));
        indexOutput.writeInt(Float.floatToIntBits(corrections.additionalCorrection()));
        indexOutput.writeInt(corrections.quantizedComponentSum());
    }

    static class OffHeapCentroidSupplier implements CentroidSupplier {
        private final IndexInput centroidsInput;
        private final int numCentroids;
        private final int dimension;
        private final float[] scratch;
        private final KMeansResult clusters;
        private int currOrd = -1;

        OffHeapCentroidSupplier(IndexInput centroidsInput, int numCentroids, FieldInfo info, KMeansResult clusters) {
            this.centroidsInput = centroidsInput;
            this.numCentroids = numCentroids;
            this.dimension = info.getVectorDimension();
            this.scratch = new float[dimension];
            this.clusters = clusters;
        }

        @Override
        public int size() {
            return numCentroids;
        }

        @Override
        public float[] centroid(int centroidOrdinal) throws IOException {
            if (centroidOrdinal == currOrd) {
                return scratch;
            }
            centroidsInput.seek((long) centroidOrdinal * dimension * Float.BYTES);
            centroidsInput.readFloats(scratch, 0, dimension);
            this.currOrd = centroidOrdinal;
            return scratch;
        }

        @Override
        public KMeansResult secondLevelClusters() {
            return clusters;
        }

        @Override
        public KMeansFloatVectorValues asKmeansFloatVectorValues() throws IOException {
            return KMeansFloatVectorValues.build(centroidsInput, null, numCentroids, dimension);
        }
    }

    static class QuantizedCentroids implements QuantizedVectorValues {
        private final CentroidSupplier supplier;
        private final OptimizedScalarQuantizer quantizer;
        private final byte[] quantizedVector;
        private final int[] quantizedVectorScratch;
        private final float[] floatVectorScratch;
        private OptimizedScalarQuantizer.QuantizationResult corrections;
        private final float[] centroid;
        private int currOrd = -1;
        private IntToIntFunction ordTransformer = i -> i;
        int size;

        QuantizedCentroids(CentroidSupplier supplier, int dimension, OptimizedScalarQuantizer quantizer, float[] centroid) {
            this.supplier = supplier;
            this.quantizer = quantizer;
            this.quantizedVector = new byte[dimension];
            this.floatVectorScratch = new float[dimension];
            this.quantizedVectorScratch = new int[dimension];
            this.centroid = centroid;
            size = supplier.size();
        }

        @Override
        public int count() {
            return size;
        }

        void reset(IntToIntFunction ordTransformer, int size) {
            this.ordTransformer = ordTransformer;
            this.currOrd = -1;
            this.size = size;
            this.corrections = null;
        }

        @Override
        public byte[] next() throws IOException {
            if (currOrd >= count() - 1) {
                throw new IllegalStateException("No more vectors to read, current ord: " + currOrd + ", count: " + count());
            }
            currOrd++;
            float[] vector = supplier.centroid(ordTransformer.apply(currOrd));
            corrections = quantizer.scalarQuantize(vector, floatVectorScratch, quantizedVectorScratch, (byte) 7, centroid);
            for (int i = 0; i < quantizedVectorScratch.length; i++) {
                quantizedVector[i] = (byte) quantizedVectorScratch[i];
            }
            return quantizedVector;
        }

        @Override
        public OptimizedScalarQuantizer.QuantizationResult getCorrections() {
            return corrections;
        }
    }

    static class OnHeapQuantizedVectors implements QuantizedVectorValues {
        private final FloatVectorValues vectorValues;
        private final OptimizedScalarQuantizer quantizer;
        private final byte[] quantizedVector;
        private final int[] quantizedVectorScratch;
        private final float[] floatVectorScratch;
        private final ESNextDiskBBQVectorsFormat.QuantEncoding encoding;
        private OptimizedScalarQuantizer.QuantizationResult corrections;
        private final VectorSimilarityFunction similarityFunction;
        private float[] currentCentroid, currentParentCentroid;
        private IntToIntFunction ordTransformer = null;
        private int currOrd = -1;
        private int count;

        OnHeapQuantizedVectors(
            FloatVectorValues vectorValues,
            VectorSimilarityFunction similarityFunction,
            ESNextDiskBBQVectorsFormat.QuantEncoding encoding,
            int dimension,
            OptimizedScalarQuantizer quantizer
        ) {
            this.vectorValues = vectorValues;
            this.similarityFunction = similarityFunction;
            this.encoding = encoding;
            this.quantizer = quantizer;
            this.quantizedVector = new byte[encoding.getDocPackedLength(dimension)];
            this.floatVectorScratch = new float[dimension];
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
            float[] vector = vectorValues.vectorValue(ord);
            corrections = quantizer.scalarQuantize(vector, floatVectorScratch, quantizedVectorScratch, encoding.bits(), currentCentroid);
            // note, with a parent centroid, our correction needs to take it into account
            if (currentParentCentroid != null) {
                float additionalCorrection = similarityFunction == VectorSimilarityFunction.EUCLIDEAN
                    ? ESVectorUtil.squareDistance(vector, currentParentCentroid)
                    : ESVectorUtil.dotProduct(floatVectorScratch, currentParentCentroid);
                corrections = new OptimizedScalarQuantizer.QuantizationResult(
                    corrections.lowerInterval(),
                    corrections.upperInterval(),
                    additionalCorrection,
                    corrections.quantizedComponentSum()
                );
            }
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
        private final byte[] binaryScratch;
        private final float[] corrections = new float[3];

        private final int vectorByteSize;
        private int bitSum;
        private int currOrd = -1;
        private int count;
        private IntToBooleanFunction isOverspill = null;
        private IntToIntFunction ordTransformer = null;

        OffHeapQuantizedVectors(IndexInput quantizedVectorsInput, ESNextDiskBBQVectorsFormat.QuantEncoding encoding, int dimension) {
            this.quantizedVectorsInput = quantizedVectorsInput;
            this.binaryScratch = new byte[encoding.getDocPackedLength(dimension)];
            this.vectorByteSize = (binaryScratch.length + 3 * Float.BYTES + Integer.BYTES);
        }

        private void reset(int count, IntToBooleanFunction isOverspill, IntToIntFunction ordTransformer) {
            this.count = count;
            this.isOverspill = isOverspill;
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
            boolean isOverspill = this.isOverspill.apply(currOrd);
            return getVector(ord, isOverspill);
        }

        @Override
        public OptimizedScalarQuantizer.QuantizationResult getCorrections() {
            if (currOrd == -1) {
                throw new IllegalStateException("No vector read yet, call readQuantizedVector first");
            }
            return new OptimizedScalarQuantizer.QuantizationResult(corrections[0], corrections[1], corrections[2], bitSum);
        }

        byte[] getVector(int ord, boolean isOverspill) throws IOException {
            readQuantizedVector(ord, isOverspill);
            return binaryScratch;
        }

        public void readQuantizedVector(int ord, boolean isOverspill) throws IOException {
            long offset = (long) ord * (vectorByteSize * 2L) + (isOverspill ? vectorByteSize : 0);
            quantizedVectorsInput.seek(offset);
            quantizedVectorsInput.readBytes(binaryScratch, 0, binaryScratch.length);
            quantizedVectorsInput.readFloats(corrections, 0, 3);
            bitSum = quantizedVectorsInput.readInt();
        }
    }
}
