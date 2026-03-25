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
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.store.Directory;
import org.elasticsearch.index.codec.vectors.cluster.KMeansFloatVectorValues;
import org.elasticsearch.index.codec.vectors.cluster.KMeansResult;
import org.elasticsearch.index.codec.vectors.diskbbq.CentroidAssignments;
import org.elasticsearch.index.codec.vectors.diskbbq.CentroidSupplier;
import org.elasticsearch.test.ESTestCase;

import java.io.IOException;
import java.lang.reflect.Field;
import java.util.List;
import java.util.Map;
import java.util.Random;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class CalibratingAutoQuantizationSelectorTests extends ESTestCase {

    public void testParityDefaultsMatchCxxCalibration() {
        assertEquals(0.97, CalibratingAutoQuantizationSelector.DEFAULT_TARGET_RECALL, 0.0d);
        assertEquals(100, CalibratingAutoQuantizationSelector.DEFAULT_K);
    }

    public void testSelectReturnsValidEncoding() throws IOException {
        int dimension = 16;
        int numVectors = 200;
        Random rng = new Random(42);
        float[][] vectors = new float[numVectors][dimension];
        float[] globalCentroid = new float[dimension];
        for (int i = 0; i < numVectors; i++) {
            for (int d = 0; d < dimension; d++) {
                vectors[i][d] = rng.nextFloat() - 0.5f;
                globalCentroid[d] += vectors[i][d];
            }
        }
        for (int d = 0; d < dimension; d++) {
            globalCentroid[d] /= numVectors;
        }
        CentroidAssignments assignments = new CentroidAssignments(
            dimension,
            new float[][] { globalCentroid },
            new int[numVectors],
            new int[0]
        );
        CentroidSupplier supplier = CentroidSupplier.fromArray(
            assignments.centroids(),
            KMeansResult.singleCluster(assignments.globalCentroid(), assignments.numCentroids()),
            dimension
        );
        List<float[]> vectorsList = List.of(vectors);
        KMeansFloatVectorValues fvv = KMeansFloatVectorValues.build(vectorsList, null, dimension);

        FieldInfo fieldInfo = getFieldInfoFromIndex(dimension, VectorSimilarityFunction.EUCLIDEAN);

        CalibratingAutoQuantizationSelector selector = new CalibratingAutoQuantizationSelector(
            ESNextDiskBBQVectorsFormat.DEFAULT_VECTORS_PER_CLUSTER,
            false
        );

        AutoQuantizationSelector.CalibrationResult result = selector.select(
            fieldInfo,
            fvv,
            supplier,
            assignments.assignments(),
            assignments.overspillAssignments(),
            null
        );
        assertNotNull(result);
        assertNotNull(result.encoding());
    }

    public void testSelectWithDotProduct() throws IOException {
        int dimension = 16;
        int numVectors = 200;
        Random rng = new Random(99);
        float[][] vectors = new float[numVectors][dimension];
        float[] globalCentroid = new float[dimension];
        for (int i = 0; i < numVectors; i++) {
            for (int d = 0; d < dimension; d++) {
                vectors[i][d] = rng.nextFloat() - 0.5f;
                globalCentroid[d] += vectors[i][d];
            }
        }
        for (int d = 0; d < dimension; d++) {
            globalCentroid[d] /= numVectors;
        }
        CentroidAssignments assignments = new CentroidAssignments(
            dimension,
            new float[][] { globalCentroid },
            new int[numVectors],
            new int[0]
        );
        CentroidSupplier supplier = CentroidSupplier.fromArray(
            assignments.centroids(),
            KMeansResult.singleCluster(assignments.globalCentroid(), assignments.numCentroids()),
            dimension
        );
        List<float[]> vectorsList = List.of(vectors);
        KMeansFloatVectorValues fvv = KMeansFloatVectorValues.build(vectorsList, null, dimension);

        FieldInfo fieldInfo = getFieldInfoFromIndex(dimension, VectorSimilarityFunction.DOT_PRODUCT);

        CalibratingAutoQuantizationSelector selector = new CalibratingAutoQuantizationSelector(
            ESNextDiskBBQVectorsFormat.DEFAULT_VECTORS_PER_CLUSTER,
            false
        );

        AutoQuantizationSelector.CalibrationResult result = selector.select(
            fieldInfo,
            fvv,
            supplier,
            assignments.assignments(),
            assignments.overspillAssignments(),
            null
        );
        assertNotNull(result);
        assertNotNull(result.encoding());
    }

    public void testSelectWithCosine() throws IOException {
        int dimension = 16;
        int numVectors = 200;
        Random rng = new Random(77);
        float[][] vectors = new float[numVectors][dimension];
        float[] globalCentroid = new float[dimension];
        for (int i = 0; i < numVectors; i++) {
            double norm = 0;
            for (int d = 0; d < dimension; d++) {
                vectors[i][d] = rng.nextFloat() - 0.5f;
                norm += vectors[i][d] * vectors[i][d];
            }
            norm = Math.sqrt(norm);
            for (int d = 0; d < dimension; d++) {
                vectors[i][d] /= (float) norm;
                globalCentroid[d] += vectors[i][d];
            }
        }
        for (int d = 0; d < dimension; d++) {
            globalCentroid[d] /= numVectors;
        }
        CentroidAssignments assignments = new CentroidAssignments(
            dimension,
            new float[][] { globalCentroid },
            new int[numVectors],
            new int[0]
        );
        CentroidSupplier supplier = CentroidSupplier.fromArray(
            assignments.centroids(),
            KMeansResult.singleCluster(assignments.globalCentroid(), assignments.numCentroids()),
            dimension
        );
        List<float[]> vectorsList = List.of(vectors);
        KMeansFloatVectorValues fvv = KMeansFloatVectorValues.build(vectorsList, null, dimension);

        FieldInfo fieldInfo = getFieldInfoFromIndex(dimension, VectorSimilarityFunction.COSINE);

        CalibratingAutoQuantizationSelector selector = new CalibratingAutoQuantizationSelector(
            ESNextDiskBBQVectorsFormat.DEFAULT_VECTORS_PER_CLUSTER,
            false
        );

        AutoQuantizationSelector.CalibrationResult result = selector.select(
            fieldInfo,
            fvv,
            supplier,
            assignments.assignments(),
            assignments.overspillAssignments(),
            null
        );
        assertNotNull(result);
        assertNotNull(result.encoding());
    }

    public void testSelectWithSingleVector() throws IOException {
        int dimension = 8;
        float[][] vectors = { { 1, 2, 3, 4, 5, 6, 7, 8 } };
        CentroidAssignments assignments = new CentroidAssignments(dimension, new float[][] { vectors[0] }, new int[] { 0 }, new int[0]);
        CentroidSupplier supplier = CentroidSupplier.fromArray(
            assignments.centroids(),
            KMeansResult.singleCluster(assignments.globalCentroid(), assignments.numCentroids()),
            dimension
        );
        KMeansFloatVectorValues fvv = KMeansFloatVectorValues.build(List.of(vectors), null, dimension);
        FieldInfo fieldInfo = getFieldInfoFromIndex(dimension, VectorSimilarityFunction.EUCLIDEAN);

        CalibratingAutoQuantizationSelector selector = new CalibratingAutoQuantizationSelector(384, false);

        AutoQuantizationSelector.CalibrationResult result = selector.select(
            fieldInfo,
            fvv,
            supplier,
            assignments.assignments(),
            assignments.overspillAssignments(),
            null
        );
        assertSame(ESNextDiskBBQVectorsFormat.QuantEncoding.FOUR_BIT_SYMMETRIC, result.encoding());
        assertEquals(AutoQuantizationSelector.DEFAULT_CALIBRATED_OVERSAMPLE, result.oversample(), 0.0f);
    }

    public void testCalibrationResultOversampleSemantics() {
        AutoQuantizationSelector.CalibrationResult r1 = new AutoQuantizationSelector.CalibrationResult(
            ESNextDiskBBQVectorsFormat.QuantEncoding.ONE_BIT_4BIT_QUERY,
            1.5f,
            false
        );
        assertEquals(ESNextDiskBBQVectorsFormat.QuantEncoding.ONE_BIT_4BIT_QUERY, r1.encoding());
        assertEquals(1.5f, r1.oversample(), 0.0f);
        assertFalse(r1.doPrecondition());

        AutoQuantizationSelector.CalibrationResult r2 = new AutoQuantizationSelector.CalibrationResult(
            ESNextDiskBBQVectorsFormat.QuantEncoding.SEVEN_BIT_SYMMETRIC,
            AutoQuantizationSelector.NO_CALIBRATED_OVERSAMPLE,
            false
        );
        assertEquals(AutoQuantizationSelector.NO_CALIBRATED_OVERSAMPLE, r2.oversample(), 0.0f);

        AutoQuantizationSelector.CalibrationResult r3 = new AutoQuantizationSelector.CalibrationResult(
            ESNextDiskBBQVectorsFormat.QuantEncoding.FOUR_BIT_SYMMETRIC,
            3.0f,
            false
        );
        assertEquals(3.0f, r3.oversample(), 0.0f);
    }

    public void testSelectFromMergeState_unanimousReuse() throws IOException {
        int dimension = 16;
        FieldInfo fieldInfo = getFieldInfoFromIndex(dimension, VectorSimilarityFunction.EUCLIDEAN);
        float[] centroid = new float[dimension];

        StubCalibrationReader reader1 = new StubCalibrationReader(
            ESNextDiskBBQVectorsFormat.QuantEncoding.TWO_BIT_4BIT_QUERY,
            2.0f,
            false,
            centroid
        );
        StubCalibrationReader reader2 = new StubCalibrationReader(
            ESNextDiskBBQVectorsFormat.QuantEncoding.TWO_BIT_4BIT_QUERY,
            2.0f,
            false,
            centroid
        );

        MergeState mergeState = mockMergeState(new KnnVectorsReader[] { reader1, reader2 }, new int[] { 5000, 5000 });
        FloatVectorValues fvv = mockFloatVectorValues(10000);
        CentroidSupplier supplier = CentroidSupplier.fromArray(
            new float[][] { centroid },
            KMeansResult.singleCluster(centroid, 1),
            dimension
        );

        CalibratingAutoQuantizationSelector selector = new CalibratingAutoQuantizationSelector(384, false);
        AutoQuantizationSelector.CalibrationResult result = selector.selectFromMergeState(fieldInfo, fvv, supplier, mergeState);

        assertNotNull(result);
        assertEquals(ESNextDiskBBQVectorsFormat.QuantEncoding.TWO_BIT_4BIT_QUERY, result.encoding());
        assertEquals(2.0f, result.oversample(), 0.0f);
        assertFalse(result.doPrecondition());
    }

    public void testSelectFromMergeState_majorityVote() throws IOException {
        int dimension = 16;
        FieldInfo fieldInfo = getFieldInfoFromIndex(dimension, VectorSimilarityFunction.EUCLIDEAN);
        float[] centroid = new float[dimension];

        StubCalibrationReader reader1 = new StubCalibrationReader(
            ESNextDiskBBQVectorsFormat.QuantEncoding.FOUR_BIT_SYMMETRIC,
            1.5f,
            false,
            centroid
        );
        StubCalibrationReader reader2 = new StubCalibrationReader(
            ESNextDiskBBQVectorsFormat.QuantEncoding.TWO_BIT_4BIT_QUERY,
            2.0f,
            false,
            centroid
        );

        MergeState mergeState = mockMergeState(new KnnVectorsReader[] { reader1, reader2 }, new int[] { 9000, 1000 });
        FloatVectorValues fvv = mockFloatVectorValues(10000);
        CentroidSupplier supplier = CentroidSupplier.fromArray(
            new float[][] { centroid },
            KMeansResult.singleCluster(centroid, 1),
            dimension
        );

        CalibratingAutoQuantizationSelector selector = new CalibratingAutoQuantizationSelector(384, false);
        AutoQuantizationSelector.CalibrationResult result = selector.selectFromMergeState(fieldInfo, fvv, supplier, mergeState);

        assertNotNull(result);
        assertEquals(ESNextDiskBBQVectorsFormat.QuantEncoding.FOUR_BIT_SYMMETRIC, result.encoding());
    }

    public void testSelectFromMergeState_noCalibrationReadersReturnsNull() throws IOException {
        int dimension = 16;
        FieldInfo fieldInfo = getFieldInfoFromIndex(dimension, VectorSimilarityFunction.EUCLIDEAN);

        KnnVectorsReader plainReader = new StubNonCalibrationReader();
        MergeState mergeState = mockMergeState(new KnnVectorsReader[] { plainReader }, new int[] { 5000 });
        FloatVectorValues fvv = mockFloatVectorValues(5000);

        CalibratingAutoQuantizationSelector selector = new CalibratingAutoQuantizationSelector(384, false);
        AutoQuantizationSelector.CalibrationResult result = selector.selectFromMergeState(fieldInfo, fvv, null, mergeState);

        assertNull(result);
    }

    public void testSelectFromMergeState_growthRatioTriggersRecalibration() throws IOException {
        int dimension = 16;
        FieldInfo fieldInfo = getFieldInfoFromIndex(dimension, VectorSimilarityFunction.EUCLIDEAN);
        float[] centroid = new float[dimension];

        StubCalibrationReader reader = new StubCalibrationReader(
            ESNextDiskBBQVectorsFormat.QuantEncoding.FOUR_BIT_SYMMETRIC,
            1.5f,
            false,
            centroid
        );

        MergeState mergeState = mockMergeState(new KnnVectorsReader[] { reader }, new int[] { 1000 });
        FloatVectorValues fvv = mockFloatVectorValues(5000);
        CentroidSupplier supplier = CentroidSupplier.fromArray(
            new float[][] { centroid },
            KMeansResult.singleCluster(centroid, 1),
            dimension
        );

        CalibratingAutoQuantizationSelector selector = new CalibratingAutoQuantizationSelector(384, false);
        AutoQuantizationSelector.CalibrationResult result = selector.selectFromMergeState(fieldInfo, fvv, supplier, mergeState);

        assertNull(result);
    }

    public void testSelectFromMergeState_centroidDriftTriggersRecalibration() throws IOException {
        int dimension = 16;
        FieldInfo fieldInfo = getFieldInfoFromIndex(dimension, VectorSimilarityFunction.EUCLIDEAN);
        float[] centroid = new float[dimension];
        float[] driftedCentroid = new float[dimension];
        for (int d = 0; d < dimension; d++) {
            driftedCentroid[d] = 10.0f;
        }

        StubCalibrationReader reader = new StubCalibrationReader(
            ESNextDiskBBQVectorsFormat.QuantEncoding.FOUR_BIT_SYMMETRIC,
            1.5f,
            false,
            driftedCentroid
        );

        MergeState mergeState = mockMergeState(new KnnVectorsReader[] { reader }, new int[] { 5000 });
        FloatVectorValues fvv = mockFloatVectorValues(5000);
        CentroidSupplier supplier = CentroidSupplier.fromArray(
            new float[][] { centroid },
            KMeansResult.singleCluster(centroid, 1),
            dimension
        );

        CalibratingAutoQuantizationSelector selector = new CalibratingAutoQuantizationSelector(384, false);
        AutoQuantizationSelector.CalibrationResult result = selector.selectFromMergeState(fieldInfo, fvv, supplier, mergeState);

        assertNull(result);
    }

    public void testSelectFromMergeState_encodingDisagreementTriggersRecalibration() throws IOException {
        int dimension = 16;
        FieldInfo fieldInfo = getFieldInfoFromIndex(dimension, VectorSimilarityFunction.EUCLIDEAN);
        float[] centroid = new float[dimension];

        StubCalibrationReader reader1 = new StubCalibrationReader(
            ESNextDiskBBQVectorsFormat.QuantEncoding.ONE_BIT_4BIT_QUERY,
            1.5f,
            false,
            centroid
        );
        StubCalibrationReader reader2 = new StubCalibrationReader(
            ESNextDiskBBQVectorsFormat.QuantEncoding.FOUR_BIT_SYMMETRIC,
            2.0f,
            false,
            centroid
        );

        MergeState mergeState = mockMergeState(new KnnVectorsReader[] { reader1, reader2 }, new int[] { 5000, 5000 });
        FloatVectorValues fvv = mockFloatVectorValues(10000);
        CentroidSupplier supplier = CentroidSupplier.fromArray(
            new float[][] { centroid },
            KMeansResult.singleCluster(centroid, 1),
            dimension
        );

        CalibratingAutoQuantizationSelector selector = new CalibratingAutoQuantizationSelector(384, false);
        AutoQuantizationSelector.CalibrationResult result = selector.selectFromMergeState(fieldInfo, fvv, supplier, mergeState);

        assertNull(result);
    }

    public void testSelectFromMergeState_preconditionMajorityVote() throws IOException {
        int dimension = 16;
        FieldInfo fieldInfo = getFieldInfoFromIndex(dimension, VectorSimilarityFunction.EUCLIDEAN);
        float[] centroid = new float[dimension];

        StubCalibrationReader reader1 = new StubCalibrationReader(
            ESNextDiskBBQVectorsFormat.QuantEncoding.FOUR_BIT_SYMMETRIC,
            1.5f,
            true,
            centroid
        );
        StubCalibrationReader reader2 = new StubCalibrationReader(
            ESNextDiskBBQVectorsFormat.QuantEncoding.FOUR_BIT_SYMMETRIC,
            1.5f,
            true,
            centroid
        );
        StubCalibrationReader reader3 = new StubCalibrationReader(
            ESNextDiskBBQVectorsFormat.QuantEncoding.FOUR_BIT_SYMMETRIC,
            1.5f,
            false,
            centroid
        );

        MergeState mergeState = mockMergeState(new KnnVectorsReader[] { reader1, reader2, reader3 }, new int[] { 4000, 4000, 2000 });
        FloatVectorValues fvv = mockFloatVectorValues(10000);
        CentroidSupplier supplier = CentroidSupplier.fromArray(
            new float[][] { centroid },
            KMeansResult.singleCluster(centroid, 1),
            dimension
        );

        CalibratingAutoQuantizationSelector selector = new CalibratingAutoQuantizationSelector(384, false);
        AutoQuantizationSelector.CalibrationResult result = selector.selectFromMergeState(fieldInfo, fvv, supplier, mergeState);

        assertNotNull(result);
        assertTrue(result.doPrecondition());
    }

    private static MergeState mockMergeState(KnnVectorsReader[] readers, int[] maxDocs) {
        MergeState mergeState = mock(MergeState.class);
        try {
            Field readersField = MergeState.class.getField("knnVectorsReaders");
            readersField.setAccessible(true);
            readersField.set(mergeState, readers);
            Field maxDocsField = MergeState.class.getField("maxDocs");
            maxDocsField.setAccessible(true);
            maxDocsField.set(mergeState, maxDocs);
        } catch (ReflectiveOperationException e) {
            throw new AssertionError("Failed to set MergeState fields", e);
        }
        return mergeState;
    }

    private static FloatVectorValues mockFloatVectorValues(int size) {
        FloatVectorValues fvv = mock(FloatVectorValues.class);
        when(fvv.size()).thenReturn(size);
        return fvv;
    }

    /**
     * Stub that is both a KnnVectorsReader and CalibrationAwareReader for testing the merge path.
     */
    private static class StubCalibrationReader extends KnnVectorsReader implements CalibrationAwareReader {
        private final ESNextDiskBBQVectorsFormat.QuantEncoding encoding;
        private final float oversample;
        private final boolean precondition;
        private final float[] globalCentroid;

        StubCalibrationReader(
            ESNextDiskBBQVectorsFormat.QuantEncoding encoding,
            float oversample,
            boolean precondition,
            float[] globalCentroid
        ) {
            this.encoding = encoding;
            this.oversample = oversample;
            this.precondition = precondition;
            this.globalCentroid = globalCentroid;
        }

        @Override
        public float getOversampleFactor(FieldInfo fieldInfo) {
            return oversample;
        }

        @Override
        public boolean shouldPrecondition(FieldInfo fieldInfo) {
            return precondition;
        }

        @Override
        public ESNextDiskBBQVectorsFormat.QuantEncoding getQuantEncoding(FieldInfo fieldInfo) {
            return encoding;
        }

        @Override
        public float[] getGlobalCentroid(FieldInfo fieldInfo) {
            return globalCentroid;
        }

        @Override
        public void checkIntegrity() {}

        @Override
        public FloatVectorValues getFloatVectorValues(String field) {
            return null;
        }

        @Override
        public ByteVectorValues getByteVectorValues(String field) {
            return null;
        }

        @Override
        public void search(String field, float[] target, KnnCollector knnCollector, AcceptDocs acceptDocs) {}

        @Override
        public void search(String field, byte[] target, KnnCollector knnCollector, AcceptDocs acceptDocs) {}

        @Override
        public Map<String, Long> getOffHeapByteSize(FieldInfo fieldInfo) {
            return Map.of();
        }

        @Override
        public void close() {}
    }

    /**
     * Plain KnnVectorsReader without CalibrationAwareReader interface.
     */
    private static class StubNonCalibrationReader extends KnnVectorsReader {
        @Override
        public void checkIntegrity() {}

        @Override
        public FloatVectorValues getFloatVectorValues(String field) {
            return null;
        }

        @Override
        public ByteVectorValues getByteVectorValues(String field) {
            return null;
        }

        @Override
        public void search(String field, float[] target, KnnCollector knnCollector, AcceptDocs acceptDocs) {}

        @Override
        public void search(String field, byte[] target, KnnCollector knnCollector, AcceptDocs acceptDocs) {}

        @Override
        public Map<String, Long> getOffHeapByteSize(FieldInfo fieldInfo) {
            return Map.of();
        }

        @Override
        public void close() {}
    }

    private static FieldInfo getFieldInfoFromIndex(int dimension, VectorSimilarityFunction similarity) throws IOException {
        try (Directory dir = newDirectory()) {
            try (IndexWriter w = new IndexWriter(dir, newIndexWriterConfig())) {
                Document doc = new Document();
                float[] vec = new float[dimension];
                java.util.Arrays.fill(vec, 1.0f);
                doc.add(new KnnFloatVectorField("f", vec, similarity));
                w.addDocument(doc);
                w.commit();
            }
            try (IndexReader reader = DirectoryReader.open(dir)) {
                LeafReader leafReader = getOnlyLeafReader(reader);
                FieldInfo info = leafReader.getFieldInfos().fieldInfo("f");
                assert info != null;
                return info;
            }
        }
    }
}
