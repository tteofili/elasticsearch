/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq.next;

import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.Directory;
import org.elasticsearch.index.codec.vectors.cluster.KMeansFloatVectorValues;
import org.elasticsearch.index.codec.vectors.cluster.KMeansResult;
import org.elasticsearch.index.codec.vectors.diskbbq.CentroidAssignments;
import org.elasticsearch.index.codec.vectors.diskbbq.CentroidSupplier;
import org.elasticsearch.test.ESTestCase;

import java.io.IOException;
import java.util.List;
import java.util.Random;

public class CalibratingAutoQuantizationSelectorTests extends ESTestCase {

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
