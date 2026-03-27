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

/**
 * Contract tests for {@link AutoQuantizationSelector}. Precondition selection for
 * {@link CalibratingAutoQuantizationSelector} is covered in
 * {@link CalibratingAutoQuantizationSelectorTests}.
 */
public class AutoQuantizationSelectorTests extends ESTestCase {

    public void testDefaultSelectorReturnsValidEncoding() throws IOException {
        int dimension = 4;
        int numVectors = 10;
        float[][] vectors = new float[numVectors][dimension];
        float[] globalCentroid = new float[dimension];
        for (int i = 0; i < numVectors; i++) {
            for (int d = 0; d < dimension; d++) {
                vectors[i][d] = randomFloat();
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

        FieldInfo fieldInfo = getFieldInfoFromIndex(dimension);

        AutoQuantizationSelector.CalibrationResult result = NoOpAutomaticQuantizationSelector.INSTANCE.select(
            fieldInfo,
            fvv,
            supplier,
            assignments.assignments(),
            assignments.overspillAssignments(),
            null
        );
        assertNotNull(result);
        assertSame(ESNextDiskBBQVectorsFormat.QuantEncoding.ONE_BIT_4BIT_QUERY, result.encoding());
        assertEquals(AutoQuantizationSelector.NO_CALIBRATED_OVERSAMPLE, result.oversample(), 0.0f);
    }

    public void testNoOpSelectorAlwaysReportsPreconditionFalse() throws IOException {
        int dimension = 4;
        int numVectors = 10;
        float[][] vectors = new float[numVectors][dimension];
        float[] globalCentroid = new float[dimension];
        for (int i = 0; i < numVectors; i++) {
            for (int d = 0; d < dimension; d++) {
                vectors[i][d] = randomFloat();
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
        FieldInfo fieldInfo = getFieldInfoFromIndex(dimension);

        AutoQuantizationSelector.CalibrationResult result = NoOpAutomaticQuantizationSelector.INSTANCE.select(
            fieldInfo,
            fvv,
            supplier,
            assignments.assignments(),
            assignments.overspillAssignments(),
            null
        );
        assertFalse(result.doPrecondition());
    }

    private static FieldInfo getFieldInfoFromIndex(int dimension) throws IOException {
        try (Directory dir = newDirectory()) {
            try (IndexWriter w = new IndexWriter(dir, newIndexWriterConfig())) {
                Document doc = new Document();
                doc.add(new KnnFloatVectorField("f", new float[dimension], VectorSimilarityFunction.DOT_PRODUCT));
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
