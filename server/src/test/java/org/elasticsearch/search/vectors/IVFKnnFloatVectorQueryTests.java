/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */
package org.elasticsearch.search.vectors;

import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.util.TestUtil;
import org.apache.lucene.util.VectorUtil;
import org.elasticsearch.index.codec.vectors.diskbbq.CentroidSupplier;
import org.elasticsearch.index.codec.vectors.diskbbq.next.AutoQuantizationSelector;
import org.elasticsearch.index.codec.vectors.diskbbq.next.ESNextDiskBBQVectorsFormat;
import org.elasticsearch.index.mapper.vectors.DenseVectorFieldMapper;

import java.io.IOException;

import static com.carrotsearch.randomizedtesting.RandomizedTest.randomFloat;

public class IVFKnnFloatVectorQueryTests extends AbstractIVFKnnVectorQueryTestCase {

    @Override
    IVFKnnFloatVectorQuery getKnnVectorQuery(String field, float[] query, int k, Query queryFilter, float visitRatio) {
        return new IVFKnnFloatVectorQuery(field, query, k, k, queryFilter, visitRatio, random().nextBoolean());
    }

    @Override
    float[] randomVector(int dim) {
        float[] vector = new float[dim];
        for (int i = 0; i < dim; i++) {
            vector[i] = randomFloat();
        }
        VectorUtil.l2normalize(vector);
        return vector;
    }

    @Override
    Field getKnnVectorField(String name, float[] vector, VectorSimilarityFunction similarityFunction) {
        return new KnnFloatVectorField(name, vector, similarityFunction);
    }

    @Override
    Field getKnnVectorField(String name, float[] vector) {
        return new KnnFloatVectorField(name, vector);
    }

    public void testToString() throws IOException {
        try (
            Directory indexStore = getIndexStore("field", new float[] { 0, 1 }, new float[] { 1, 2 }, new float[] { 0, 0 });
            IndexReader reader = DirectoryReader.open(indexStore)
        ) {
            AbstractIVFKnnVectorQuery query = getKnnVectorQuery("field", new float[] { 0.0f, 1.0f }, 10);
            assertEquals("IVFKnnFloatVectorQuery:field[0.0,...][10]", query.toString("ignored"));

            assertDocScoreQueryToString(query.rewrite(newSearcher(reader)));

            // test with filter
            Query filter = new TermQuery(new Term("id", "text"));
            query = getKnnVectorQuery("field", new float[] { 0.0f, 1.0f }, 10, filter);
            assertEquals("IVFKnnFloatVectorQuery:field[0.0,...][10][id:text]", query.toString("ignored"));
        }
    }

    public void testRewriteWithCalibrationSentinelDoesNotAutoRescore() throws IOException {
        int dimensions = 16;
        int numDocs = 800;
        AutoQuantizationSelector fixedSelector = new AutoQuantizationSelector() {
            @Override
            public CalibrationResult select(
                FieldInfo fieldInfo,
                FloatVectorValues floatVectorValues,
                CentroidSupplier centroidSupplier,
                int[] assignments,
                int[] overspillAssignments,
                MergeState mergeState
            ) {
                return new CalibrationResult(
                    ESNextDiskBBQVectorsFormat.QuantEncoding.SEVEN_BIT_SYMMETRIC,
                    AutoQuantizationSelector.NO_CALIBRATED_OVERSAMPLE,
                    false
                );
            }
        };
        KnnVectorsFormat testFormat = new ESNextDiskBBQVectorsFormat(
            true,
            fixedSelector,
            ESNextDiskBBQVectorsFormat.MIN_VECTORS_PER_CLUSTER,
            ESNextDiskBBQVectorsFormat.DEFAULT_CENTROIDS_PER_PARENT_CLUSTER,
            DenseVectorFieldMapper.ElementType.FLOAT,
            false,
            null,
            1,
            false,
            ESNextDiskBBQVectorsFormat.DEFAULT_PRECONDITIONING_BLOCK_DIMENSION,
            0
        );
        IndexWriterConfig iwc = newIndexWriterConfig().setCodec(TestUtil.alwaysKnnVectorsFormat(testFormat));
        try (Directory dir = newDirectoryForTest(); IndexWriter w = new IndexWriter(dir, iwc)) {
            for (int i = 0; i < numDocs; i++) {
                Document doc = new Document();
                doc.add(new KnnFloatVectorField("f", randomVector(dimensions), VectorSimilarityFunction.DOT_PRODUCT));
                w.addDocument(doc);
            }
            w.commit();
            w.forceMerge(1);
            try (IndexReader reader = DirectoryReader.open(w)) {
                IndexSearcher searcher = newSearcher(reader);
                Query rewritten = new IVFKnnFloatVectorQuery(
                    "f",
                    randomVector(dimensions),
                    10,
                    100,
                    null,
                    0.1f,
                    false,
                    true,
                    VectorSimilarityFunction.DOT_PRODUCT
                ).rewrite(searcher);
                assertFalse(rewritten instanceof RescoreKnnVectorQuery);
            }
        }
    }

    public void testRewriteWithCalibrationOversampleTriggersAutoRescore() throws IOException {
        int dimensions = 16;
        int numDocs = 800;
        float calibratedOversample = 2.0f;

        AutoQuantizationSelector fixedSelector = new AutoQuantizationSelector() {
            @Override
            public CalibrationResult select(
                FieldInfo fieldInfo,
                FloatVectorValues floatVectorValues,
                CentroidSupplier centroidSupplier,
                int[] assignments,
                int[] overspillAssignments,
                MergeState mergeState
            ) {
                return new CalibrationResult(ESNextDiskBBQVectorsFormat.QuantEncoding.SEVEN_BIT_SYMMETRIC, calibratedOversample, false);
            }
        };
        KnnVectorsFormat testFormat = new ESNextDiskBBQVectorsFormat(
            true,
            fixedSelector,
            ESNextDiskBBQVectorsFormat.MIN_VECTORS_PER_CLUSTER,
            ESNextDiskBBQVectorsFormat.DEFAULT_CENTROIDS_PER_PARENT_CLUSTER,
            DenseVectorFieldMapper.ElementType.FLOAT,
            false,
            null,
            1,
            false,
            ESNextDiskBBQVectorsFormat.DEFAULT_PRECONDITIONING_BLOCK_DIMENSION,
            0
        );
        IndexWriterConfig iwc = newIndexWriterConfig().setCodec(TestUtil.alwaysKnnVectorsFormat(testFormat));
        try (Directory dir = newDirectoryForTest(); IndexWriter w = new IndexWriter(dir, iwc)) {
            for (int i = 0; i < numDocs; i++) {
                Document doc = new Document();
                doc.add(new KnnFloatVectorField("f", randomVector(dimensions), VectorSimilarityFunction.DOT_PRODUCT));
                w.addDocument(doc);
            }
            w.commit();
            w.forceMerge(1);
            try (IndexReader reader = DirectoryReader.open(w)) {
                IndexSearcher searcher = newSearcher(reader);
                Query rewritten = new IVFKnnFloatVectorQuery(
                    "f",
                    randomVector(dimensions),
                    10,
                    100,
                    null,
                    0.1f,
                    false,
                    true,
                    VectorSimilarityFunction.DOT_PRODUCT
                ).rewrite(searcher);
                assertTrue(rewritten instanceof KnnScoreDocQuery);
            }
        }
    }
}
