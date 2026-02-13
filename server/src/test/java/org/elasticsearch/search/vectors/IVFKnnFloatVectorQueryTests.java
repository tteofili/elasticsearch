/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */
package org.elasticsearch.search.vectors;

import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.analysis.MockAnalyzer;
import org.apache.lucene.tests.index.RandomIndexWriter;
import org.apache.lucene.tests.util.TestUtil;
import org.apache.lucene.util.VectorUtil;
import org.elasticsearch.index.codec.vectors.diskbbq.ES920DiskBBQVectorsFormat;

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

    /**
     * Tests that global best-first multi-segment IVF search returns correct results when
     * the index has multiple segments (all with IVF format).
     */
    public void testGlobalBestFirstMultiSegment() throws IOException {
        Directory dir = newDirectory();
        try {
            IndexWriterConfig iwc = new IndexWriterConfig(new MockAnalyzer(random()));
            iwc.setMergePolicy(NoMergePolicy.INSTANCE);
            iwc.setCodec(TestUtil.alwaysKnnVectorsFormat(new ES920DiskBBQVectorsFormat(128, 4)));

            try (RandomIndexWriter writer = new RandomIndexWriter(random(), dir, iwc)) {
                // First segment
                for (int i = 0; i < 30; i++) {
                    Document doc = new Document();
                    float[] v = randomVector(4);
                    doc.add(getKnnVectorField("field", v));
                    doc.add(new StringField("id", "s0_" + i, Field.Store.YES));
                    writer.addDocument(doc);
                }
                writer.commit();

                // Second segment
                for (int i = 0; i < 30; i++) {
                    Document doc = new Document();
                    float[] v = randomVector(4);
                    doc.add(getKnnVectorField("field", v));
                    doc.add(new StringField("id", "s1_" + i, Field.Store.YES));
                    writer.addDocument(doc);
                }
                writer.commit();
            }

            try (IndexReader reader = DirectoryReader.open(dir)) {
                assertEquals("Index should have 2 segments", 2, reader.leaves().size());
                IndexSearcher searcher = newSearcher(reader);
                float[] queryVector = randomVector(4);
                IVFKnnFloatVectorQuery query = new IVFKnnFloatVectorQuery(
                    "field",
                    queryVector,
                    10,
                    20,
                    null,
                    0.2f,
                    false
                );
                TopDocs results = searcher.search(query, 10);
                assertNotNull(results);
                assertTrue("Should find results", results.scoreDocs.length > 0);
                assertTrue("Should find at most k results", results.scoreDocs.length <= 10);
                float lastScore = Float.MAX_VALUE;
                for (ScoreDoc sd : results.scoreDocs) {
                    assertTrue("Results should be in descending score order", sd.score <= lastScore);
                    lastScore = sd.score;
                }
            }
        } finally {
            dir.close();
        }
    }

    /**
     * Single-segment index: search should complete successfully (no global path; baseline per-segment).
     */
    public void testSingleSegmentUnchanged() throws IOException {
        try (
            Directory indexStore = getIndexStore("field", new float[] { 0, 1 }, new float[] { 1, 0 }, new float[] { 0.5f, 0.5f });
            IndexReader reader = DirectoryReader.open(indexStore)
        ) {
            IndexSearcher searcher = newSearcher(reader);
            IVFKnnFloatVectorQuery query = new IVFKnnFloatVectorQuery(
                "field",
                new float[] { 0.1f, 0.9f },
                3,
                5,
                null,
                0.1f,
                false
            );
            TopDocs results = searcher.search(query, 5);
            assertNotNull(results);
            assertEquals(3, results.scoreDocs.length);
        }
    }
}
