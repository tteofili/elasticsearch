/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq.next;

import org.apache.lucene.codecs.Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.util.TestUtil;
import org.elasticsearch.index.codec.vectors.diskbbq.IvfFlushConfigSource;
import org.elasticsearch.index.codec.vectors.diskbbq.IvfMergeConfigResolver;
import org.elasticsearch.index.codec.vectors.diskbbq.IvfSegmentConfig;
import org.elasticsearch.index.mapper.vectors.DenseVectorFieldMapper;
import org.elasticsearch.test.ESTestCase;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

import static org.hamcrest.Matchers.arrayWithSize;
import static org.hamcrest.Matchers.greaterThan;
import static org.hamcrest.Matchers.greaterThanOrEqualTo;

/**
 * End-to-end tests of the ASH format with the experimental E8 codebook: write, merge and search, for one and two stages,
 * with and without dimensionality reduction, for every supported similarity.
 */
public class ESNextDiskASHE8VectorsTests extends ESTestCase {

    private static final int DIMS = 64;

    public void testIndexAndSearch() throws IOException {
        for (int bits : new int[] { 1, 2 }) {
            for (float fraction : new float[] { 0.5f, 1.0f }) {
                for (VectorSimilarityFunction sim : new VectorSimilarityFunction[] {
                    VectorSimilarityFunction.DOT_PRODUCT,
                    VectorSimilarityFunction.EUCLIDEAN,
                    VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT }) {
                    double recall = indexAndMeasureRecall(e8Format(bits, fraction), sim, 300, 10);
                    // random data at one or two bits per coded dimension: the bar only guards against a broken scorer,
                    // whose results would be no better than chance (about 10 / 300)
                    assertThat("bits=" + bits + " fraction=" + fraction + " sim=" + sim, recall, greaterThan(0.15));
                }
            }
        }
    }

    private static ESNextDiskASHVectorsFormat e8Format(int bits, float fraction) {
        var ashConfig = IvfSegmentConfig.AshConfig.of(
            bits,
            IvfSegmentConfig.AshConfig.DEFAULT_QUERY_BITS_PER_DIM,
            fraction,
            IvfSegmentConfig.AshConfig.DEFAULT_MAX_PROJECTED_DIMS
        )
            .withCodebook(IvfSegmentConfig.AshConfig.CODEBOOK_E8);
        return new ESNextDiskASHVectorsFormat(
            ashConfig,
            ESNextDiskASHVectorsFormat.MIN_VECTORS_PER_CLUSTER,
            ESNextDiskASHVectorsFormat.DEFAULT_CENTROIDS_PER_PARENT_CLUSTER,
            DenseVectorFieldMapper.ElementType.FLOAT,
            false,
            null,
            1,
            null,
            -1,
            null,
            IvfFlushConfigSource.empty(),
            IvfMergeConfigResolver.useCodecDefault(),
            false
        );
    }

    private double indexAndMeasureRecall(ESNextDiskASHVectorsFormat format, VectorSimilarityFunction sim, int numDocs, int numQueries)
        throws IOException {
        Codec codec = TestUtil.alwaysKnnVectorsFormat(format);
        float[][] docs = new float[numDocs][];
        for (int i = 0; i < numDocs; i++) {
            docs[i] = randomUnitVector();
        }
        try (Directory dir = newDirectory()) {
            IndexWriterConfig iwc = newIndexWriterConfig();
            iwc.setCodec(codec);
            try (IndexWriter w = new IndexWriter(dir, iwc)) {
                for (int i = 0; i < numDocs; i++) {
                    Document doc = new Document();
                    doc.add(new KnnFloatVectorField("f", docs[i], sim));
                    w.addDocument(doc);
                }
                w.forceMerge(1);
                try (IndexReader reader = DirectoryReader.open(w)) {
                    assertEquals(1, reader.leaves().size());
                    LeafReader leaf = reader.leaves().get(0).reader();
                    double recallSum = 0;
                    for (int qi = 0; qi < numQueries; qi++) {
                        float[] query = randomUnitVector();
                        int k = 10;
                        TopDocs topDocs = leaf.searchNearestVectors(
                            "f",
                            query,
                            k,
                            AcceptDocs.fromLiveDocs(leaf.getLiveDocs(), leaf.maxDoc()),
                            Integer.MAX_VALUE
                        );
                        assertThat(topDocs.scoreDocs, arrayWithSize(k));
                        for (int i = 0; i < topDocs.scoreDocs.length - 1; i++) {
                            assertThat(topDocs.scoreDocs[i].score, greaterThanOrEqualTo(topDocs.scoreDocs[i + 1].score));
                        }
                        // force-merged into one segment with no deletions, so doc ids are insertion order
                        Set<Integer> truth = exactTopK(docs, query, sim, k);
                        int hits = 0;
                        for (ScoreDoc sd : topDocs.scoreDocs) {
                            if (truth.contains(sd.doc)) {
                                hits++;
                            }
                        }
                        recallSum += (double) hits / k;
                    }
                    return recallSum / numQueries;
                }
            }
        }
    }

    private static Set<Integer> exactTopK(float[][] docs, float[] query, VectorSimilarityFunction sim, int k) {
        Integer[] order = new Integer[docs.length];
        float[] scores = new float[docs.length];
        for (int i = 0; i < docs.length; i++) {
            order[i] = i;
            scores[i] = sim.compare(query, docs[i]);
        }
        Arrays.sort(order, (a, b) -> Float.compare(scores[b], scores[a]));
        return new HashSet<>(Arrays.asList(order).subList(0, k));
    }

    private static float[] randomUnitVector() {
        float[] v = new float[DIMS];
        double n = 0;
        for (int i = 0; i < DIMS; i++) {
            v[i] = (float) random().nextGaussian();
            n += v[i] * v[i];
        }
        float inv = (float) (1 / Math.sqrt(n));
        for (int i = 0; i < DIMS; i++) {
            v[i] *= inv;
        }
        return v;
    }
}
