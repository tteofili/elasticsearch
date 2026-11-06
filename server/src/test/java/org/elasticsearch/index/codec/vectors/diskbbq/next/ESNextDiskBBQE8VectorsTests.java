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
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.util.TestUtil;
import org.elasticsearch.index.codec.vectors.diskbbq.IvfFlushConfigSource;
import org.elasticsearch.index.codec.vectors.diskbbq.IvfMergeConfigResolver;
import org.elasticsearch.index.codec.vectors.diskbbq.Preconditioner;
import org.elasticsearch.index.codec.vectors.diskbbq.QuantEncoding;
import org.elasticsearch.index.codec.vectors.diskbbq.VectorPreconditioner;
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
 * End-to-end tests of the ESNext DiskBBQ format with the experimental {@link QuantEncoding#E8_ROOTS} encoding: write,
 * merge and search, with and without preconditioning, for every supported similarity, including a dimension that is not
 * a multiple of eight (zero-padded last block).
 */
public class ESNextDiskBBQE8VectorsTests extends ESTestCase {

    private static final int DIMS = 68;

    public void testIndexAndSearch() throws IOException {
        for (boolean precondition : new boolean[] { false, true }) {
            for (VectorSimilarityFunction sim : new VectorSimilarityFunction[] {
                VectorSimilarityFunction.DOT_PRODUCT,
                VectorSimilarityFunction.EUCLIDEAN,
                VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT }) {
                double recall = indexAndMeasureRecall(e8Format(precondition), sim, 300, 10);
                // random data at about one bit per dimension: the bar only guards against a broken scorer, whose results
                // would be no better than chance (about 10 / 300)
                assertThat("precondition=" + precondition + " sim=" + sim, recall, greaterThan(0.15));
            }
        }
    }

    public void testByteVectorsAreRejected() {
        expectThrows(
            IllegalArgumentException.class,
            () -> new ESNextDiskBBQVectorsFormat(
                QuantEncoding.E8_ROOTS,
                ESNextDiskBBQVectorsFormat.MIN_VECTORS_PER_CLUSTER,
                ESNextDiskBBQVectorsFormat.DEFAULT_CENTROIDS_PER_PARENT_CLUSTER,
                DenseVectorFieldMapper.ElementType.BYTE,
                false,
                null,
                1,
                false,
                ESNextDiskBBQVectorsFormat.DEFAULT_PRECONDITIONING_BLOCK_DIMENSION,
                -1,
                null,
                IvfFlushConfigSource.empty(),
                IvfMergeConfigResolver.useCodecDefault(),
                false
            )
        );
    }

    private static ESNextDiskBBQVectorsFormat e8Format(boolean precondition) {
        return new ESNextDiskBBQVectorsFormat(
            QuantEncoding.E8_ROOTS,
            ESNextDiskBBQVectorsFormat.MIN_VECTORS_PER_CLUSTER,
            ESNextDiskBBQVectorsFormat.DEFAULT_CENTROIDS_PER_PARENT_CLUSTER,
            DenseVectorFieldMapper.ElementType.FLOAT,
            false,
            null,
            1,
            precondition,
            ESNextDiskBBQVectorsFormat.DEFAULT_PRECONDITIONING_BLOCK_DIMENSION,
            -1,
            null,
            IvfFlushConfigSource.empty(),
            IvfMergeConfigResolver.useCodecDefault(),
            false
        );
    }

    private double indexAndMeasureRecall(ESNextDiskBBQVectorsFormat format, VectorSimilarityFunction sim, int numDocs, int numQueries)
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
                        // the IVF query applies the segment's preconditioner; searching the leaf directly, do the same
                        float[] searchQuery = preconditioned(leaf, query);
                        int k = 10;
                        TopDocs topDocs = leaf.searchNearestVectors(
                            "f",
                            searchQuery,
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

    private static float[] preconditioned(LeafReader leaf, float[] query) {
        if (leaf instanceof SegmentReader sr
            && sr.getVectorReader() instanceof PerFieldKnnVectorsFormat.FieldsReader fr
            && fr.getFieldReader("f") instanceof VectorPreconditioner vp) {
            Preconditioner preconditioner = vp.getPreconditioner(leaf.getFieldInfos().fieldInfo("f"));
            if (preconditioner != null) {
                float[] out = new float[query.length];
                preconditioner.applyTransform(query, out);
                return out;
            }
        }
        return query;
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
