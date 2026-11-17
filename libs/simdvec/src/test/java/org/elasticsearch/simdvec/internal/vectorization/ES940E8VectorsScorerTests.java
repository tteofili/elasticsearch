/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.simdvec.internal.vectorization;

import com.carrotsearch.randomizedtesting.annotations.ParametersFactory;

import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.store.NIOFSDirectory;
import org.elasticsearch.simdvec.BaseVectorizationTests;
import org.elasticsearch.simdvec.BulkMask;
import org.elasticsearch.simdvec.E8Roots;
import org.elasticsearch.simdvec.ES940OSQVectorsScorer;
import org.elasticsearch.simdvec.ScoringQuery;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static org.elasticsearch.simdvec.ES940OSQVectorsScorer.BitEncoding.STRIPED;

/** Checks that the scalar, Panama and native scorers of E8 root codes agree, in bulk and masked, with tails. */
public class ES940E8VectorsScorerTests extends BaseVectorizationTests {

    private static final byte INDEX_BITS = 1;
    private static final byte QUERY_BITS = 8;

    private final boolean mmap;
    private final VectorSimilarityFunction similarityFunction;

    public ES940E8VectorsScorerTests(boolean mmap, VectorSimilarityFunction similarityFunction) {
        this.mmap = mmap;
        this.similarityFunction = similarityFunction;
    }

    public void testScoreBulk() throws IOException {
        doTestScoreBulk(ES940OSQVectorsScorer.BULK_SIZE);
    }

    public void testScoreBulkTail() throws IOException {
        doTestScoreBulk(randomIntBetween(1, ES940OSQVectorsScorer.BULK_SIZE - 1));
    }

    private void doTestScoreBulk(int count) throws IOException {
        // up to 4096 dimensions exercises the flush of the int16 accumulators (more than 256 blocks)
        final int dims = randomBoolean() ? randomIntBetween(1, 1024) : randomIntBetween(2049, 4096);
        final int blocks = E8Roots.blocks(dims);
        final int numBulks = randomIntBetween(1, 5);
        final int padding = randomIntBetween(0, 64);
        final float[] y = new float[dims];
        for (int i = 0; i < dims; i++) {
            y[i] = (float) random().nextGaussian();
        }
        final byte[] table = new byte[blocks * E8Roots.TABLE_SIZE];
        final float tableScale = E8Roots.buildQueryTable(y, table);
        final ScoringQuery query = new ScoringQuery(table);

        try (Directory dir = mmap ? new MMapDirectory(createTempDir()) : new NIOFSDirectory(createTempDir())) {
            List<float[]> expectedRaw = new ArrayList<>();
            try (IndexOutput out = dir.createOutput("e8.bin", IOContext.DEFAULT)) {
                out.writeBytes(new byte[padding], padding);
                final float[] x = new float[dims];
                final int[] codes = new int[blocks];
                for (int bulk = 0; bulk < numBulks; bulk++) {
                    byte[] transposed = new byte[blocks * count];
                    float[] raw = new float[count];
                    for (int d = 0; d < count; d++) {
                        for (int i = 0; i < dims; i++) {
                            x[i] = (float) random().nextGaussian();
                        }
                        E8Roots.encode(x, codes);
                        for (int b = 0; b < blocks; b++) {
                            transposed[b * count + d] = (byte) codes[b];
                        }
                    }
                    E8Roots.scoreBulk(transposed, 0, table, blocks, count, raw);
                    expectedRaw.add(raw);
                    out.writeBytes(transposed, transposed.length);
                    for (int d = 0; d < count; d++) {
                        out.writeInt(Float.floatToIntBits(0f)); // lower interval
                    }
                    for (int d = 0; d < count; d++) {
                        out.writeInt(Float.floatToIntBits(0.5f + random().nextFloat())); // ratio scale
                    }
                    for (int d = 0; d < count; d++) {
                        out.writeInt(0); // component sum, unused
                    }
                    for (int d = 0; d < count; d++) {
                        out.writeInt(Float.floatToIntBits((float) random().nextGaussian())); // additional correction
                    }
                }
                CodecUtil.writeFooter(out);
            }

            final float queryUpper = 255f * tableScale;
            final float queryAdditional = (float) random().nextGaussian();
            try (IndexInput in = dir.openInput("e8.bin", IOContext.DEFAULT)) {
                final long perBulk = (long) count * (blocks + 16);
                final IndexInput[] inputs = new IndexInput[] { in.clone(), in.clone(), in.clone() };
                final ES940OSQVectorsScorer[] scorers = new ES940OSQVectorsScorer[3];
                scorers[0] = defaultProvider().getVectorScorerFactory()
                    .newES940OSQVectorsScorer(inputs[0], QUERY_BITS, INDEX_BITS, dims, blocks, ES940OSQVectorsScorer.BULK_SIZE, STRIPED);
                scorers[1] = panamaProvider().getVectorScorerFactory()
                    .newES940OSQVectorsScorer(inputs[1], QUERY_BITS, INDEX_BITS, dims, blocks, ES940OSQVectorsScorer.BULK_SIZE, STRIPED);
                scorers[2] = nativeProvider().getVectorScorerFactory()
                    .newES940OSQVectorsScorer(inputs[2], QUERY_BITS, INDEX_BITS, dims, blocks, ES940OSQVectorsScorer.BULK_SIZE, STRIPED);
                for (int bulk = 0; bulk < numBulks; bulk++) {
                    final boolean masked = randomBoolean();
                    final int mask = masked ? randomIntBetween(0, Integer.MAX_VALUE) & BulkMask.all(count) | 1 : BulkMask.all(count);
                    final float[][] scores = new float[3][ES940OSQVectorsScorer.BULK_SIZE];
                    final float[] max = new float[3];
                    for (int s = 0; s < 3; s++) {
                        inputs[s].seek(padding + bulk * perBulk);
                        if (s == 0 && random().nextBoolean()) {
                            // raw lookup-table sums, exact integers
                            scorers[s].quantizeScoreBulk(query, count, scores[s]);
                            for (int d = 0; d < count; d++) {
                                assertEquals(expectedRaw.get(bulk)[d], scores[s][d], 0f);
                            }
                            inputs[s].seek(padding + bulk * perBulk);
                        }
                        max[s] = masked
                            ? scorers[s].scoreBulkMasked(
                                query,
                                0f,
                                queryUpper,
                                0,
                                queryAdditional,
                                similarityFunction,
                                0f,
                                mask,
                                scores[s],
                                count
                            )
                            : scorers[s].scoreBulk(query, 0f, queryUpper, 0, queryAdditional, similarityFunction, 0f, scores[s], count);
                        assertEquals(padding + (bulk + 1) * perBulk, inputs[s].getFilePointer());
                    }
                    for (int s = 1; s < 3; s++) {
                        assertEquals(max[0], max[s], 1e-4f * Math.max(1f, Math.abs(max[0])));
                        for (int d = 0; d < count; d++) {
                            if (BulkMask.isSelected(mask, d)) {
                                assertEquals(scores[0][d], scores[s][d], 1e-4f * Math.max(1f, Math.abs(scores[0][d])));
                            }
                        }
                    }
                }
            }
        }
    }

    @ParametersFactory
    public static Iterable<Object[]> parametersFactory() {
        List<Object[]> params = new ArrayList<>();
        for (boolean mmap : new boolean[] { true, false }) {
            for (VectorSimilarityFunction f : new VectorSimilarityFunction[] {
                VectorSimilarityFunction.EUCLIDEAN,
                VectorSimilarityFunction.DOT_PRODUCT,
                VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT }) {
                params.add(new Object[] { mmap, f });
            }
        }
        return params;
    }
}
