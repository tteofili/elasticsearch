/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.ash;

import org.apache.lucene.store.IndexInput;
import org.elasticsearch.simdvec.AshScorer;
import org.elasticsearch.simdvec.BulkMask;
import org.elasticsearch.simdvec.E8Lattice;
import org.elasticsearch.simdvec.E8VectorsScorer;

import java.io.IOException;

import static org.elasticsearch.simdvec.ES940OSQVectorsScorer.BULK_SIZE;

/**
 * Scores E8-coded ASH vectors (see {@link E8Codebook}) with the E8 table-lookup kernels of {@link E8VectorsScorer}
 * (native, Panama or scalar). The codes of a bulk are stored transposed (see {@code AshPostingsListWriter}), stage by
 * stage, so a two-stage code is scanned as {@code 2 * nBlocks} "virtual" blocks whose tables are the first-stage table
 * and the same table weighted by {@link E8Codebook#TWO_STAGE_BETA}, both quantized to int8 with one scale. The result is
 * the inner product between the projected query and the code's reconstruction, so the visitor applies the float-path
 * corrections.
 */
final class E8AshScorer implements AshScorer<byte[]> {

    /** The int8 lookup table of a query and the scale that turns its integer sums into inner products. */
    record QueryTable(byte[] table, float scale) {}

    static QueryTable buildQueryTable(float[] queryTransformed, int nDims, int stages) {
        E8Codebook.validateDims(nDims);
        int nBlocks = nDims >>> 3;
        float[] t = new float[nBlocks * E8Lattice.TABLE_STRIDE];
        E8Lattice.buildFloatTable(queryTransformed, nDims, t);
        byte[] table = new byte[stages * nBlocks * E8Lattice.TABLE_STRIDE];
        float scale = stages == 1
            ? E8Lattice.quantizeTable(new float[][] { t }, new float[] { 1f }, nBlocks, table)
            : E8Lattice.quantizeTable(new float[][] { t, t }, new float[] { 1f, E8Codebook.TWO_STAGE_BETA }, nBlocks, table);
        return new QueryTable(table, scale);
    }

    private final E8VectorsScorer kernel;
    private final float scale;
    private final int[] sums = new int[BULK_SIZE];

    E8AshScorer(IndexInput input, int nDims, int stages, float scale) {
        E8Codebook.validateDims(nDims);
        this.kernel = E8VectorsScorer.create(input, E8Codebook.codeBytes(nDims, stages), BULK_SIZE);
        this.scale = scale;
    }

    @Override
    public float score(byte[] table) {
        throw new UnsupportedOperationException("E8 codes are stored transposed within bulks: score them with scoreBulk");
    }

    @Override
    public void scoreBulk(byte[] table, int blockSize, float[] scores) throws IOException {
        assert blockSize <= BULK_SIZE;
        kernel.lookupBulk(table, blockSize, sums);
        for (int j = 0; j < blockSize; j++) {
            scores[j] = scale * sums[j];
        }
    }

    @Override
    public void scoreBulkMasked(byte[] table, int mask, float[] scores, int blockSize) throws IOException {
        assert blockSize <= BULK_SIZE;
        kernel.lookupBulk(table, blockSize, sums);
        for (int m = mask; m != 0; m = BulkMask.deselectFirst(m)) {
            int j = BulkMask.first(m);
            scores[j] = scale * sums[j];
        }
    }
}
