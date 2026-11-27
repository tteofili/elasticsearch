/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.ash;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.elasticsearch.index.codec.vectors.diskbbq.quantize.PreparedQuery;
import org.elasticsearch.simdvec.ESVectorUtil;
import org.elasticsearch.test.ESTestCase;

import java.io.IOException;

import static org.hamcrest.Matchers.closeTo;
import static org.hamcrest.Matchers.greaterThan;
import static org.hamcrest.Matchers.lessThan;

/** The query side of ASH, as a {@link AshQueryEncoder}: projected once, quantized once, on a min-max grid. */
public class AshQueryEncoderTests extends ESTestCase {

    private static final int DIM = 64;

    private static float[] randomVector(double scale) {
        float[] v = new float[DIM];
        for (int d = 0; d < DIM; d++) {
            v[d] = (float) (random().nextGaussian() * scale);
        }
        return v;
    }

    private static double distance(PreparedQuery prepared, int latentDimension) throws IOException {
        float[] exact = new float[latentDimension];
        float[] served = new float[latentDimension];
        prepared.exactLatent(exact);
        prepared.servedLatent(served);
        double sum = 0;
        for (int d = 0; d < latentDimension; d++) {
            sum += (exact[d] - served[d]) * (double) (exact[d] - served[d]);
        }
        return Math.sqrt(sum);
    }

    private static double norm(float[] v) {
        return Math.sqrt(ESVectorUtil.dotProduct(v, v));
    }

    private static AshProjectionMatrix randomMatrix(int bits) throws IOException {
        AsymmetricHashingQuantizer trainer = new AsymmetricHashingQuantizer(
            1.0f,
            Integer.MAX_VALUE,
            bits,
            AsymmetricHashingQuantizer.Method.RANDOM,
            0,
            1,
            randomLong(),
            null
        );
        return trainer.train(ord -> new float[DIM], 1, DIM, ord -> new float[DIM]).matrix();
    }

    public void testAshProjectsTheQueryOnceWhateverTheList() throws IOException {
        AshProjectionMatrix matrix = randomMatrix(2);
        float[] query = randomVector(1);
        PreparedQuery prepared = new AshQueryEncoder(matrix, 2, 4, VectorSimilarityFunction.DOT_PRODUCT, ordinal -> new float[DIM]).prepare(
            query
        );
        float[] first = new float[matrix.nDims()];
        float[] second = new float[matrix.nDims()];
        prepared.forList(0);
        prepared.servedLatent(first);
        prepared.forList(randomIntBetween(1, 1000));
        prepared.servedLatent(second);
        assertArrayEquals(first, second, 0f);
    }

    public void testAshWithoutQueryQuantizationServesTheInt8Projection() throws IOException {
        AshProjectionMatrix matrix = randomMatrix(1);
        float[] query = randomVector(1);
        PreparedQuery prepared = new AshQueryEncoder(matrix, 1, 0, VectorSimilarityFunction.DOT_PRODUCT, ordinal -> new float[DIM]).prepare(
            query
        );
        prepared.forList(0);
        float[] served = new float[matrix.nDims()];
        prepared.servedLatent(served);
        assertArrayEquals(matrix.projectQuery(query), served, 0f);
        // which is the exact projection up to the int8 rounding of the matrix and the query
        assertThat(distance(prepared, matrix.nDims()), lessThan(0.05 * norm(query)));
    }

    public void testAshQueryBitsMeanASmallerError() throws IOException {
        AshProjectionMatrix matrix = randomMatrix(1);
        float[] query = randomVector(1);
        double previous = Double.MAX_VALUE;
        for (int queryBits : new int[] { 1, 2, 4, 8 }) {
            PreparedQuery prepared = new AshQueryEncoder(
                matrix,
                1,
                queryBits,
                VectorSimilarityFunction.DOT_PRODUCT,
                ordinal -> new float[DIM]
            ).prepare(query);
            prepared.forList(0);
            double error = distance(prepared, matrix.nDims());
            assertThat("queryBits=" + queryBits, error, lessThan(previous));
            previous = error;
        }
    }

    public void testAshGridStandsForWhatItQuantized() {
        float[] projected = { -2f, 0f, 1f, 6f };
        int[] levels = new int[projected.length];
        AshQueryEncoder.Grid grid = AshQueryEncoder.quantizeToLevels(projected, projected.length, 2, levels);
        assertThat((double) grid.min(), closeTo(-2.0, 0));
        assertThat((double) grid.inverse(), closeTo(8.0 / 3, 1e-6));
        assertArrayEquals(new int[] { 0, 1, 1, 3 }, levels);
        // and a constant query does not divide by zero
        int[] constant = new int[3];
        AshQueryEncoder.Grid flat = AshQueryEncoder.quantizeToLevels(new float[] { 1f, 1f, 1f }, 3, 4, constant);
        assertThat((double) flat.inverse(), closeTo(0.0, 0));
        assertThat((double) flat.scale(), greaterThan(0.0));
        assertArrayEquals(new int[] { 0, 0, 0 }, constant);
    }
}
