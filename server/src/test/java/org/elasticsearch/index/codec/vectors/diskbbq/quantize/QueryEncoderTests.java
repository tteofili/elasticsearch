/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq.quantize;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.elasticsearch.index.codec.vectors.diskbbq.QuantEncoding;
import org.elasticsearch.simdvec.ESVectorUtil;
import org.elasticsearch.test.ESTestCase;

import java.io.IOException;
import java.util.concurrent.atomic.AtomicInteger;

import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.lessThan;

/**
 * What any {@link QueryEncoder} has to satisfy: a prepared query follows the lists it is asked for, and what the scorer
 * ends up using in place of the exact query is close to it, closer the more bits the query has.
 */
public class QueryEncoderTests extends ESTestCase {

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

    public void testOsqQuantizesTheQueryAgainstTheCentroidOfTheList() throws IOException {
        float[] query = randomVector(1);
        float[] centroid = randomVector(0.2);
        OsqQueryEncoder encoder = new OsqQueryEncoder(
            VectorSimilarityFunction.DOT_PRODUCT,
            QuantEncoding.ONE_BIT_4BIT_QUERY,
            DIM,
            ordinal -> centroid
        );
        PreparedQuery prepared = encoder.prepare(query);
        prepared.forList(3);
        float[] exact = new float[DIM];
        prepared.exactLatent(exact);
        for (int d = 0; d < DIM; d++) {
            assertEquals(query[d] - centroid[d], exact[d], 0f);
        }
    }

    public void testMoreQueryBitsMeanASmallerError() throws IOException {
        float[] query = randomVector(1);
        float[] centroid = randomVector(0.2);
        double previous = Double.MAX_VALUE;
        for (QuantEncoding encoding : new QuantEncoding[] {
            QuantEncoding.ONE_BIT_1BIT_QUERY,
            QuantEncoding.ONE_BIT_4BIT_QUERY,
            QuantEncoding.SEVEN_BIT_SYMMETRIC }) {
            PreparedQuery prepared = new OsqQueryEncoder(VectorSimilarityFunction.DOT_PRODUCT, encoding, DIM, ordinal -> centroid).prepare(
                query
            );
            prepared.forList(0);
            double error = distance(prepared, DIM);
            assertThat(encoding.toString(), error, lessThan(previous));
            previous = error;
        }
        // 7 bits is a few percent of the norm of what is quantized
        assertThat(previous, lessThan(0.05 * norm(query)));
    }

    public void testOsqReadsACentroidOnlyForTheListsItQuantizesFor() throws IOException {
        float[] centroid = randomVector(0.2);
        AtomicInteger reads = new AtomicInteger();
        OsqQueryEncoder.Prepared prepared = new OsqQueryEncoder(
            VectorSimilarityFunction.DOT_PRODUCT,
            QuantEncoding.FOUR_BIT_SYMMETRIC,
            DIM,
            ordinal -> {
                reads.incrementAndGet();
                return centroid;
            }
        ).prepare(randomVector(1));
        // moving to a list reads nothing: a list that has nothing to score must not cost a quantization
        prepared.reset(1);
        prepared.reset(2);
        assertThat(reads.get(), equalTo(0));
        prepared.quantizeQueryIfNecessary();
        assertThat(reads.get(), equalTo(1));
        // coming back to a list that was quantized for is served from the cache, and so is staying on it
        prepared.quantizeQueryIfNecessary();
        prepared.reset(1);
        prepared.quantizeQueryIfNecessary();
        prepared.reset(2);
        prepared.quantizeQueryIfNecessary();
        assertThat(reads.get(), equalTo(2));
    }

    public void testOsqKeepsTheQuantizedQueriesOfTheLastFewLists() throws IOException {
        float[] centroid = randomVector(0.2);
        AtomicInteger reads = new AtomicInteger();
        OsqQueryEncoder.Prepared prepared = new OsqQueryEncoder(
            VectorSimilarityFunction.DOT_PRODUCT,
            QuantEncoding.FOUR_BIT_SYMMETRIC,
            DIM,
            ordinal -> {
                reads.incrementAndGet();
                return centroid;
            }
        ).prepare(randomVector(1));
        for (int list = 0; list < 16; list++) {
            prepared.forList(list);
        }
        assertThat(reads.get(), equalTo(16));
        // the sixteen are all kept
        for (int list = 0; list < 16; list++) {
            prepared.forList(list);
        }
        assertThat(reads.get(), equalTo(16));
        // a seventeenth evicts the least recently used one, which then has to be quantized again
        prepared.forList(16);
        prepared.forList(0);
        assertThat(reads.get(), equalTo(18));
    }
}
