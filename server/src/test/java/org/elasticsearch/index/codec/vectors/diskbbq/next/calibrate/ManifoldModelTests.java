/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq.next.calibrate;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.elasticsearch.index.codec.vectors.cluster.KMeansFloatVectorValues;
import org.elasticsearch.test.ESTestCase;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class ManifoldModelTests extends ESTestCase {

    public void testRanksForKFollowCxxRule() {
        int[] ranks = ManifoldModel.ranksForK(100);
        assertEquals(25, ranks.length);
        assertEquals(580, ranks[0]);
        assertEquals(100, ranks[24]);
    }

    public void testRanksForKFastFollowCxxRule() {
        int[] ranks = ManifoldModel.ranksForKFast(100);
        assertArrayEquals(new int[] { 580, 500, 420, 340, 260, 180, 140, 100 }, ranks);
    }

    public void testExpectedRankDistanceEuclidean() {
        double alpha = 2.0;
        double invDim = 0.5;
        int N = 10000;
        double d1 = ManifoldModel.expectedRankDistance(VectorSimilarityFunction.EUCLIDEAN, alpha, invDim, N, 1);
        double d10 = ManifoldModel.expectedRankDistance(VectorSimilarityFunction.EUCLIDEAN, alpha, invDim, N, 10);
        assertTrue("distance at rank 10 should be >= distance at rank 1", d10 >= d1);
        assertTrue("distance should be positive for Euclidean", d1 > 0);
    }

    public void testExpectedRankDistanceDot() {
        double alpha = 2.0;
        double invDim = 0.5;
        int N = 10000;
        double d1 = ManifoldModel.expectedRankDistance(VectorSimilarityFunction.DOT_PRODUCT, alpha, invDim, N, 1);
        double d10 = ManifoldModel.expectedRankDistance(VectorSimilarityFunction.DOT_PRODUCT, alpha, invDim, N, 10);
        assertTrue("dot product distance should be negative (negated similarity)", d1 < 0);
        assertTrue("rank 1 should be less negative than rank 10 (higher similarity)", d1 > d10);
    }

    public void testEstimateManifoldParametersReturnsNonZero() throws IOException {
        int dim = 8;
        int nQueries = 50;
        int nCorpus = 16384;
        Random rng = new Random(42);
        float[][] queries = new float[nQueries][dim];
        List<float[]> corpusList = new ArrayList<>();
        for (int i = 0; i < nQueries; i++) {
            for (int d = 0; d < dim; d++) {
                queries[i][d] = rng.nextFloat() - 0.5f;
            }
        }
        for (int i = 0; i < nCorpus; i++) {
            float[] v = new float[dim];
            for (int d = 0; d < dim; d++) {
                v[d] = rng.nextFloat() - 0.5f;
            }
            corpusList.add(v);
        }
        KMeansFloatVectorValues fvv = KMeansFloatVectorValues.build(corpusList, null, dim);
        int[] corpusOrdinals = new int[nCorpus];
        for (int i = 0; i < nCorpus; i++) {
            corpusOrdinals[i] = i;
        }
        double[] result = ManifoldModel.estimateManifoldParameters(
            VectorSimilarityFunction.EUCLIDEAN,
            dim,
            queries,
            fvv,
            corpusOrdinals,
            false,
            10
        );
        assertEquals(2, result.length);
        assertNotEquals(0.0, result[0], 1e-15);
        assertNotEquals(0.0, result[1], 1e-15);
    }

    public void testEstimateManifoldParametersTooFewVectors() throws IOException {
        int dim = 8;
        float[][] queries = new float[2][dim];
        List<float[]> corpusList = new ArrayList<>();
        for (int i = 0; i < 100; i++) {
            corpusList.add(new float[dim]);
        }
        KMeansFloatVectorValues fvv = KMeansFloatVectorValues.build(corpusList, null, dim);
        int[] corpusOrdinals = new int[100];
        for (int i = 0; i < 100; i++) {
            corpusOrdinals[i] = i;
        }
        double[] result = ManifoldModel.estimateManifoldParameters(
            VectorSimilarityFunction.EUCLIDEAN,
            dim,
            queries,
            fvv,
            corpusOrdinals,
            false,
            10
        );
        assertEquals(2, result.length);
    }
}
