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
import org.elasticsearch.test.ESTestCase;

public class ExpectedRecallTests extends ESTestCase {

    public void testExpectedRecallAtKBoundedZeroToOne() {
        double alpha = 2.0;
        double invDim = 0.5;
        int N = 10000;
        int k = 10;
        int rerank = 20;
        double errorStd = 0.1;
        double recall = ExpectedRecall.expectedRecallAtK(VectorSimilarityFunction.EUCLIDEAN, N, alpha, invDim, errorStd, k, rerank);
        assertTrue("recall should be >= 0", recall >= 0);
        assertTrue("recall should be <= 1", recall <= 1);
    }

    public void testHigherRerankImproveRecall() {
        double alpha = 2.0;
        double invDim = 0.5;
        int N = 10000;
        int k = 10;
        double errorStd = 0.5;
        double recallLow = ExpectedRecall.expectedRecallAtK(VectorSimilarityFunction.EUCLIDEAN, N, alpha, invDim, errorStd, k, 15);
        double recallHigh = ExpectedRecall.expectedRecallAtK(VectorSimilarityFunction.EUCLIDEAN, N, alpha, invDim, errorStd, k, 30);
        assertTrue("more reranking should yield >= recall", recallHigh >= recallLow);
    }

    public void testLowerErrorStdGivesHigherRecall() {
        double alpha = 2.0;
        double invDim = 0.5;
        int N = 10000;
        int k = 10;
        int rerank = 20;
        double recallGood = ExpectedRecall.expectedRecallAtK(VectorSimilarityFunction.EUCLIDEAN, N, alpha, invDim, 0.01, k, rerank);
        double recallBad = ExpectedRecall.expectedRecallAtK(VectorSimilarityFunction.EUCLIDEAN, N, alpha, invDim, 1.0, k, rerank);
        assertTrue("lower error std should yield higher recall", recallGood >= recallBad);
    }

    public void testRerankN() {
        assertEquals(15, ExpectedRecall.rerankN(10, 15, 10));
        assertEquals(20, ExpectedRecall.rerankN(10, 2, 1));
        assertEquals(30, ExpectedRecall.rerankN(10, 3, 1));
        assertEquals(150, ExpectedRecall.rerankN(100, 15, 10));
        assertEquals(200, ExpectedRecall.rerankN(100, 2, 1));
        assertEquals(300, ExpectedRecall.rerankN(100, 3, 1));
    }

    public void testNormalPdf() {
        double pdf = ExpectedRecall.normalPdf(0, 0, 1);
        assertEquals(1.0 / Math.sqrt(2.0 * Math.PI), pdf, 1e-10);
    }

    public void testNormalCdf() {
        double cdf = ExpectedRecall.normalCdf(0, 0, 1);
        assertEquals(0.5, cdf, 1e-6);
    }

    public void testErf() {
        assertEquals(0.0, ExpectedRecall.erf(0), 1e-6);
        assertTrue(ExpectedRecall.erf(3.0) > 0.99);
        assertTrue(ExpectedRecall.erf(-3.0) < -0.99);
    }
}
