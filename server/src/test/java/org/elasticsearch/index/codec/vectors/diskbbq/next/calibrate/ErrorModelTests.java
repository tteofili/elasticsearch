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

public class ErrorModelTests extends ESTestCase {

    public void testCxxParityConstants() {
        assertEquals(32, ErrorModel.N_QUERY_CLUSTERS);
        assertArrayEquals(
            new int[] { 256, 240, 224, 216, 200, 184, 176, 160, 144, 136, 120, 104, 96, 80, 64 },
            ErrorModel.N_DOCS_PER_CLUSTER_SCALING
        );
        assertArrayEquals(new int[] { 64, 72, 80, 88, 96, 104, 112, 120, 128 }, ErrorModel.N_DOCS_PER_CLUSTER_MAGNITUDE);
        assertEquals(4096, ErrorModel.SAMPLE_SIZE_MAGNITUDE);
    }

    public void testOnlineMeanAndVariance() {
        OnlineMeanAndVariance mv = new OnlineMeanAndVariance();
        assertEquals(0.0, mv.mean(), 1e-15);
        assertEquals(0.0, mv.var(), 1e-15);

        mv.add(2.0);
        assertEquals(2.0, mv.mean(), 1e-15);
        assertEquals(0.0, mv.var(), 1e-15);

        mv.add(4.0);
        assertEquals(3.0, mv.mean(), 1e-15);
        assertEquals(2.0, mv.var(), 1e-15);

        mv.add(6.0);
        assertEquals(4.0, mv.mean(), 1e-15);
        assertEquals(4.0, mv.var(), 1e-15);
    }

    public void testOnlineMeanAndVarianceConstant() {
        OnlineMeanAndVariance mv = new OnlineMeanAndVariance();
        for (int i = 0; i < 100; i++) {
            mv.add(5.0);
        }
        assertEquals(5.0, mv.mean(), 1e-10);
        assertEquals(0.0, mv.var(), 1e-10);
    }

    public void testSimExactDotProduct() {
        float[] a = { 1, 2, 3 };
        float[] b = { 4, 5, 6 };
        assertEquals(32.0, ErrorModel.simExact(VectorSimilarityFunction.DOT_PRODUCT, 3, a, b), 1e-10);
    }

    public void testSimExactCosine() {
        float[] a = { 1, 0 };
        float[] b = { 0, 1 };
        assertEquals(0.0, ErrorModel.simExact(VectorSimilarityFunction.COSINE, 2, a, b), 1e-10);
    }

    public void testSimExactEuclidean() {
        float[] a = { 1, 0 };
        float[] b = { 0, 1 };
        // sim = 2*dot(a,b) - dot(b,b) = 2*0 - 1 = -1
        assertEquals(-1.0, ErrorModel.simExact(VectorSimilarityFunction.EUCLIDEAN, 2, a, b), 1e-10);
    }

    public void testSimExactEuclideanSameVector() {
        float[] a = { 3, 4 };
        // sim = 2*dot(a,a) - dot(a,a) = dot(a,a) = 25
        assertEquals(25.0, ErrorModel.simExact(VectorSimilarityFunction.EUCLIDEAN, 2, a, a), 1e-10);
    }

    public void testDotInt() {
        int[] x = { 1, 2, 3 };
        int[] y = { 4, 5, 6 };
        assertEquals(32L, ErrorModel.dotInt(3, x, 0, y, 0));
    }

    public void testDotIntWithOffset() {
        int[] x = { 0, 1, 2, 3 };
        int[] y = { 0, 0, 4, 5, 6 };
        assertEquals(32L, ErrorModel.dotInt(3, x, 1, y, 2));
    }

    public void testCentroidRepErrorStdPerfectClusters() throws IOException {
        int dim = 2;
        float[][] queries = { { 1f, 0f } };
        List<float[]> corpusList = new ArrayList<>();
        corpusList.add(new float[] { 1f, 0f });
        corpusList.add(new float[] { 1f, 0.1f });
        KMeansFloatVectorValues fvv = KMeansFloatVectorValues.build(corpusList, null, dim);
        int[] corpusOrdinals = { 0, 1 };
        float[][] centroids = { { 1f, 0.05f } };
        int[][] perCluster = { { 0, 1 } };

        CalibrationQueries calibrationQueries = CalibrationQueries.fromMaterializedRows(queries, dim, false, false, null, dim);
        double std = ErrorModel.centroidRepErrorStd(
            VectorSimilarityFunction.DOT_PRODUCT,
            dim,
            calibrationQueries,
            false,
            fvv,
            corpusOrdinals,
            false,
            perCluster,
            centroids
        );
        assertTrue("centroid error std should be non-negative", std >= 0);
    }

    public void testCentroidRepErrorStdIdentityCentroids() throws IOException {
        int dim = 2;
        float[][] queries = { { 1f, 0f } };
        List<float[]> corpusList = new ArrayList<>();
        corpusList.add(new float[] { 1f, 0f });
        corpusList.add(new float[] { 0f, 1f });
        KMeansFloatVectorValues fvv = KMeansFloatVectorValues.build(corpusList, null, dim);
        int[] corpusOrdinals = { 0, 1 };
        float[][] centroids = { { 1f, 0f }, { 0f, 1f } };
        int[][] perCluster = { { 0 }, { 1 } };

        CalibrationQueries calibrationQueries = CalibrationQueries.fromMaterializedRows(queries, dim, false, false, null, dim);
        double std = ErrorModel.centroidRepErrorStd(
            VectorSimilarityFunction.DOT_PRODUCT,
            dim,
            calibrationQueries,
            false,
            fvv,
            corpusOrdinals,
            false,
            perCluster,
            centroids
        );
        assertEquals(0.0, std, 1e-10);
    }

    public void testQuantizedRepErrorStdSmall() throws IOException {
        int dim = 4;
        float[][] queries = { { 1f, 0f, 0f, 0f }, { 0f, 1f, 0f, 0f } };
        List<float[]> corpusList = new ArrayList<>();
        for (int i = 0; i < 32; i++) {
            float[] v = new float[dim];
            for (int j = 0; j < dim; j++) {
                v[j] = randomFloat() * 2 - 1;
            }
            corpusList.add(v);
        }
        KMeansFloatVectorValues fvv = KMeansFloatVectorValues.build(corpusList, null, dim);
        int[] corpusOrdinals = new int[32];
        for (int i = 0; i < 32; i++) {
            corpusOrdinals[i] = i;
        }
        float[][] centroids = { { 0.5f, 0f, 0f, 0f }, { -0.5f, 0f, 0f, 0f } };
        int[] assignments = new int[32];
        for (int i = 0; i < 32; i++) {
            assignments[i] = i < 16 ? 0 : 1;
        }

        CalibrationQueries calibrationQueries = CalibrationQueries.fromMaterializedRows(queries, dim, false, false, null, dim);
        double std = ErrorModel.quantizedRepErrorStd(
            VectorSimilarityFunction.DOT_PRODUCT,
            dim,
            calibrationQueries,
            false,
            fvv,
            corpusOrdinals,
            false,
            assignments,
            centroids,
            2,
            16,
            4,
            1,
            10
        );
        assertTrue("quantized error std should be non-negative", std >= 0);
    }

    public void testRepErrorStdsSmallCorpus() throws IOException {
        int dim = 4;
        float[][] queries = { { 1f, 0f, 0f, 0f } };
        List<float[]> corpusList = new ArrayList<>();
        for (int i = 0; i < 64; i++) {
            float[] v = new float[dim];
            for (int j = 0; j < dim; j++) {
                v[j] = randomFloat() * 2 - 1;
            }
            corpusList.add(v);
        }
        KMeansFloatVectorValues fvv = KMeansFloatVectorValues.build(corpusList, null, dim);
        int[] corpusOrdinals = new int[64];
        for (int i = 0; i < 64; i++) {
            corpusOrdinals[i] = i;
        }

        CalibrationQueries calibrationQueries = CalibrationQueries.fromMaterializedRows(queries, dim, false, false, null, dim);
        double[] stds = ErrorModel.repErrorStds(
            VectorSimilarityFunction.DOT_PRODUCT,
            dim,
            calibrationQueries,
            false,
            fvv,
            corpusOrdinals,
            corpusOrdinals.length,
            false,
            2,
            32,
            4,
            1,
            10
        );
        assertEquals(2, stds.length);
        assertTrue("centroid std should be non-negative", stds[0] >= 0);
        assertTrue("quantized std should be non-negative", stds[1] >= 0);
    }

    public void testFitRepErrorStdPlugin() {
        Regression.OLSResult scalingQ = new Regression.OLSResult(-2.0, 0.5, 0.01, 0.01, 0.0, 0.01);
        RepErrorStdModel scalingModel = new RepErrorStdModel(Regression.OLSResult.ZERO, scalingQ);

        double[] logNDocs = { Math.log(64), Math.log(72), Math.log(80), Math.log(88) };
        double[] logSizes = { Math.log(4096), Math.log(4096), Math.log(4096), Math.log(4096) };
        double[] logErrorStd = new double[4];
        for (int i = 0; i < 4; i++) {
            double x = logNDocs[i] - logSizes[i];
            logErrorStd[i] = -1.5 + 0.5 * x;
        }

        Regression.OLSResult result = ErrorModel.fitRepErrorStdPlugin(scalingModel, logNDocs, logSizes, logErrorStd);

        assertEquals(0.5, result.beta1(), 1e-10);
        assertEquals(-1.5, result.beta0(), 1e-6);
        assertTrue("sigmaSq should be near zero for perfect data", result.sigmaSq() < 1e-10);
    }

    public void testScalingParameterTooFewSamples() {
        float[][] queries = { { 1f, 0f } };
        List<float[]> corpusList = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            corpusList.add(new float[] { randomFloat(), randomFloat() });
        }
        KMeansFloatVectorValues fvv = KMeansFloatVectorValues.build(corpusList, null, 2);
        int[] corpusOrdinals = new int[10];
        for (int i = 0; i < 10; i++) {
            corpusOrdinals[i] = i;
        }

        CalibrationQueries calibrationQueries = CalibrationQueries.fromMaterializedRows(queries, 2, false, false, null, 2);
        RepErrorStdModel model = ErrorModel.estimateRepErrorStdScalingParameter(
            VectorSimilarityFunction.DOT_PRODUCT,
            2,
            calibrationQueries,
            fvv,
            corpusOrdinals,
            false,
            10
        );
        assertSame(Regression.OLSResult.ZERO, model.cparams());
        assertSame(Regression.OLSResult.ZERO, model.qparams());
    }

    public void testMagnitudeParameterPreservesSlope() {
        Regression.OLSResult c = new Regression.OLSResult(-1, 0.5, 0.01, 0.01, 0, 0.01);
        Regression.OLSResult q = new Regression.OLSResult(-2, 0.3, 0.01, 0.01, 0, 0.01);
        RepErrorStdModel scalingModel = new RepErrorStdModel(c, q);

        int dim = 4;
        float[][] queries = { { 1f, 0f, 0f, 0f } };
        List<float[]> corpusList = new ArrayList<>();
        for (int i = 0; i < 64; i++) {
            float[] v = new float[dim];
            for (int j = 0; j < dim; j++) {
                v[j] = randomFloat() * 2 - 1;
            }
            corpusList.add(v);
        }
        KMeansFloatVectorValues fvv = KMeansFloatVectorValues.build(corpusList, null, dim);
        int[] corpusOrdinals = new int[64];
        for (int i = 0; i < 64; i++) {
            corpusOrdinals[i] = i;
        }

        CalibrationQueries calibrationQueries = CalibrationQueries.fromMaterializedRows(queries, dim, false, false, null, dim);
        RepErrorStdModel result = ErrorModel.estimateRepErrorStdMagnitudeParameter(
            scalingModel,
            VectorSimilarityFunction.DOT_PRODUCT,
            dim,
            calibrationQueries,
            false,
            fvv,
            corpusOrdinals,
            false,
            10,
            4,
            1
        );
        assertSame("cparams should be preserved from scaling model", c, result.cparams());
        assertEquals("slope should be preserved from scaling model", q.beta1(), result.qparams().beta1(), 1e-10);
    }

    public void testMagnitudeParameterSmallCorpusStillFits() {
        Regression.OLSResult c = new Regression.OLSResult(-1, 0.5, 0.01, 0.01, 0, 0.01);
        Regression.OLSResult q = new Regression.OLSResult(-2, 0.3, 0.01, 0.01, 0, 0.01);
        RepErrorStdModel scalingModel = new RepErrorStdModel(c, q);

        float[][] queries = { { 1f } };
        List<float[]> corpusList = new ArrayList<>();
        corpusList.add(new float[] { 0.5f });
        KMeansFloatVectorValues fvv = KMeansFloatVectorValues.build(corpusList, null, 1);
        int[] corpusOrdinals = { 0 };

        CalibrationQueries calibrationQueries = CalibrationQueries.fromMaterializedRows(queries, 1, false, false, null, 1);
        RepErrorStdModel result = ErrorModel.estimateRepErrorStdMagnitudeParameter(
            scalingModel,
            VectorSimilarityFunction.DOT_PRODUCT,
            1,
            calibrationQueries,
            false,
            fvv,
            corpusOrdinals,
            false,
            10,
            4,
            1
        );
        assertSame("cparams should always be preserved", c, result.cparams());
        assertEquals("slope should be preserved from scaling model", q.beta1(), result.qparams().beta1(), 1e-10);
    }

    public void testRepErrorStdModelAccessors() {
        Regression.OLSResult c = new Regression.OLSResult(1, 2, 3, 4, 5, 6);
        Regression.OLSResult q = new Regression.OLSResult(7, 8, 9, 10, 11, 12);
        RepErrorStdModel model = new RepErrorStdModel(c, q);
        assertSame(c, model.cparams());
        assertSame(q, model.qparams());
    }
}
