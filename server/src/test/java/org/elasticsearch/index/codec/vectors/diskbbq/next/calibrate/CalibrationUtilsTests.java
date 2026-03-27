/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq.next.calibrate;

import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.elasticsearch.index.codec.vectors.cluster.KMeansFloatVectorValues;
import org.elasticsearch.test.ESTestCase;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class CalibrationUtilsTests extends ESTestCase {

    public void testDot() {
        float[] a = { 1, 2, 3 };
        float[] b = { 4, 5, 6 };
        assertEquals(32.0, CalibrationUtils.dot(3, a, b), 1e-10);
    }

    public void testEuclideanSq() {
        float[] a = { 1, 0, 0 };
        float[] b = { 0, 1, 0 };
        assertEquals(2.0, CalibrationUtils.euclideanSq(3, a, b), 1e-10);
    }

    public void testNormalize() {
        float[][] vectors = { { 3, 4 } };
        CalibrationUtils.normalize(vectors);
        assertEquals(0.6f, vectors[0][0], 1e-6);
        assertEquals(0.8f, vectors[0][1], 1e-6);
    }

    public void testNormalizeZeroVector() {
        float[][] vectors = { { 0, 0 } };
        CalibrationUtils.normalize(vectors);
        assertEquals(0f, vectors[0][0], 1e-10);
        assertEquals(0f, vectors[0][1], 1e-10);
    }

    public void testNormalizeVector() {
        float[] v = { 3, 4 };
        CalibrationUtils.normalizeVector(v);
        assertEquals(0.6f, v[0], 1e-6);
        assertEquals(0.8f, v[1], 1e-6);
    }

    public void testCopyAndNormalize() {
        float[] src = { 3, 4 };
        float[] scratch = new float[2];
        float[] result = CalibrationUtils.copyAndNormalize(src, scratch);
        assertSame(scratch, result);
        assertEquals(0.6f, scratch[0], 1e-6);
        assertEquals(0.8f, scratch[1], 1e-6);
        assertEquals(3f, src[0], 1e-10);
        assertEquals(4f, src[1], 1e-10);
    }

    public void testSampleDataReturnsDisjointSets() throws IOException {
        int dim = 4;
        int n = 300;
        List<float[]> vectors = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            float[] v = new float[dim];
            for (int d = 0; d < dim; d++) {
                v[d] = randomFloat();
            }
            vectors.add(v);
        }
        KMeansFloatVectorValues fvv = KMeansFloatVectorValues.build(vectors, null, dim);

        CalibrationUtils.SampledData sampled = CalibrationUtils.sampleData(fvv, dim);
        assertNotNull(sampled.queryOrdinals());
        assertNotNull(sampled.corpusOrdinals());
        assertTrue(sampled.queryOrdinals().length > 0);
        assertTrue(sampled.corpusOrdinals().length > 0);
        assertEquals(n, sampled.queryOrdinals().length + sampled.corpusOrdinals().length);
        Set<Integer> querySet = new HashSet<>();
        for (int o : sampled.queryOrdinals()) {
            querySet.add(o);
        }
        for (int o : sampled.corpusOrdinals()) {
            assertFalse("query and corpus ordinals must be disjoint", querySet.contains(o));
        }
    }

    public void testSampleDataUsesCxxQueryCap() throws IOException {
        int dim = 3;
        int n = 4000;
        List<float[]> vectors = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            vectors.add(new float[] { randomFloat(), randomFloat(), randomFloat() });
        }
        KMeansFloatVectorValues fvv = KMeansFloatVectorValues.build(vectors, null, dim);

        CalibrationUtils.SampledData sampled = CalibrationUtils.sampleData(fvv, dim);
        assertEquals(1024, sampled.queryOrdinals().length);
        assertEquals(n - 1024, sampled.corpusOrdinals().length);
    }

    public void testNeedsNeyshaburSrebroLift() {
        assertTrue(CalibrationUtils.needsNeyshaburSrebroLift(VectorSimilarityFunction.DOT_PRODUCT));
        assertTrue(CalibrationUtils.needsNeyshaburSrebroLift(VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT));
        assertFalse(CalibrationUtils.needsNeyshaburSrebroLift(VectorSimilarityFunction.COSINE));
        assertFalse(CalibrationUtils.needsNeyshaburSrebroLift(VectorSimilarityFunction.EUCLIDEAN));
    }

    public void testCalibrationQueriesNeyshaburAppendsZero() throws IOException {
        float[][] q = { { 1, 2, 3 }, { 0, 0, 0 } };
        CalibrationQueries c = CalibrationQueries.fromMaterializedRows(q, 3, false, true, null, 4);
        float[] dst = new float[4];
        c.copyQuery(0, false, dst);
        assertEquals(0f, dst[3], 0f);
        c.copyQuery(1, false, dst);
        assertEquals(0f, dst[3], 0f);
    }

    public void testCalibrationQueriesCosineNormalizes() throws IOException {
        float[][] rows = { { 3f, 4f } };
        CalibrationQueries c = CalibrationQueries.fromMaterializedRows(rows, 2, true, false, null, 2);
        float[] dst = new float[2];
        c.copyQuery(0, false, dst);
        assertEquals(0.6f, dst[0], 1e-5f);
        assertEquals(0.8f, dst[1], 1e-5f);
    }

    public void testNeyshaburCorpusLiftMatchesMaxNorm() throws IOException {
        int dim = 2;
        List<float[]> vectors = List.of(new float[] { 3, 4 }, new float[] { 0, 1 });
        FloatVectorValues fvv = KMeansFloatVectorValues.build(vectors, null, dim);
        int[] corpusOrdinals = { 0, 1 };
        double maxNormSq = CalibrationUtils.maxSquaredNormOverCorpusSample(fvv, corpusOrdinals, dim);
        assertEquals(25.0, maxNormSq, 1e-10);
        CalibrationUtils.NeyshaburCorpusFloatVectorValues lifted = new CalibrationUtils.NeyshaburCorpusFloatVectorValues(
            fvv,
            dim,
            maxNormSq
        );
        assertEquals(3, lifted.dimension());
        float[] v0 = lifted.vectorValue(0);
        assertEquals(3f, v0[0], 0f);
        assertEquals(4f, v0[1], 0f);
        assertEquals(0f, v0[2], 0f);
        float[] v1 = lifted.vectorValue(1);
        assertEquals(0f, v1[0], 0f);
        assertEquals(1f, v1[1], 0f);
        assertEquals((float) Math.sqrt(25.0 - 1.0), v1[2], 1e-5f);
    }
}
