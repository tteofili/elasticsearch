/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq.next.calibrate;

import org.elasticsearch.index.codec.vectors.cluster.KMeansFloatVectorValues;
import org.elasticsearch.test.ESTestCase;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

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
        assertNotNull(sampled.queries());
        assertNotNull(sampled.corpusOrdinals());
        assertTrue(sampled.queries().length > 0);
        assertTrue(sampled.corpusOrdinals().length > 0);
        assertEquals(n, sampled.queries().length + sampled.corpusOrdinals().length);
    }
}
