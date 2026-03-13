/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq.next.calibrate;

import org.elasticsearch.test.ESTestCase;

public class RegressionTests extends ESTestCase {

    public void testFitOlsPerfectLine() {
        double[] x = { 1, 2, 3, 4, 5 };
        double[] y = { 3, 5, 7, 9, 11 };
        Regression.OLSResult result = Regression.fitOls(x, y);
        assertEquals(1.0, result.beta0(), 1e-10);
        assertEquals(2.0, result.beta1(), 1e-10);
        assertEquals(0.0, result.sigmaSq(), 1e-10);
    }

    public void testFitOlsTooFewPoints() {
        double[] x = { 1, 2 };
        double[] y = { 3, 5 };
        Regression.OLSResult result = Regression.fitOls(x, y);
        assertSame(Regression.OLSResult.ZERO, result);
    }

    public void testFitOlsConstantX() {
        double[] x = { 5, 5, 5 };
        double[] y = { 1, 2, 3 };
        Regression.OLSResult result = Regression.fitOls(x, y);
        assertSame(Regression.OLSResult.ZERO, result);
    }

    public void testFitOlsWithNoise() {
        double[] x = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
        double[] y = new double[10];
        for (int i = 0; i < 10; i++) {
            y[i] = 2.0 + 0.5 * x[i];
        }
        y[3] += 0.1;
        y[7] -= 0.05;
        Regression.OLSResult result = Regression.fitOls(x, y);
        assertEquals(2.0, result.beta0(), 0.1);
        assertEquals(0.5, result.beta1(), 0.05);
        assertTrue(result.sigmaSq() > 0);
    }

    public void testPredictOls() {
        Regression.OLSResult model = new Regression.OLSResult(1.0, 2.0, 0.01, 0.01, 0.0, 0.04);
        Regression.Prediction p = Regression.predictOls(model, 3.0);
        assertEquals(7.0, p.mean(), 1e-10);
        assertTrue(p.std() > 0);
    }
}
