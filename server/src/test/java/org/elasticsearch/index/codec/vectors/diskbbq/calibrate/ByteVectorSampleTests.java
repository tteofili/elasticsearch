/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq.calibrate;

import org.elasticsearch.index.codec.vectors.diskbbq.calibrate.DistortionRateTestUtils.Dataset;
import org.elasticsearch.test.ESTestCase;

import java.io.IOException;
import java.util.List;

import static org.elasticsearch.index.codec.vectors.diskbbq.calibrate.DistortionRateTestUtils.byteBackedSampleOf;
import static org.elasticsearch.index.codec.vectors.diskbbq.calibrate.DistortionRateTestUtils.byteBackedValues;
import static org.elasticsearch.index.codec.vectors.diskbbq.calibrate.DistortionRateTestUtils.clusteredDataset;
import static org.elasticsearch.index.codec.vectors.diskbbq.calibrate.DistortionRateTestUtils.haar;
import static org.elasticsearch.index.codec.vectors.diskbbq.calibrate.DistortionRateTestUtils.ordinals;
import static org.elasticsearch.index.codec.vectors.diskbbq.calibrate.DistortionRateTestUtils.osq;
import static org.elasticsearch.index.codec.vectors.diskbbq.calibrate.DistortionRateTestUtils.roundedToBytes;
import static org.hamcrest.Matchers.closeTo;
import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.greaterThan;
import static org.hamcrest.Matchers.greaterThanOrEqualTo;
import static org.hamcrest.Matchers.lessThanOrEqualTo;

/**
 * Byte vectors are read through the same abstractions as float ones, widened into a buffer that is reused from one read
 * to the next. These tests hold a corpus of byte values twice, as floats and as bytes, and check that the estimator
 * cannot tell them apart: a read that outlived the next one would show as a difference.
 */
public class ByteVectorSampleTests extends ESTestCase {

    private static final int DIM = 64;
    private static final int K = 10;
    private static final float POOL_TOLERANCE = 0.5f;

    private final Dataset data = roundedToBytes(clusteredDataset(40, 8, DIM, 6, randomLong(), 1.0f));
    private final ClusteredSample floats = data.sample();
    private final ClusteredSample bytes = byteBackedSampleOf(data);

    public void testByteBackedSampleHoldsTheNumbersOfItsFloats() throws IOException {
        assertEquals(floats.size(), bytes.size());
        assertEquals(floats.dimension(), bytes.dimension());
        DistortionRateQuantizer.Residuals floatResiduals = floats.residuals(floats.size());
        DistortionRateQuantizer.Residuals byteResiduals = bytes.residuals(bytes.size());
        for (int i = 0; i < floats.size(); i++) {
            assertArrayEquals(floats.vector(i), bytes.vector(i), 0f);
            assertArrayEquals(floats.centroid(i), bytes.centroid(i), 0f);
            assertArrayEquals(floatResiduals.residual(i), byteResiduals.residual(i), 0f);
        }
    }

    public void testPoolsDoNotDependOnHowTheVectorsAreHeld() throws IOException {
        DistortionRateQuantizer quantizer = osq(DIM, 1, 0.1f, RotationKind.HAAR, haar(DIM, 3));
        boolean quantizedQuery = randomBoolean();
        List<RecallEstimator.Pool> expected = RecallEstimator.pools(
            quantizer,
            floats,
            data.queryValues(),
            data.queryOrdinals(),
            K,
            100,
            null,
            quantizedQuery
        );
        List<RecallEstimator.Pool> actual = RecallEstimator.pools(
            quantizer,
            bytes,
            byteBackedValues(data.queries()),
            ordinals(data.queries().length),
            K,
            100,
            null,
            quantizedQuery
        );
        assertThat(actual.size(), equalTo(expected.size()));
        for (int q = 0; q < expected.size(); q++) {
            RecallEstimator.Pool e = expected.get(q);
            RecallEstimator.Pool a = actual.get(q);
            assertThat(a.size(), equalTo(e.size()));
            assertArrayEquals(e.centroidScore(), a.centroidScore(), POOL_TOLERANCE);
            assertArrayEquals(e.gain(), a.gain(), POOL_TOLERANCE);
            assertArrayEquals(e.alpha(), a.alpha(), POOL_TOLERANCE);
            assertArrayEquals(e.r(), a.r(), POOL_TOLERANCE);
            assertArrayEquals(e.isTrue(), a.isTrue());
            if (quantizedQuery) {
                assertArrayEquals(e.queryError(), a.queryError(), POOL_TOLERANCE);
            }
        }
    }

    public void testRecallEstimateOfByteVectorsIsAProbabilityThatGrowsWithOversampling() throws IOException {
        DistortionRateQuantizer quantizer = osq(DIM, 2, 0.1f, RotationKind.HAAR, haar(DIM, 5));
        CodebookProfile profile = CodebookProfile.sample(quantizer, CodebookProfile.DEFAULT_SAMPLES, 7);
        List<RecallEstimator.Pool> pools = RecallEstimator.pools(
            quantizer,
            bytes,
            byteBackedValues(data.queries()),
            ordinals(data.queries().length),
            K,
            100,
            null,
            false
        );
        double[] recall = RecallEstimator.estimate(profile, pools, new int[] { 1, 3 }, K, quantizer.latentDimension(), 16, 11);
        assertThat(recall[0], greaterThan(0.0));
        assertThat(recall[1], greaterThanOrEqualTo(recall[0]));
        assertThat(recall[1], lessThanOrEqualTo(1.0));
    }

    public void testIsotropyCheckSeesTheSameResidualsWhetherHeldAsBytesOrFloats() throws IOException {
        DistortionRateQuantizer quantizer = osq(DIM, 1, 0.1f, RotationKind.HAAR, haar(DIM, 9));
        CodebookProfile isotropic = CodebookProfile.sample(quantizer, 1000, 1);
        IsotropyCheck.Report fromFloats = IsotropyCheck.check(quantizer, floats.residuals(floats.size()), isotropic);
        IsotropyCheck.Report fromBytes = IsotropyCheck.check(quantizer, bytes.residuals(bytes.size()), isotropic);
        assertThat(fromBytes.gap(), closeTo(fromFloats.gap(), 1e-6));
        assertThat(fromBytes.weakRotation(), equalTo(fromFloats.weakRotation()));
    }
}
