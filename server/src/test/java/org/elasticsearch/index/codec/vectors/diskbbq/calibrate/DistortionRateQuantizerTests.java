/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq.calibrate;

import org.elasticsearch.index.codec.vectors.ash.AshDistortionRateQuantizer;
import org.elasticsearch.simdvec.ESVectorUtil;
import org.elasticsearch.test.ESTestCase;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.elasticsearch.index.codec.vectors.diskbbq.calibrate.DistortionRateTestUtils.anisotropicResiduals;
import static org.elasticsearch.index.codec.vectors.diskbbq.calibrate.DistortionRateTestUtils.haar;
import static org.elasticsearch.index.codec.vectors.diskbbq.calibrate.DistortionRateTestUtils.identity;
import static org.elasticsearch.index.codec.vectors.diskbbq.calibrate.DistortionRateTestUtils.osq;
import static org.elasticsearch.index.codec.vectors.diskbbq.calibrate.DistortionRateTestUtils.residualsOf;
import static org.hamcrest.Matchers.closeTo;
import static org.hamcrest.Matchers.greaterThan;
import static org.hamcrest.Matchers.greaterThanOrEqualTo;
import static org.hamcrest.Matchers.lessThan;
import static org.hamcrest.Matchers.lessThanOrEqualTo;

/** OSQ and ASH both satisfy the contract of {@link DistortionRateQuantizer}, and behave as the theory says. */
public class DistortionRateQuantizerTests extends ESTestCase {

    private static final int DIM = 128;

    /** Both quantizer families, at several bit widths, rotated by a dense random rotation (or ASH's own random projection). */
    private static List<DistortionRateQuantizer> quantizers() throws IOException {
        List<DistortionRateQuantizer> out = new ArrayList<>();
        for (int bits : new int[] { 1, 2 }) {
            out.add(osq(DIM, bits, 0.1f, RotationKind.HAAR, haar(DIM, 7)));
        }
        for (int bits : new int[] { 1, 2, 4 }) {
            out.add(AshDistortionRateQuantizer.random(DIM, bits, 4, 11));
        }
        return out;
    }

    public void testEveryQuantizerHonorsTheContract() throws IOException {
        Random random = new Random(1);
        for (DistortionRateQuantizer q : quantizers()) {
            String name = q.getClass().getSimpleName() + " b=" + q.bits();
            assertThat(name, q.latentDimension(), lessThanOrEqualTo(q.dimension()));
            // a residual sampled for the codebook has a unit latent image
            float[] residual = q.sampleResidual(random);
            assertEquals(name, q.dimension(), residual.length);
            float[] latent = q.project(residual);
            assertEquals(name, q.latentDimension(), latent.length);
            assertThat(name, (double) ESVectorUtil.dotProduct(latent, latent), closeTo(1.0, 1e-3));
            // a quantizer reports something for every residual with a direction, and nothing for one without
            DistortionRateQuantizer.Reconstruction[] reconstructions = q.reconstruct(
                residualsOf(new float[][] { residual, new float[q.dimension()] })
            );
            assertFalse(name, reconstructions[0].isEmpty());
            assertTrue(name, reconstructions[1].isEmpty());
            assertThat(name, reconstructions[0].cosine(), greaterThan(0.0));
            assertThat(name, reconstructions[0].cosine(), lessThanOrEqualTo(1.0 + 1e-6));
            assertThat(name, reconstructions[0].b(), greaterThanOrEqualTo(0.0));
            float[] query = q.sampleResidual(random);
            float[] centroid = q.sampleResidual(random);
            double error = q.queryError(query, centroid);
            assertThat(name, error, greaterThanOrEqualTo(0.0));
            assertTrue(name, Double.isFinite(error));
        }
    }

    public void testCodebookConstantsMatchThePaper() throws IOException {
        // c = 2 (1 - E[cos]) on random directions: 2 (1 - sqrt(2 / pi)) = 0.404 at one bit, about 0.12 at two, 0.01 at four
        for (DistortionRateQuantizer q : quantizers()) {
            double c = CodebookProfile.sample(q, 2000, 3).angularError();
            String name = q.getClass().getSimpleName() + " b=" + q.bits();
            switch (q.bits()) {
                case 1 -> assertThat(name, c, closeTo(0.404, 0.02));
                case 2 -> assertThat(name, c, closeTo(0.125, 0.02));
                case 4 -> assertThat(name, c, closeTo(0.010, 0.004));
                default -> throw new AssertionError("unexpected bits " + q.bits());
            }
        }
    }

    public void testAshStoresTheNormAsGain() throws IOException {
        // with the norm as gain the component along the direction is the cosine itself
        for (int bits : new int[] { 1, 2, 4 }) {
            AshDistortionRateQuantizer q = AshDistortionRateQuantizer.random(DIM, bits, 4, 5);
            CodebookProfile profile = CodebookProfile.sample(q, 500, 2);
            for (int i = 0; i < profile.size(); i++) {
                double a = profile.a()[i];
                double b = profile.b()[i];
                assertThat(a * a + b * b, closeTo(1.0, 1e-3)); // the reconstruction has the norm of the residual
            }
        }
    }

    public void testOsqGainFollowsTheClosedForm() throws IOException {
        // after a rotation OSQ's gain is C / (lambda + (1 - lambda) C^2) times the norm, so A = C^2 / (lambda + (1 - lambda) C^2)
        // (the interval is the optimum for the code, which is exactly the case at one bit)
        for (float lambda : new float[] { 0.1f, 1.0f }) {
            OsqDistortionRateQuantizer q = osq(DIM, 1, lambda, RotationKind.HAAR, haar(DIM, 9));
            Random random = new Random(4);
            float[][] residuals = new float[1500][];
            for (int i = 0; i < residuals.length; i++) {
                residuals[i] = q.sampleResidual(random);
            }
            double meanA = 0;
            double meanClosedForm = 0;
            DistortionRateQuantizer.Reconstruction[] reconstructions = q.reconstruct(residualsOf(residuals));
            for (DistortionRateQuantizer.Reconstruction r : reconstructions) {
                double c = r.cosine();
                meanA += r.a();
                meanClosedForm += c * c / (lambda + (1 - lambda) * c * c);
            }
            assertThat("lambda " + lambda, meanA / reconstructions.length, closeTo(meanClosedForm / reconstructions.length, 0.015));
        }
        // lambda 1 shrinks the reconstruction by cos^2 = 2/pi, lambda 0.1 nearly restores the norm
        assertThat(meanA(1.0f), closeTo(2 / Math.PI, 0.02));
        assertThat(meanA(0.1f), greaterThan(0.9));
        assertThat(meanA(0.1f), lessThan(1.0));
    }

    private static double meanA(float lambda) throws IOException {
        return mean(CodebookProfile.sample(osq(DIM, 1, lambda, RotationKind.HAAR, haar(DIM, 9)), 1000, 6).a());
    }

    private static double mean(float[] v) {
        double s = 0;
        for (float x : v) {
            s += x;
        }
        return s / v.length;
    }

    public void testIsotropyCheckFlagsAWeakRotationAndPassesAStrongOne() throws IOException {
        // residuals whose energy is concentrated in the first coordinates; the dimension is large enough for a dense random
        // rotation to spread it (with only a handful of effective dimensions no single rotation would)
        int dim = 512;
        float[][] residuals = anisotropicResiduals(2000, dim, 21);
        OsqDistortionRateQuantizer none = osq(dim, 1, 0.1f, RotationKind.NONE, identity());
        IsotropyCheck.Report weak = IsotropyCheck.check(none, residualsOf(residuals), CodebookProfile.sample(none, 2000, 1));
        assertTrue("gap " + weak.gap(), weak.weakRotation());
        assertThat(weak.gap(), greaterThan(IsotropyCheck.WEAK_ROTATION_THRESHOLD));
        // a dense random rotation, and ASH's random projection, spread the energy
        OsqDistortionRateQuantizer rotated = osq(dim, 1, 0.1f, RotationKind.HAAR, haar(dim, 13));
        IsotropyCheck.Report strong = IsotropyCheck.check(rotated, residualsOf(residuals), CodebookProfile.sample(rotated, 2000, 1));
        assertFalse("gap " + strong.gap(), strong.weakRotation());
        AshDistortionRateQuantizer ash = AshDistortionRateQuantizer.random(dim, 1, 4, 17);
        IsotropyCheck.Report ashReport = IsotropyCheck.check(ash, residualsOf(residuals), CodebookProfile.sample(ash, 2000, 1));
        assertFalse("gap " + ashReport.gap(), ashReport.weakRotation());
    }
}
