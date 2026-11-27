/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq.calibrate;

import java.io.IOException;
import java.util.Random;

/**
 * How a quantizer reconstructs directions, as a sample of pairs {@code (A, B)}: {@code A} is the component of the
 * reconstruction along the quantized direction and {@code B} the norm of the rest, per unit of gain. On uniformly
 * random directions this is the codebook's constant of the distortion-rate law (Algorithm 1 of the paper, generalized
 * from the cosine {@code C} to any gain: for a stored norm {@code A = C} and {@code B = sqrt(1 - C^2)}).
 *
 * @param a the sampled {@code A}s
 * @param b the sampled {@code B}s
 * @param meanCosine mean cosine between a direction and its code
 */
public record CodebookProfile(float[] a, float[] b, double meanCosine) {

    public static final int DEFAULT_SAMPLES = 3000;

    /** The codebook's constant {@code c = 2 (1 - E[C])}: the expected squared angular error on a random direction. */
    public double angularError() {
        return 2 * (1 - meanCosine);
    }

    public int size() {
        return a.length;
    }

    /** Characterizes the codebook on uniformly random directions, with no corpus. */
    public static CodebookProfile sample(DistortionRateQuantizer quantizer, int samples, long seed) throws IOException {
        Random random = new Random(seed);
        return of(quantizer.reconstruct(new DistortionRateQuantizer.Residuals() {
            @Override
            public int size() {
                return samples;
            }

            @Override
            public float[] residual(int i) {
                return quantizer.sampleResidual(random);
            }
        }));
    }

    /** Characterizes the codebook on real residuals, which is what {@link IsotropyCheck} compares with random directions. */
    public static CodebookProfile measure(DistortionRateQuantizer quantizer, DistortionRateQuantizer.Residuals residuals)
        throws IOException {
        return of(quantizer.reconstruct(residuals));
    }

    static CodebookProfile of(DistortionRateQuantizer.Reconstruction[] reconstructions) {
        int n = 0;
        for (DistortionRateQuantizer.Reconstruction r : reconstructions) {
            if (r.isEmpty() == false) {
                n++;
            }
        }
        if (n == 0) {
            throw new IllegalArgumentException("no residual with a direction to characterize the codebook on");
        }
        float[] a = new float[n];
        float[] b = new float[n];
        double sum = 0;
        int j = 0;
        for (DistortionRateQuantizer.Reconstruction r : reconstructions) {
            if (r.isEmpty()) {
                continue;
            }
            a[j] = (float) r.a();
            b[j] = (float) r.b();
            sum += r.cosine();
            j++;
        }
        return new CodebookProfile(a, b, sum / n);
    }
}
