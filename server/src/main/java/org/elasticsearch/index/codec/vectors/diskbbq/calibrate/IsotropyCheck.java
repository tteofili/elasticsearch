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

/**
 * Tells whether the rotation applied before quantization is strong enough for the distortion-rate law, and therefore for
 * {@link RecallEstimator}, to be trusted on a given corpus.
 * <p>
 * The estimator learns how a codebook behaves by quantizing uniformly random directions, which is what a Haar-random
 * rotation would turn every residual into. A weaker rotation (none, or blocks of coordinates) leaves residuals whose
 * energy is concentrated in a few coordinates, the quantizer is less accurate on them, and the estimator over-predicts
 * recall. The check quantizes a few thousand real residuals and compares the codebook's angular error with the one on
 * random directions; the relative difference is the {@link Report#gap() gap}.
 * <p>
 * {@link #WEAK_ROTATION_THRESHOLD} was calibrated on 17 embedding, image and pixel datasets (208 to 2048 dimensions), for
 * OSQ with lambda 0.1 at 1 and 2 bits under no rotation, blocks of 8, 32 and 128 coordinates, and a dense random
 * rotation, against the excess recall error of the estimator over its own error under the dense rotation of the same
 * dataset. At or below the threshold no case in 96 had an excess above 3 points of recall (the largest, 2.5, is within the
 * measurement noise); above it the excess grows from a few points to tens of points. At the threshold no failure was
 * missed and about one case in seven that was flagged was harmless. A threshold fitted to the data instead of fixed with
 * a margin missed failures when a dataset was left out, so it is fixed. It does not cover the estimator's own error when
 * the rotation is fine, which was up to about 2.5 points of recall, and more at low recall.
 */
public final class IsotropyCheck {

    /** Relative difference of the angular error, above which the rotation is considered too weak. */
    public static final double WEAK_ROTATION_THRESHOLD = 0.03;

    /** Enough residuals for the measured angular error to be stable to about a tenth of the threshold. */
    public static final int RECOMMENDED_SAMPLE_SIZE = 2000;

    private IsotropyCheck() {}

    /**
     * @param isotropicError the codebook's angular error on random directions
     * @param measuredError the codebook's angular error on the real residuals
     * @param gap {@code (measured - isotropic) / isotropic}
     * @param weakRotation whether {@code |gap|} exceeds {@link #WEAK_ROTATION_THRESHOLD}: the estimator is not to be trusted
     */
    public record Report(double isotropicError, double measuredError, double gap, boolean weakRotation) {}

    /**
     * @param quantizer the quantizer, with the rotation it will be used with
     * @param residuals real residuals of the corpus, {@link #RECOMMENDED_SAMPLE_SIZE} or more; see {@link ClusteredSample#residuals}
     * @param isotropic the codebook sampled on random directions, see {@link CodebookProfile#sample}
     */
    public static Report check(DistortionRateQuantizer quantizer, DistortionRateQuantizer.Residuals residuals, CodebookProfile isotropic)
        throws IOException {
        CodebookProfile measured = CodebookProfile.measure(quantizer, residuals);
        double gap = (measured.angularError() - isotropic.angularError()) / isotropic.angularError();
        return new Report(isotropic.angularError(), measured.angularError(), gap, Math.abs(gap) > WEAK_ROTATION_THRESHOLD);
    }
}
