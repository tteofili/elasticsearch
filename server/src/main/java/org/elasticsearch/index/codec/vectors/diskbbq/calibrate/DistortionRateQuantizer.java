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
 * A quantizer seen through the distortion-rate law of "Configuring Project-then-Quantize Embedding Compression": a
 * residual (a vector minus its centroid) is projected onto a {@link #latentDimension() latent space}, where the quantizer
 * encodes its direction, and the score of a query against the code is {@code gain * <latent query, reconstruction>}
 * plus the exact term of the centroid.
 * <p>
 * What the law needs of a quantizer is how its reconstruction relates to the direction it quantized, which is all
 * {@link #reconstruct} reports. With that, the codebook can be characterized without encoding a corpus
 * ({@link CodebookProfile}), the rotation can be checked ({@link IsotropyCheck}) and recall can be estimated
 * ({@link RecallEstimator}), identically for OSQ, ASH or any other quantizer that implements this interface.
 */
public interface DistortionRateQuantizer {

    /** The dimension of the vectors and residuals the quantizer is given. */
    int dimension();

    /** The dimension of the space the quantizer works in, at most {@link #dimension()}; smaller when it projects. */
    int latentDimension();

    int bits();

    /** How residuals are rotated before they are quantized, which decides whether the law's isotropy assumption holds. */
    RotationKind rotation();

    /**
     * Writes into {@code latent} the coordinates of a vector of the original space in the latent space; their norm is that
     * of the retained part. The buffer is the caller's, so that projecting many vectors allocates nothing.
     */
    void project(float[] vector, float[] latent);

    /** As {@link #project(float[], float[])}, into a new array. */
    default float[] project(float[] vector) {
        float[] latent = new float[latentDimension()];
        project(vector, latent);
        return latent;
    }

    /**
     * A residual whose latent image is a uniformly random unit vector, for characterizing the codebook on random
     * directions.
     */
    float[] sampleResidual(Random random);

    /**
     * Quantizes residuals (vectors minus their centroid, in the original space) and reports for each what the scorer
     * would reconstruct. Residuals of zero norm have no direction and are reported as {@link Reconstruction#EMPTY}.
     * <p>
     * The residuals are pulled one at a time and none is kept past the call that reads it, so an implementation holds
     * at most a block of them, however many the source has. The reconstructions are two numbers per residual.
     */
    Reconstruction[] reconstruct(Residuals residuals) throws IOException;

    /**
     * A sequence of residuals that is produced on demand, typically from vectors that stay off-heap.
     */
    interface Residuals {
        int size();

        /**
         * The {@code i}-th residual. The array may be reused by the next call: a caller that needs it later copies it.
         */
        float[] residual(int i) throws IOException;
    }

    /**
     * The norm, in latent coordinates, of the error made by quantizing a query the way this quantizer's query side
     * does it, for scoring against the posting list of {@code centroid}.
     * <p>
     * Implementations reuse scratch buffers across calls, so a quantizer is not safe to share between threads.
     */
    double queryError(float[] query, float[] centroid);

    /**
     * What the scorer reconstructs for one residual, per unit of the residual's gain, relative to the unit latent
     * direction {@code u} of the residual: {@code a = <u, reconstruction> / gain} and
     * {@code normSq = |reconstruction|^2 / gain^2}. For a stored-norm gain, {@code a} is the cosine between the direction
     * and its code; a quantizer with another gain, such as OSQ's interval fit, reports its own.
     */
    record Reconstruction(double a, double normSq) {
        public static final Reconstruction EMPTY = new Reconstruction(0, 0);

        public boolean isEmpty() {
            return normSq == 0;
        }

        /** The part of the reconstruction orthogonal to the direction, which is what makes scores noisy. */
        public double b() {
            return Math.sqrt(Math.max(0, normSq - a * a));
        }

        /** The cosine between the direction and the reconstruction, whatever the gain. */
        public double cosine() {
            return a / Math.sqrt(normSq);
        }
    }
}
