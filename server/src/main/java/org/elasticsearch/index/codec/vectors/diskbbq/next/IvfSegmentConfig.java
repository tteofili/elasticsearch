/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq.next;

import org.elasticsearch.index.codec.vectors.diskbbq.next.ash.AsymmetricHashingQuantizer;

/**
 * Per-segment (per-field) IVF configuration persisted in {@code mivf}. It has three
 * parts: {@link #quantEncoding()} for scalar quant used when indexing doc vectors,
 * {@link #usePrecondition()} for whether a preconditioner is written and used on flush/merge and on the
 * reader, and {@link #rescoreOversample()} for kNN rescore candidate expansion,
 * read with query.
 * The effective config from flush/merge is written to stay consistent with the quantization and
 * preconditioning data stored for the segment.
 * Search-time scoring for quant and preconditioning continues to follow the on-disk {@code mivf} and
 * reader.
 * When the stored rescore is not finite (e.g. {@code NaN}), query and mapping rescore then apply in the usual order.
 * <p>
 * For ASH-encoded segments, the preconditioner slot stores the learned projection matrix W,
 * and the ASH-specific configuration (projectedDimsFraction, bitsPerDim, method) controls encoding behavior.
 */
public record IvfSegmentConfig(
    ESNextDiskBBQVectorsFormat.QuantEncoding quantEncoding,
    boolean usePrecondition,
    float rescoreOversample,
    float ashProjectedDimsFraction,
    int ashBitsPerDim,
    AsymmetricHashingQuantizer.Method ashMethod,
    int ashTrainingIterations,
    int ashTrainingFactor,
    int ashNumClusters,
    int ashKMeansMaxIterations,
    long ashSeed
) {

    /**
     * Default fraction of original dimensions to project to.
     * With 768d input and fraction 0.5: projectedDims = 384 -> 48 bytes per bit plane (SIMD aligned).
     */
    public static final float DEFAULT_ASH_PROJECTED_DIMS_FRACTION = 0.5f;
    /** Default ASH bits per projected dimension. */
    public static final int DEFAULT_ASH_BITS_PER_DIM = 2;
    /** Default ASH training iterations. */
    public static final int DEFAULT_ASH_TRAINING_ITERATIONS = 10;
    /** Default ASH training factor (subsample multiplier for training set). */
    public static final int DEFAULT_ASH_TRAINING_FACTOR = 10;
    /** Default number of ASH centering clusters (independent of IVF cluster count). */
    public static final int DEFAULT_ASH_NUM_CLUSTERS = 16;
    /** Default max iterations for ASH k-means. */
    public static final int DEFAULT_ASH_KMEANS_MAX_ITERATIONS = 50;
    /** Default random seed for ASH training reproducibility. */
    public static final long DEFAULT_ASH_SEED = 42L;

    /**
     * Computes the number of projected dimensions for ASH given the original vector dimension.
     */
    public int ashProjectedDims(int originalDim) {
        return (int) (originalDim * ashProjectedDimsFraction);
    }

    public static IvfSegmentConfig fromCodecDefaults(ESNextDiskBBQVectorsFormat.QuantEncoding quantEncoding, boolean doPrecondition) {
        return new IvfSegmentConfig(
            quantEncoding,
            doPrecondition,
            Float.NaN,
            DEFAULT_ASH_PROJECTED_DIMS_FRACTION,
            DEFAULT_ASH_BITS_PER_DIM,
            AsymmetricHashingQuantizer.Method.LEARNED,
            DEFAULT_ASH_TRAINING_ITERATIONS,
            DEFAULT_ASH_TRAINING_FACTOR,
            DEFAULT_ASH_NUM_CLUSTERS,
            DEFAULT_ASH_KMEANS_MAX_ITERATIONS,
            DEFAULT_ASH_SEED
        );
    }

    /**
     * Creates a non-ASH config with a custom rescore oversample.
     */
    public static IvfSegmentConfig withRescore(
        ESNextDiskBBQVectorsFormat.QuantEncoding quantEncoding,
        boolean doPrecondition,
        float rescoreOversample
    ) {
        return new IvfSegmentConfig(
            quantEncoding,
            doPrecondition,
            rescoreOversample,
            DEFAULT_ASH_PROJECTED_DIMS_FRACTION,
            DEFAULT_ASH_BITS_PER_DIM,
            AsymmetricHashingQuantizer.Method.LEARNED,
            DEFAULT_ASH_TRAINING_ITERATIONS,
            DEFAULT_ASH_TRAINING_FACTOR,
            DEFAULT_ASH_NUM_CLUSTERS,
            DEFAULT_ASH_KMEANS_MAX_ITERATIONS,
            DEFAULT_ASH_SEED
        );
    }

    /**
     * Creates a config with explicit ASH parameters.
     */
    public static IvfSegmentConfig withAsh(
        ESNextDiskBBQVectorsFormat.QuantEncoding quantEncoding,
        float rescoreOversample,
        float ashProjectedDimsFraction,
        int ashBitsPerDim,
        AsymmetricHashingQuantizer.Method ashMethod,
        int ashTrainingIterations
    ) {
        return new IvfSegmentConfig(
            quantEncoding,
            false,
            rescoreOversample,
            ashProjectedDimsFraction,
            ashBitsPerDim,
            ashMethod,
            ashTrainingIterations,
            DEFAULT_ASH_TRAINING_FACTOR,
            DEFAULT_ASH_NUM_CLUSTERS,
            DEFAULT_ASH_KMEANS_MAX_ITERATIONS,
            DEFAULT_ASH_SEED
        );
    }

    /**
     * Whether this config uses ASH encoding.
     */
    public boolean isAsh() {
        return quantEncoding.isAsymmetricHashing();
    }
}
