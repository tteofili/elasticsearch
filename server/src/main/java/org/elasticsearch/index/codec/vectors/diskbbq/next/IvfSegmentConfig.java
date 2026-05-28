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
 * and the ASH-specific configuration (totalBits, bitsPerDim, method) controls encoding behavior.
 */
public record IvfSegmentConfig(
    ESNextDiskBBQVectorsFormat.QuantEncoding quantEncoding,
    boolean usePrecondition,
    float rescoreOversample,
    int ashTotalBits,
    int ashBitsPerDim,
    AsymmetricHashingQuantizer.Method ashMethod,
    int ashTrainingIterations
) {

    /** Default ASH total bits budget. */
    public static final int DEFAULT_ASH_TOTAL_BITS = 384;
    /** Default ASH bits per projected dimension. */
    public static final int DEFAULT_ASH_BITS_PER_DIM = 1;
    /** Default ASH training iterations. */
    public static final int DEFAULT_ASH_TRAINING_ITERATIONS = 20;
    /** Default ASH training factor. */
    public static final int DEFAULT_ASH_TRAINING_FACTOR = 10;

    public static IvfSegmentConfig fromCodecDefaults(ESNextDiskBBQVectorsFormat.QuantEncoding quantEncoding, boolean doPrecondition) {
        return new IvfSegmentConfig(
            quantEncoding,
            doPrecondition,
            Float.NaN,
            DEFAULT_ASH_TOTAL_BITS,
            DEFAULT_ASH_BITS_PER_DIM,
            AsymmetricHashingQuantizer.Method.LEARNED,
            DEFAULT_ASH_TRAINING_ITERATIONS
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
            DEFAULT_ASH_TOTAL_BITS,
            DEFAULT_ASH_BITS_PER_DIM,
            AsymmetricHashingQuantizer.Method.LEARNED,
            DEFAULT_ASH_TRAINING_ITERATIONS
        );
    }

    /**
     * Creates a config with explicit ASH parameters.
     */
    public static IvfSegmentConfig withAsh(
        ESNextDiskBBQVectorsFormat.QuantEncoding quantEncoding,
        float rescoreOversample,
        int ashTotalBits,
        int ashBitsPerDim,
        AsymmetricHashingQuantizer.Method ashMethod,
        int ashTrainingIterations
    ) {
        return new IvfSegmentConfig(quantEncoding, false, rescoreOversample, ashTotalBits, ashBitsPerDim, ashMethod, ashTrainingIterations);
    }

    /**
     * Whether this config uses ASH encoding.
     */
    public boolean isAsh() {
        return quantEncoding.isAsymmetricHashing();
    }
}
