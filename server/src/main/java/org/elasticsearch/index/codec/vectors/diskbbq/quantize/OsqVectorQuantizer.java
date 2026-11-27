/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq.quantize;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.elasticsearch.core.Nullable;
import org.elasticsearch.index.codec.vectors.OptimizedScalarQuantizer;
import org.elasticsearch.simdvec.ESVectorUtil;

/**
 * What a segment writer does to quantize one vector of a posting list with OSQ: quantize it against the centroid of the
 * list and, when the centroids are clustered in two levels, make the additional correction relative to the parent
 * centroid, which is what scoring a list through its parent needs.
 * <p>
 * It is a single-vector, copy-free form of {@link OsqDocEncoder}, for the writers' loop over the vectors of a segment:
 * the levels go to the caller's array, in which the caller packs them, and nothing is kept between calls. Float and byte
 * vectors are quantized without widening the bytes.
 * <p>
 * Not thread safe: it holds scratch.
 */
public final class OsqVectorQuantizer {

    private final OptimizedScalarQuantizer quantizer;
    private final VectorSimilarityFunction similarity;
    private final byte bits;
    private final float[] scratch;

    public OsqVectorQuantizer(VectorSimilarityFunction similarity, int dimension, int bits) {
        this(new OptimizedScalarQuantizer(similarity), similarity, dimension, bits);
    }

    public OsqVectorQuantizer(OptimizedScalarQuantizer quantizer, VectorSimilarityFunction similarity, int dimension, int bits) {
        this.quantizer = quantizer;
        this.similarity = similarity;
        this.bits = (byte) bits;
        this.scratch = new float[dimension];
    }

    /**
     * Quantizes {@code vector} against {@code centroid}.
     *
     * @param parentCentroid the centroid of the list's parent, or {@code null} when the centroids are not in two levels
     * @param levels receives the level of each coordinate; at least as long as the vector, and padded with what the caller
     *               left in the rest of it
     */
    public OptimizedScalarQuantizer.QuantizationResult quantize(
        float[] vector,
        float[] centroid,
        @Nullable float[] parentCentroid,
        int[] levels
    ) {
        OptimizedScalarQuantizer.QuantizationResult result = quantizer.scalarQuantize(vector, scratch, levels, bits, centroid);
        if (parentCentroid == null) {
            return result;
        }
        float additionalCorrection = switch (similarity) {
            case EUCLIDEAN -> ESVectorUtil.squareDistance(vector, parentCentroid);
            case DOT_PRODUCT, MAXIMUM_INNER_PRODUCT -> ESVectorUtil.dotProduct(scratch, parentCentroid);
            default -> throw new AssertionError(similarity);
        };
        return withAdditionalCorrection(result, additionalCorrection);
    }

    /** As {@link #quantize(float[], float[], float[], int[])}, for a byte vector against a byte centroid. */
    public OptimizedScalarQuantizer.QuantizationResult quantize(
        byte[] vector,
        byte[] centroid,
        @Nullable float[] parentCentroid,
        int[] levels
    ) {
        OptimizedScalarQuantizer.QuantizationResult result = quantizer.scalarQuantize(vector, scratch, levels, bits, centroid);
        if (parentCentroid == null) {
            return result;
        }
        float additionalCorrection = switch (similarity) {
            case EUCLIDEAN -> ESVectorUtil.squareDistance(vector, parentCentroid);
            case DOT_PRODUCT, MAXIMUM_INNER_PRODUCT -> ESVectorUtil.dotProduct(scratch, parentCentroid);
            default -> throw new AssertionError(similarity);
        };
        return withAdditionalCorrection(result, additionalCorrection);
    }

    private static OptimizedScalarQuantizer.QuantizationResult withAdditionalCorrection(
        OptimizedScalarQuantizer.QuantizationResult result,
        float additionalCorrection
    ) {
        return new OptimizedScalarQuantizer.QuantizationResult(
            result.lowerInterval(),
            result.upperInterval(),
            additionalCorrection,
            result.quantizedComponentSum()
        );
    }
}
