/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.VectorUtil;
import org.elasticsearch.index.codec.vectors.OptimizedScalarQuantizer;
import org.elasticsearch.simdvec.E8Roots;
import org.elasticsearch.simdvec.ESVectorUtil;

import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;

/**
 * Codes a centred vector with the roots of the E8 lattice ({@link E8Roots}): one byte per eight dimensions, the
 * same storage as a one-bit code, with an exact constant-time encoder and no training.
 * <p>
 * The corrections reuse the {@link OptimizedScalarQuantizer} layout and score formulas: the lower interval is zero and the
 * upper interval is the ratio scale {@code α = |r|² / <r, r̂>} of the residual {@code r} and its reconstruction
 * {@code r̂}, which makes the estimate {@code <y, r> ≈ α <y, r̂>} exact along {@code r}. The additional correction is the
 * one OSQ stores. Queries become an int8 lookup table, with the table scale carried in the upper interval.
 */
public final class E8RootQuantizer extends OptimizedScalarQuantizer {

    private final VectorSimilarityFunction similarityFunction;

    public E8RootQuantizer(VectorSimilarityFunction similarityFunction) {
        super(similarityFunction);
        this.similarityFunction = similarityFunction;
    }

    @Override
    public QuantizationResult scalarQuantize(float[] vector, float[] residualDestination, int[] destination, byte bits, float[] centroid) {
        assert vector.length == residualDestination.length;
        float centroidDot = 0f;
        for (int i = 0; i < vector.length; i++) {
            residualDestination[i] = vector[i] - centroid[i];
            centroidDot += vector[i] * centroid[i];
        }
        return encode(residualDestination, destination, centroidDot);
    }

    @Override
    public QuantizationResult scalarQuantize(byte[] vector, float[] residualDestination, int[] destination, byte bits, byte[] centroid) {
        assert vector.length == residualDestination.length;
        float centroidDot = 0f;
        for (int i = 0; i < vector.length; i++) {
            residualDestination[i] = vector[i] - centroid[i];
            centroidDot += vector[i] * centroid[i];
        }
        return encode(residualDestination, destination, centroidDot);
    }

    private QuantizationResult encode(float[] residual, int[] destination, float centroidDot) {
        float norm2 = VectorUtil.dotProduct(residual, residual);
        float dot = E8Roots.encode(residual, destination);
        float alpha = dot > 0f ? norm2 / dot : 0f;
        return new QuantizationResult(0f, alpha, similarityFunction == EUCLIDEAN ? norm2 : centroidDot, 0);
    }

    /**
     * Turns the query, centred on {@code centroid}, into the int8 lookup table {@code table}. The scorers read an 8-bit
     * query scale as {@code (upper - lower) / 255}, so the table scale is returned as {@code upper = 255 * scale}.
     */
    static QuantizationResult quantizeQuery(float[] query, float[] centroid, float[] residualScratch, byte[] table) {
        for (int i = 0; i < query.length; i++) {
            residualScratch[i] = query[i] - centroid[i];
        }
        float scale = ESVectorUtil.buildE8QueryTable(residualScratch, table);
        return new QuantizationResult(0f, 255f * scale, 0f, 0);
    }
}
