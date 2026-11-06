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
import org.elasticsearch.index.codec.vectors.OptimizedScalarQuantizer;
import org.elasticsearch.simdvec.E8Lattice;

import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;

/**
 * Document-side quantizer for {@link QuantEncoding#E8_ROOTS} (experimental), with the call shape of
 * {@link OptimizedScalarQuantizer#scalarQuantize(float[], float[], int[], byte, float[])} so that the DiskBBQ writers can
 * use either. The residual {@code r = x - c} is written to {@code residualDestination}, one root code per block to
 * {@code destination}, and the result carries:
 * <ul>
 *   <li>{@code lowerInterval}: the ratio scale {@code alpha = |r|^2 / <r, r̂>}, so {@code alpha <y, r̂>} estimates
 *       {@code <y, r>} without bias for directions {@code y} independent of the code (RaBitQ's estimator);</li>
 *   <li>{@code upperInterval} and {@code quantizedComponentSum}: unused (zero);</li>
 *   <li>{@code additionalCorrection}: the same value as OSQ, {@code |r|^2} for Euclidean and {@code <x, c>} otherwise
 *       (the writers replace it with the parent-centroid term, exactly as for OSQ).</li>
 * </ul>
 */
public final class E8ResidualQuantizer {

    private final VectorSimilarityFunction similarityFunction;
    private byte[] codes = new byte[0];

    public E8ResidualQuantizer(VectorSimilarityFunction similarityFunction) {
        this.similarityFunction = similarityFunction;
    }

    public OptimizedScalarQuantizer.QuantizationResult quantize(
        float[] vector,
        float[] residualDestination,
        int[] destination,
        float[] centroid
    ) {
        final int dims = vector.length;
        double norm2 = 0, centroidDot = 0;
        for (int i = 0; i < dims; i++) {
            float r = vector[i] - centroid[i];
            residualDestination[i] = r;
            norm2 += (double) r * r;
            centroidDot += (double) vector[i] * centroid[i];
        }
        final int blocks = E8Lattice.blocks(dims);
        if (codes.length < blocks) {
            codes = new byte[blocks];
        }
        float dot = E8Lattice.encode(residualDestination, dims, codes, 0);
        for (int b = 0; b < blocks; b++) {
            destination[b] = codes[b] & 0xFF;
        }
        float alpha = dot > 0f ? (float) (norm2 / dot) : 0f;
        float additional = similarityFunction == EUCLIDEAN ? (float) norm2 : (float) centroidDot;
        return new OptimizedScalarQuantizer.QuantizationResult(alpha, 0f, additional, 0);
    }

    /**
     * Quantizes with E8 for {@link QuantEncoding#E8_ROOTS} and with OSQ otherwise.
     */
    public static OptimizedScalarQuantizer.QuantizationResult quantize(
        QuantEncoding encoding,
        OptimizedScalarQuantizer osq,
        E8ResidualQuantizer e8,
        float[] vector,
        float[] residualDestination,
        int[] destination,
        float[] centroid
    ) {
        if (encoding == QuantEncoding.E8_ROOTS) {
            return e8.quantize(vector, residualDestination, destination, centroid);
        }
        return osq.scalarQuantize(vector, residualDestination, destination, encoding.bits(), centroid);
    }

    /**
     * Experimental, off by default: {@code -Des.vectors.diskbbq.fold_leaf_centroid=true} makes the DiskBBQ writers store
     * a different per-vector correction for dot-product similarities, for every encoding (OSQ and E8), with no reader change.
     * <p>
     * The reader estimates {@code <q, x> = <q, c> + <q - c, r> + <c, r>} (leaf centroid {@code c}, residual {@code r})
     * with the integer part computed against the query re-centred on the parent {@code p}: it returns
     * {@code est(<q - p, r>) + additional + <q, c>}. By default {@code additional = <p, r>}, which estimates
     * {@code <q - p, r>} from the code. Storing {@code additional = <c, r> - <c - p, r̂>} instead (r̂ the code's
     * reconstruction, in the estimator's units) turns the same integer sum into an estimate of {@code <q - c, r>}: the
     * query is effectively re-centred on each document's own leaf centroid, as in faiss's IVF-RaBitQ, at no query-time
     * or storage cost. The estimate's noise grows with the norm of the re-centred query, which is smaller for the leaves
     * near the query, i.e. for the documents that matter.
     */
    public static final boolean FOLD_LEAF_CENTROID = org.elasticsearch.core.Booleans.parseBoolean(
        System.getProperty("es.vectors.diskbbq.fold_leaf_centroid", "false")
    );

    /**
     * The additional correction for dot-product similarities: {@code <p, r>}, or with {@link #FOLD_LEAF_CENTROID}
     * {@code <c, r> - <c - p, r̂>}.
     *
     * @param residual {@code r = x - c}, as left by the quantizer
     * @param quantized the quantized ints of the vector (OSQ levels, or one E8 code per block)
     * @param result the quantizer's result (OSQ interval, or the E8 ratio scale as {@code lowerInterval})
     */
    public static float dotProductCorrection(
        QuantEncoding encoding,
        OptimizedScalarQuantizer.QuantizationResult result,
        int[] quantized,
        float[] residual,
        float[] centroid,
        float[] parentCentroid
    ) {
        final int dims = residual.length;
        if (FOLD_LEAF_CENTROID == false) {
            // exactly what the writers computed before this option existed (same SIMD dot product, same bits)
            return org.elasticsearch.simdvec.ESVectorUtil.dotProduct(residual, parentCentroid);
        }
        double cr = 0, shiftDotRec = 0;
        for (int i = 0; i < dims; i++) {
            cr += (double) centroid[i] * residual[i];
        }
        if (encoding == QuantEncoding.E8_ROOTS) {
            float alpha = result.lowerInterval();
            for (int b = 0; b * E8Lattice.BLOCK < dims; b++) {
                int c = quantized[b] & 0xFF;
                int pair = c & 0x7F;
                float sign = (c & 0x80) != 0 ? -1f : 1f;
                for (int t = 0; t < E8Lattice.BLOCK && b * E8Lattice.BLOCK + t < dims; t++) {
                    int i = b * E8Lattice.BLOCK + t;
                    shiftDotRec += (double) (centroid[i] - parentCentroid[i]) * sign * E8Lattice.representative(pair, t);
                }
            }
            shiftDotRec *= alpha;
        } else {
            float a = result.lowerInterval();
            float step = (result.upperInterval() - a) / ((1 << encoding.bits()) - 1);
            for (int i = 0; i < dims; i++) {
                shiftDotRec += (double) (centroid[i] - parentCentroid[i]) * (a + step * quantized[i]);
            }
        }
        return (float) (cr - shiftDotRec);
    }
}
