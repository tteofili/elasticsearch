/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq.calibrate;

import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FloatVectorValues;
import org.elasticsearch.index.codec.vectors.diskbbq.CentroidSupplier;
import org.elasticsearch.simdvec.ESVectorUtil;

import java.io.IOException;

/**
 * A clustered sample of a corpus, for quantizers that learn something from the data before they quantize it, as ASH
 * learns its projection, and for estimating recall.
 * <p>
 * It is a view: the vectors are read from {@code vectors} (which may be off-heap) by ordinal when they are needed, and
 * the centroids from a {@link CentroidSupplier}, so that no sample is ever copied into an array of arrays. The arrays it
 * does hold are one {@code int} per sampled vector.
 * <p>
 * The arrays returned by {@link #vector} and {@link #centroid} may be reused by the next read of the same kind.
 *
 * @param vectors the corpus
 * @param ordinals the ordinals in {@code vectors} of the sampled vectors, in sample order
 * @param centroids the landmarks
 * @param assignment for each sampled vector the ordinal in {@code centroids} of its landmark
 */
public record ClusteredSample(FloatVectorValues vectors, int[] ordinals, CentroidSupplier centroids, int[] assignment) {

    public ClusteredSample {
        if (ordinals.length != assignment.length) {
            throw new IllegalArgumentException("one assignment per sampled vector: " + ordinals.length + " != " + assignment.length);
        }
    }

    /**
     * A sample of byte vectors, which are read as the floats they hold and kept at one byte per dimension until then.
     * The centroids may be byte-backed too, see {@link CentroidSupplier#fromByteArray}.
     */
    public static ClusteredSample ofBytes(ByteVectorValues vectors, int[] ordinals, CentroidSupplier centroids, int[] assignment) {
        return new ClusteredSample(new WidenedByteVectorValues(vectors), ordinals, centroids, assignment);
    }

    public int size() {
        return ordinals.length;
    }

    public int dimension() {
        return vectors.dimension();
    }

    public float[] vector(int i) throws IOException {
        return vectors.vectorValue(ordinals[i]);
    }

    public float[] centroid(int i) throws IOException {
        return centroids.centroid(assignment[i]);
    }

    /** The residuals (vector minus landmark) of the first {@code limit} sampled vectors, computed as they are read. */
    public DistortionRateQuantizer.Residuals residuals(int limit) {
        int n = Math.min(limit, size());
        float[] scratch = new float[dimension()];
        return new DistortionRateQuantizer.Residuals() {
            @Override
            public int size() {
                return n;
            }

            @Override
            public float[] residual(int i) throws IOException {
                System.arraycopy(vector(i), 0, scratch, 0, scratch.length);
                ESVectorUtil.linearCombination(-1f, centroid(i), 1f, scratch);
                return scratch;
            }
        };
    }
}
