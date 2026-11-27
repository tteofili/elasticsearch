/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq.quantize;

/**
 * Reconstructs, in the latent space and without its centroid, a vector from the levels and lanes that {@link DocEncoder}
 * made of it. What a scorer computes with a code is an inner product with this reconstruction, so it is also what the
 * distortion of a scheme is measured on.
 */
public interface DocDecoder {

    /** Writes into {@code latent}, of the latent dimension, the reconstruction of the vector that has this code. */
    void decode(int[] levels, DocLanes lanes, float[] latent);
}
