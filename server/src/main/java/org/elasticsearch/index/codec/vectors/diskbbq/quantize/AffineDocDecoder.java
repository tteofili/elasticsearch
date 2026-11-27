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
 * The reconstruction of every scheme that quantizes each coordinate to one of equally spaced values, such as OSQ and
 * ASH: {@code lower + scale * level}. A scheme whose codebook is not a grid, such as E8, brings its own decoder.
 */
public final class AffineDocDecoder implements DocDecoder {

    public static final AffineDocDecoder INSTANCE = new AffineDocDecoder();

    private AffineDocDecoder() {}

    @Override
    public void decode(int[] levels, DocLanes lanes, float[] latent) {
        for (int d = 0; d < latent.length; d++) {
            latent[d] = lanes.lower + lanes.scale * levels[d];
        }
    }
}
