/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq.calibrate;

/**
 * How residuals are rotated before quantization. The distortion-rate law is exact for a Haar-random rotation; the
 * weaker the rotation, the less the quantizer sees uniformly random directions, see {@link IsotropyCheck}.
 */
public enum RotationKind {
    /** The residuals are quantized as they are. */
    NONE,
    /** A random rotation within blocks of coordinates, with the coordinates permuted, as the preconditioner does. */
    BLOCK_DIAGONAL,
    /** A dense random rotation, for which the law is exact. */
    HAAR,
    /** A trained projection that starts from a random rotation, as ASH learns its projection matrix. */
    LEARNED
}
