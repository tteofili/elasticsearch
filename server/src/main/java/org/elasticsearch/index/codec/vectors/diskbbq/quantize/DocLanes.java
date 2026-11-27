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
 * What a scheme stores beside the code of each vector, in the form that is the same for every scheme: the reconstruction
 * of a vector from its levels is {@code lower + scale * level} in every coordinate, and the scheme's scoring needs the
 * sum of the levels and one more number. A scheme maps its own corrections on to these, and back at scoring time.
 * <p>
 * A mutable holder, to be filled by {@link DocEncoder#read} without allocating for each vector.
 */
public final class DocLanes {
    /** The size of a step between two levels. */
    public float scale;
    /** The value that level 0 stands for. */
    public float lower;
    /** The sum of the levels, which scorers need to correct for the offset of the levels. */
    public int sum;
    /** The scheme's additional correction, which depends on the similarity. */
    public float offset;
}
