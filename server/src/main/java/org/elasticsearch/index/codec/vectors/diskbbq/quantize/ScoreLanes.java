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
 * The corrections of a vector in the form that scoring takes them, which is the same for every scheme, see
 * {@link CanonicalScorer}: {@code s} multiplies what depends on the code and the query, {@code sigma} is the sum of the
 * levels, {@code l} multiplies a per-query scalar, and {@code o} is added as it is. A scheme maps what it stores on to these
 * ({@link DocEncoder#scoreLanes}), and what a similarity needs of a vector goes into {@code o}.
 * <p>
 * A mutable holder, to be filled without allocating for each vector.
 */
public final class ScoreLanes {
    public float s;
    public float sigma;
    public float l;
    public float o;
}
