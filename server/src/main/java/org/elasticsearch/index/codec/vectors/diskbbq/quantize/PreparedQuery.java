/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq.quantize;

import java.io.IOException;

/**
 * A query that is ready to be scored against the posting lists of a segment, see {@link QueryEncoder}. It is stateful: it
 * follows the posting list that is being visited, and is not thread safe.
 */
public interface PreparedQuery {

    /** Makes the query ready for the posting list of the centroid with this ordinal, as the visitor of a list does. */
    void forList(int centroidOrdinal) throws IOException;

    /** The scalars that the current list needs of the query, see {@link CanonicalScorer}. */
    QueryScalars scalars() throws IOException;

    /**
     * The raw dot product of the code of a vector, given by its levels, and the query's code, which is what the scheme's
     * kernel computes with the packed bytes, for the current list.
     */
    float rawDot(int[] levels) throws IOException;

    /**
     * Writes into {@code latent} the query as it should be scored against the current list, exactly, in the space the scorer
     * works in.
     */
    void exactLatent(float[] latent) throws IOException;

    /**
     * Writes into {@code latent} what the scorer ends up using in place of {@link #exactLatent}, the quantized query
     * brought back to the same scale and space. The distance between the two is the error that quantizing the query adds
     * to a score.
     */
    void servedLatent(float[] latent) throws IOException;
}
