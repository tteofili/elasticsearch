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
 * Prepares queries for a scheme: what has to be done to a query, once for a search and again for each posting list that
 * is visited, before its codes can be scored. OSQ quantizes the query against the centroid of each list; ASH projects and
 * quantizes it once and corrects for each list with an exact inner product.
 * <p>
 * The scorers that consume a prepared query are the scheme's own, as they are the performance boundary, so this interface
 * is the lifecycle and the two views that code written for every scheme needs of any query encoder: what the exact query
 * is, and what the scorer ends up using in its place.
 */
public interface QueryEncoder {

    /** The dimension of the space that the scorer works in. */
    int latentDimension();

    /** Prepares a query for a search. */
    PreparedQuery prepare(float[] query) throws IOException;
}
