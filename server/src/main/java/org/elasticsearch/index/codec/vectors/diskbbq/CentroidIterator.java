/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq;

import java.io.IOException;

/**
 * An iterator over centroids that provides posting list metadata.
 */
public interface CentroidIterator {
    boolean hasNext();

    PostingMetadata nextPosting() throws IOException;

    /**
     * Returns the score of the next centroid without advancing the iterator.
     * This can be used for early filtering based on minimum competitive similarity.
     * @return the score of the next centroid, or Float.NEGATIVE_INFINITY if no next centroid
     * @throws IOException if an I/O error occurs
     */
    default float peekNextScore() throws IOException {
        return Float.NEGATIVE_INFINITY;
    }
}
