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

/** Reads the centroid with an ordinal, for a query encoder that needs it. The array may be reused by the next call. */
@FunctionalInterface
public interface CentroidReader {
    float[] read(int centroidOrdinal) throws IOException;
}
