/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.simdvec.internal.vectorization;

import org.apache.lucene.store.IndexInput;

import java.lang.foreign.MemorySegment;

/** Native scorer for E8 root codes (see {@link org.elasticsearch.simdvec.E8Roots}), which are only scored in bulk. */
final class NativeE8Scorer extends NativeMemorySegmentScorer {

    NativeE8Scorer(IndexInput in, int dimensions, int dataLength, int bulkSize) {
        super(in, dimensions, dataLength, bulkSize);
    }

    @Override
    long dotProduct(MemorySegment dataset, MemorySegment query, int length) {
        throw new UnsupportedOperationException("E8 codes are transposed within a bulk and can only be scored in bulk");
    }

    @Override
    void dotProductBulk(MemorySegment dataset, MemorySegment query, int length, int count, MemorySegment scores) {
        DISTANCE_FUNCS.dotProductE8Bulk(dataset, query, length, count, scores);
    }

    @Override
    void dotProductBulkMasked(
        MemorySegment dataset,
        MemorySegment query,
        int dataLength,
        int dataStride,
        int mask,
        int count,
        MemorySegment scores
    ) {
        // codes are interleaved, scoring the whole bulk is as cheap as skipping; unselected scores are discarded by the caller
        DISTANCE_FUNCS.dotProductE8Bulk(dataset, query, dataLength, count, scores);
    }

    @Override
    float queryBitScale() {
        return bitScale(8);
    }

    @Override
    float indexBitScale() {
        return bitScale(1);
    }
}
