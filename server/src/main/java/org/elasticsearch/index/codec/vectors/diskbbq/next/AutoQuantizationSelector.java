/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq.next;

import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.MergeState;
import org.elasticsearch.index.codec.vectors.diskbbq.CentroidSupplier;

/**
 * Selects a concrete {@link ESNextDiskBBQVectorsFormat.QuantEncoding} at segment write time
 * (flush or merge) for IVF/diskbbq when automatic quantization is enabled.
 * The implementation can be replaced for more advanced selection logic.
 */
@FunctionalInterface
public interface AutoQuantizationSelector {

    /**
     * Choose the quantization encoding to use for the current segment.
     *
     * @param fieldInfo          field metadata (dimension, similarity)
     * @param floatVectorValues merged or flush vectors
     * @param centroidSupplier  centroids for the segment
     * @param assignments       centroid assignment per vector
     * @param overspillAssignments overspill assignments, or empty if none
     * @param mergeState        non-null when merging, null on flush
     * @return a concrete encoding to use for this segment (never null)
     */
    ESNextDiskBBQVectorsFormat.QuantEncoding select(
        FieldInfo fieldInfo,
        FloatVectorValues floatVectorValues,
        CentroidSupplier centroidSupplier,
        int[] assignments,
        int[] overspillAssignments,
        MergeState mergeState
    );
}
