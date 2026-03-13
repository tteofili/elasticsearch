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
 * Default implementation of {@link AutoQuantizationSelector}.
 * Returns a fixed encoding; can be replaced with a more advanced implementation.
 */
public final class NoOpAutomaticQuantizationSelector implements AutoQuantizationSelector {

    public static final NoOpAutomaticQuantizationSelector INSTANCE = new NoOpAutomaticQuantizationSelector();

    private NoOpAutomaticQuantizationSelector() {}

    @Override
    public CalibrationResult select(
        FieldInfo fieldInfo,
        FloatVectorValues floatVectorValues,
        CentroidSupplier centroidSupplier,
        int[] assignments,
        int[] overspillAssignments,
        MergeState mergeState
    ) {
        return new CalibrationResult(ESNextDiskBBQVectorsFormat.QuantEncoding.ONE_BIT_4BIT_QUERY, NO_CALIBRATED_OVERSAMPLE, false);
    }
}
