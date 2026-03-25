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
     * Default oversample used when the segment is too small for calibration.
     */
    float DEFAULT_CALIBRATED_OVERSAMPLE = 3f;

    /**
     * No calibrated oversample; indicates the segment has no calibration-derived oversample.
     */
    float NO_CALIBRATED_OVERSAMPLE = 1.5f;

    /**
     * Bundles the quantization encoding with the calibration-derived oversample ratio and
     * preconditioning decision.
     *
     * @param encoding  the quantization encoding to use
     * @param oversample the calibrated oversample ratio ({@code num/den} from the rerank ratio),
     *                   or {@link #NO_CALIBRATED_OVERSAMPLE} if not calibrated
     * @param doPrecondition whether the calibration determined that preconditioning improves recall
     */
    record CalibrationResult(ESNextDiskBBQVectorsFormat.QuantEncoding encoding, float oversample, boolean doPrecondition) {}

    /**
     * Choose the quantization encoding, oversample, and whether to precondtion for the current segment.
     *
     * @param fieldInfo          field metadata (dimension, similarity)
     * @param floatVectorValues merged or flush vectors
     * @param centroidSupplier  centroids for the segment
     * @param assignments       centroid assignment per vector
     * @param overspillAssignments overspill assignments, or empty if none
     * @param mergeState        non-null when merging, null on flush
     * @return calibration result containing the encoding and oversample (never null)
     */
    CalibrationResult select(
        FieldInfo fieldInfo,
        FloatVectorValues floatVectorValues,
        CentroidSupplier centroidSupplier,
        int[] assignments,
        int[] overspillAssignments,
        MergeState mergeState
    );
}
