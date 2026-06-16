/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq.calibrate;

import org.elasticsearch.simdvec.ESVectorUtil;

/**
 * Small calibration helpers introduced with the first consumer ({@link ErrorModel}).
 * Additional merge-time utilities are added in the PR that wires {@code IvfAutoCalibration}.
 */
public final class CalibrationUtils {

    private CalibrationUtils() {}

    /**
     * Copy {@code src} into {@code scratch} and L2-normalize in place.
     */
    public static float[] copyAndNormalize(float[] src, float[] scratch) {
        System.arraycopy(src, 0, scratch, 0, src.length);
        ESVectorUtil.l2Normalize(scratch, src.length);
        return scratch;
    }
}
