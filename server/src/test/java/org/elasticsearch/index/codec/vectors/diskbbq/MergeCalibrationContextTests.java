/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq;

import org.apache.lucene.codecs.Codec;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.store.Directory;
import org.apache.lucene.util.StringHelper;
import org.apache.lucene.util.Version;
import org.elasticsearch.test.ESTestCase;

import java.io.IOException;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.is;

public class MergeCalibrationContextTests extends ESTestCase {

    public void testFromBackgroundMerge() throws IOException {
        try (Directory dir = newDirectory()) {
            SegmentInfo segmentInfo = new SegmentInfo(
                dir,
                Version.LATEST,
                Version.LATEST,
                "bg",
                100,
                false,
                false,
                Codec.getDefault(),
                Collections.emptyMap(),
                StringHelper.randomId(),
                new HashMap<>(),
                null
            );
            MergeState mergeState = new MergeState(
                null,
                segmentInfo,
                null,
                null,
                null,
                null,
                null,
                null,
                null,
                null,
                null,
                new org.apache.lucene.codecs.KnnVectorsReader[0],
                null,
                null,
                null,
                false
            );

            MergeCalibrationContext ctx = MergeCalibrationContext.from(mergeState);
            assertThat(ctx.inputSegments(), equalTo(0));
            assertFalse(ctx.boundedForceMerge());
            assertThat(ctx.mergeKind(), equalTo("background"));
            assertThat(ctx.mergeMaxNumSegmentsForLog(), equalTo("n/a"));
        }
    }

    public void testFromForceMergeDiagnostic() throws IOException {
        try (Directory dir = newDirectory()) {
            SegmentInfo segmentInfo = new SegmentInfo(
                dir,
                Version.LATEST,
                Version.LATEST,
                "fm",
                100,
                false,
                false,
                Codec.getDefault(),
                Collections.emptyMap(),
                StringHelper.randomId(),
                new HashMap<>(),
                null
            );
            segmentInfo.addDiagnostics(Map.of("mergeMaxNumSegments", "1"));
            MergeState mergeState = new MergeState(
                null,
                segmentInfo,
                null,
                null,
                null,
                null,
                null,
                null,
                null,
                null,
                null,
                new org.apache.lucene.codecs.KnnVectorsReader[2],
                null,
                null,
                null,
                false
            );

            MergeCalibrationContext ctx = MergeCalibrationContext.from(mergeState);
            assertThat(ctx.inputSegments(), equalTo(2));
            assertTrue(ctx.boundedForceMerge());
            assertThat(ctx.mergeKind(), equalTo("force"));
            assertThat(ctx.mergeMaxNumSegments(), is(1));
        }
    }
}
