/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.search.vectors;

import org.elasticsearch.common.io.stream.BytesStreamOutput;
import org.elasticsearch.common.io.stream.StreamInput;
import org.elasticsearch.test.ESTestCase;

import java.io.IOException;
import java.util.List;

public class IVFProfileTests extends ESTestCase {

    public void testIVFSegmentProfileRoundTrip() throws IOException {
        IVFSegmentProfile original = new IVFSegmentProfile(
            1000,
            64,
            5,
            120,
            0.5f,
            0.95f,
            4.0,
            5L,
            3.5,
            10,
            40,
            120L,
            5L,
            3500L
        );
        BytesStreamOutput out = new BytesStreamOutput();
        original.writeTo(out);
        StreamInput in = out.bytes().streamInput();
        IVFSegmentProfile read = new IVFSegmentProfile(in);
        assertEquals(original.numVectors(), read.numVectors());
        assertEquals(original.numCentroids(), read.numCentroids());
        assertEquals(original.clustersVisited(), read.clustersVisited());
        assertEquals(original.vectorsVisited(), read.vectorsVisited());
        assertEquals(original.centroidScoreMin(), read.centroidScoreMin(), 0f);
        assertEquals(original.centroidScoreMax(), read.centroidScoreMax(), 0f);
        assertEquals(original.centroidScoreMean(), read.centroidScoreMean(), 0.001f);
        assertEquals(original.clusterSizeMin(), read.clusterSizeMin());
        assertEquals(original.clusterSizeMax(), read.clusterSizeMax());
    }

    public void testIVFProfileRoundTrip() throws IOException {
        IVFSegmentProfile seg = new IVFSegmentProfile(
            500,
            32,
            3,
            45,
            0.6f,
            0.9f,
            2.1,
            3L,
            1.5,
            12,
            18,
            45L,
            3L,
            700L
        );
        IVFProfile original = new IVFProfile(
            45L,
            3L,
            List.of(seg),
            0.6f,
            0.9f,
            0.7f,
            0.15f,
            12,
            18,
            15f,
            3f
        );
        BytesStreamOutput out = new BytesStreamOutput();
        original.writeTo(out);
        StreamInput in = out.bytes().streamInput();
        IVFProfile read = new IVFProfile(in);
        assertEquals(original.getTotalVectorsVisited(), read.getTotalVectorsVisited());
        assertEquals(original.getTotalClustersVisited(), read.getTotalClustersVisited());
        assertEquals(original.getSegments().size(), read.getSegments().size());
        assertEquals(original.getCentroidScoreMin(), read.getCentroidScoreMin(), 0f);
        assertEquals(original.getCentroidScoreMax(), read.getCentroidScoreMax(), 0f);
        assertEquals(original.getClusterSizeMin(), read.getClusterSizeMin());
        assertEquals(original.getClusterSizeMax(), read.getClusterSizeMax());
    }
}
