/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.search.vectors;

import org.elasticsearch.test.ESTestCase;

public class IVFProfileAccumulatorTests extends ESTestCase {

    public void testSingleSegmentBuildProfile() {
        IVFProfileAccumulator accumulator = new IVFProfileAccumulator();
        IVFProfileAccumulator.IVFSegmentBuilder builder = accumulator.createSegmentBuilder();
        builder.recordSegmentStart(1000, 64);
        builder.recordCluster(0.9f, 20);
        builder.recordCluster(0.8f, 15);
        builder.recordCluster(0.7f, 25);
        builder.recordSegmentEnd(60);
        accumulator.finishSegment(builder);

        IVFProfile profile = accumulator.buildProfile();
        assertNotNull(profile);
        assertEquals(60L, profile.getTotalVectorsVisited());
        assertEquals(3L, profile.getTotalClustersVisited());
        assertEquals(1, profile.getSegments().size());
        IVFSegmentProfile seg = profile.getSegments().get(0);
        assertEquals(1000, seg.numVectors());
        assertEquals(64, seg.numCentroids());
        assertEquals(3, seg.clustersVisited());
        assertEquals(60, seg.vectorsVisited());
        assertEquals(0.7f, seg.centroidScoreMin(), 0.001f);
        assertEquals(0.9f, seg.centroidScoreMax(), 0.001f);
        assertEquals(15, seg.clusterSizeMin());
        assertEquals(25, seg.clusterSizeMax());
    }

    public void testMultipleSegmentsBuildProfile() {
        IVFProfileAccumulator accumulator = new IVFProfileAccumulator();
        for (int s = 0; s < 2; s++) {
            IVFProfileAccumulator.IVFSegmentBuilder builder = accumulator.createSegmentBuilder();
            builder.recordSegmentStart(500, 32);
            builder.recordCluster(0.5f + s * 0.2f, 10);
            builder.recordCluster(0.4f + s * 0.2f, 12);
            builder.recordSegmentEnd(22);
            accumulator.finishSegment(builder);
        }

        IVFProfile profile = accumulator.buildProfile();
        assertNotNull(profile);
        assertEquals(44L, profile.getTotalVectorsVisited());
        assertEquals(4L, profile.getTotalClustersVisited());
        assertEquals(2, profile.getSegments().size());
        assertEquals(0.4f, profile.getCentroidScoreMin(), 0.001f);
        assertEquals(0.7f, profile.getCentroidScoreMax(), 0.001f);
        assertEquals(10, profile.getClusterSizeMin());
        assertEquals(12, profile.getClusterSizeMax());
    }

    public void testEmptyAccumulatorReturnsNull() {
        IVFProfileAccumulator accumulator = new IVFProfileAccumulator();
        assertNull(accumulator.buildProfile());
    }

    public void testSegmentBuilderWithNoClusters() {
        IVFProfileAccumulator accumulator = new IVFProfileAccumulator();
        IVFProfileAccumulator.IVFSegmentBuilder builder = accumulator.createSegmentBuilder();
        builder.recordSegmentStart(100, 10);
        builder.recordSegmentEnd(0);
        accumulator.finishSegment(builder);
        IVFProfile profile = accumulator.buildProfile();
        assertNotNull(profile);
        assertEquals(1, profile.getSegments().size());
        assertEquals(0, profile.getTotalClustersVisited());
    }
}
