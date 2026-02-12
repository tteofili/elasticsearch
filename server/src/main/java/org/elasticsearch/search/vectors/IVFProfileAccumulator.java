/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.search.vectors;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

/**
 * Thread-safe accumulator for IVF profiling. Each segment (leaf) uses a
 * dedicated segment builder; when the segment finishes, the builder is
 * converted to {@link IVFSegmentProfile} and appended to a concurrent list.
 * {@link #buildProfile()} merges all segment profiles into a single
 * {@link IVFProfile}.
 */
public final class IVFProfileAccumulator {

    private final List<IVFSegmentProfile> segments = new CopyOnWriteArrayList<>();

    /**
     * Create a new segment builder for the current leaf. Each leaf must call
     * this once, then {@link IVFSegmentBuilder#recordCluster} and finally
     * {@link #finishSegment(IVFSegmentBuilder)}.
     */
    public IVFSegmentBuilder createSegmentBuilder() {
        return new IVFSegmentBuilder();
    }

    /**
     * Finalize the segment and add its profile to the list. Call once per leaf
     * after all clusters have been recorded.
     */
    public void finishSegment(IVFSegmentBuilder builder) {
        IVFSegmentProfile profile = builder.build();
        if (profile != null) {
            segments.add(profile);
        }
    }

    /**
     * Build the aggregate profile from all completed segments. May be called
     * only once after all leaves have finished (e.g. from profile()).
     */
    public IVFProfile buildProfile() {
        if (segments.isEmpty()) {
            return null;
        }
        long totalVectors = 0;
        long totalClusters = 0;
        float centroidScoreMin = Float.POSITIVE_INFINITY;
        float centroidScoreMax = Float.NEGATIVE_INFINITY;
        double centroidScoreSum = 0;
        long centroidScoreCount = 0;
        double centroidScoreSumSq = 0;
        int clusterSizeMin = Integer.MAX_VALUE;
        int clusterSizeMax = Integer.MIN_VALUE;
        long clusterSizeSum = 0;
        long clusterSizeCount = 0;
        long clusterSizeSumSq = 0;

        for (IVFSegmentProfile seg : segments) {
            totalVectors += seg.vectorsVisited();
            totalClusters += seg.clustersVisited();
            if (seg.centroidScoreCount() > 0) {
                centroidScoreMin = Math.min(centroidScoreMin, seg.centroidScoreMin());
                centroidScoreMax = Math.max(centroidScoreMax, seg.centroidScoreMax());
                centroidScoreSum += seg.centroidScoreSum();
                centroidScoreCount += seg.centroidScoreCount();
                centroidScoreSumSq += seg.centroidScoreSumSq();
            }
            if (seg.clusterSizeCount() > 0) {
                clusterSizeMin = Math.min(clusterSizeMin, seg.clusterSizeMin());
                clusterSizeMax = Math.max(clusterSizeMax, seg.clusterSizeMax());
                clusterSizeSum += seg.clusterSizeSum();
                clusterSizeCount += seg.clusterSizeCount();
                clusterSizeSumSq += seg.clusterSizeSumSq();
            }
        }

        float centroidScoreMean = centroidScoreCount == 0 ? Float.NaN : (float) (centroidScoreSum / centroidScoreCount);
        float centroidScoreStd = Float.NaN;
        if (centroidScoreCount > 1) {
            double mean = centroidScoreSum / centroidScoreCount;
            double variance = (centroidScoreSumSq - centroidScoreCount * mean * mean) / (centroidScoreCount - 1);
            centroidScoreStd = (float) Math.sqrt(Math.max(0, variance));
        }
        if (centroidScoreCount == 0) {
            centroidScoreMin = Float.NaN;
            centroidScoreMax = Float.NaN;
        }

        float clusterSizeMean = clusterSizeCount == 0 ? Float.NaN : (float) ((double) clusterSizeSum / clusterSizeCount);
        float clusterSizeStd = Float.NaN;
        if (clusterSizeCount > 1) {
            double mean = (double) clusterSizeSum / clusterSizeCount;
            double variance = ((double) clusterSizeSumSq - clusterSizeCount * mean * mean) / (clusterSizeCount - 1);
            clusterSizeStd = (float) Math.sqrt(Math.max(0, variance));
        }
        int cMin = clusterSizeCount == 0 ? 0 : clusterSizeMin;
        int cMax = clusterSizeCount == 0 ? 0 : clusterSizeMax;

        return new IVFProfile(
            totalVectors,
            totalClusters,
            new ArrayList<>(segments),
            centroidScoreMin,
            centroidScoreMax,
            centroidScoreMean,
            centroidScoreStd,
            cMin,
            cMax,
            clusterSizeMean,
            clusterSizeStd
        );
    }

    /**
     * Mutable builder for one segment's stats. Not thread-safe; use one per
     * leaf.
     */
    public static final class IVFSegmentBuilder {
        private int numVectors = -1;
        private int numCentroids = -1;
        private int clustersVisited = 0;
        private int vectorsVisited = 0;
        private float centroidScoreMin = Float.POSITIVE_INFINITY;
        private float centroidScoreMax = Float.NEGATIVE_INFINITY;
        private double centroidScoreSum = 0;
        private long centroidScoreCount = 0;
        private double centroidScoreSumSq = 0;
        private int clusterSizeMin = Integer.MAX_VALUE;
        private int clusterSizeMax = Integer.MIN_VALUE;
        private long clusterSizeSum = 0;
        private long clusterSizeCount = 0;
        private long clusterSizeSumSq = 0;

        public void recordSegmentStart(int numVectors, int numCentroids) {
            this.numVectors = numVectors;
            this.numCentroids = numCentroids;
        }

        public void recordCluster(float centroidScore, int postingListSize) {
            clustersVisited++;
            centroidScoreMin = Math.min(centroidScoreMin, centroidScore);
            centroidScoreMax = Math.max(centroidScoreMax, centroidScore);
            centroidScoreSum += centroidScore;
            centroidScoreCount++;
            centroidScoreSumSq += (double) centroidScore * centroidScore;
            clusterSizeMin = Math.min(clusterSizeMin, postingListSize);
            clusterSizeMax = Math.max(clusterSizeMax, postingListSize);
            clusterSizeSum += postingListSize;
            clusterSizeCount++;
            clusterSizeSumSq += (long) postingListSize * postingListSize;
        }

        public void recordSegmentEnd(int vectorsVisited) {
            this.vectorsVisited = vectorsVisited;
        }

        IVFSegmentProfile build() {
            if (numVectors < 0) {
                return null;
            }
            return new IVFSegmentProfile(
                numVectors,
                numCentroids,
                clustersVisited,
                vectorsVisited,
                centroidScoreCount == 0 ? Float.NaN : centroidScoreMin,
                centroidScoreCount == 0 ? Float.NaN : centroidScoreMax,
                centroidScoreSum,
                centroidScoreCount,
                centroidScoreSumSq,
                clusterSizeCount == 0 ? 0 : clusterSizeMin,
                clusterSizeCount == 0 ? 0 : clusterSizeMax,
                clusterSizeSum,
                clusterSizeCount,
                clusterSizeSumSq
            );
        }
    }
}
