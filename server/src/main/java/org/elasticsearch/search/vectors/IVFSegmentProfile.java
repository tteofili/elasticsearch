/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.search.vectors;

import org.elasticsearch.common.io.stream.StreamInput;
import org.elasticsearch.common.io.stream.StreamOutput;
import org.elasticsearch.common.io.stream.Writeable;
import org.elasticsearch.xcontent.ToXContentObject;
import org.elasticsearch.xcontent.XContentBuilder;

import java.io.IOException;

/**
 * Immutable per-segment profiling stats for IVF vector search.
 * Captures cluster affinity, query-centroid distribution, and cluster size distribution for one segment.
 */
public record IVFSegmentProfile(
    int numVectors,
    int numCentroids,
    int clustersVisited,
    int vectorsVisited,
    float centroidScoreMin,
    float centroidScoreMax,
    double centroidScoreSum,
    long centroidScoreCount,
    double centroidScoreSumSq,
    int clusterSizeMin,
    int clusterSizeMax,
    long clusterSizeSum,
    long clusterSizeCount,
    long clusterSizeSumSq
) implements Writeable, ToXContentObject {

    public static final String NUM_VECTORS = "num_vectors";
    public static final String NUM_CENTROIDS = "num_centroids";
    public static final String CLUSTERS_VISITED = "clusters_visited";
    public static final String VECTORS_VISITED = "vectors_visited";
    public static final String CENTROID_SCORE_MIN = "centroid_score_min";
    public static final String CENTROID_SCORE_MAX = "centroid_score_max";
    public static final String CENTROID_SCORE_MEAN = "centroid_score_mean";
    public static final String CENTROID_SCORE_STD = "centroid_score_std";
    public static final String CLUSTER_SIZE_MIN = "cluster_size_min";
    public static final String CLUSTER_SIZE_MAX = "cluster_size_max";
    public static final String CLUSTER_SIZE_MEAN = "cluster_size_mean";
    public static final String CLUSTER_SIZE_STD = "cluster_size_std";

    public IVFSegmentProfile(StreamInput in) throws IOException {
        this(
            in.readVInt(),
            in.readVInt(),
            in.readVInt(),
            in.readVInt(),
            in.readFloat(),
            in.readFloat(),
            in.readDouble(),
            in.readVLong(),
            in.readDouble(),
            in.readVInt(),
            in.readVInt(),
            in.readVLong(),
            in.readVLong(),
            in.readVLong()
        );
    }

    public float centroidScoreMean() {
        return centroidScoreCount == 0 ? Float.NaN : (float) (centroidScoreSum / centroidScoreCount);
    }

    public float centroidScoreStd() {
        if (centroidScoreCount <= 1) return Float.NaN;
        double mean = centroidScoreSum / centroidScoreCount;
        double variance = (centroidScoreSumSq - centroidScoreCount * mean * mean) / (centroidScoreCount - 1);
        return (float) Math.sqrt(Math.max(0, variance));
    }

    public float clusterSizeMean() {
        return clusterSizeCount == 0 ? Float.NaN : (float) ((double) clusterSizeSum / clusterSizeCount);
    }

    public float clusterSizeStd() {
        if (clusterSizeCount <= 1) return Float.NaN;
        double mean = (double) clusterSizeSum / clusterSizeCount;
        double variance = ((double) clusterSizeSumSq - clusterSizeCount * mean * mean) / (clusterSizeCount - 1);
        return (float) Math.sqrt(Math.max(0, variance));
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        out.writeVInt(numVectors);
        out.writeVInt(numCentroids);
        out.writeVInt(clustersVisited);
        out.writeVInt(vectorsVisited);
        out.writeFloat(centroidScoreMin);
        out.writeFloat(centroidScoreMax);
        out.writeDouble(centroidScoreSum);
        out.writeVLong(centroidScoreCount);
        out.writeDouble(centroidScoreSumSq);
        out.writeVInt(clusterSizeMin);
        out.writeVInt(clusterSizeMax);
        out.writeVLong(clusterSizeSum);
        out.writeVLong(clusterSizeCount);
        out.writeVLong(clusterSizeSumSq);
    }

    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        builder.startObject();
        builder.field(NUM_VECTORS, numVectors);
        builder.field(NUM_CENTROIDS, numCentroids);
        builder.field(CLUSTERS_VISITED, clustersVisited);
        builder.field(VECTORS_VISITED, vectorsVisited);
        if (centroidScoreCount > 0) {
            builder.field(CENTROID_SCORE_MIN, centroidScoreMin);
            builder.field(CENTROID_SCORE_MAX, centroidScoreMax);
            builder.field(CENTROID_SCORE_MEAN, centroidScoreMean());
            if (centroidScoreCount > 1) {
                builder.field(CENTROID_SCORE_STD, centroidScoreStd());
            }
        }
        if (clusterSizeCount > 0) {
            builder.field(CLUSTER_SIZE_MIN, clusterSizeMin);
            builder.field(CLUSTER_SIZE_MAX, clusterSizeMax);
            builder.field(CLUSTER_SIZE_MEAN, clusterSizeMean());
            if (clusterSizeCount > 1) {
                builder.field(CLUSTER_SIZE_STD, clusterSizeStd());
            }
        }
        builder.endObject();
        return builder;
    }
}
