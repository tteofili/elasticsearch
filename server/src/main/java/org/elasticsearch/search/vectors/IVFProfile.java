/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.search.vectors;

import org.elasticsearch.common.Strings;
import org.elasticsearch.common.io.stream.StreamInput;
import org.elasticsearch.common.io.stream.StreamOutput;
import org.elasticsearch.common.io.stream.Writeable;
import org.elasticsearch.xcontent.ToXContentObject;
import org.elasticsearch.xcontent.XContentBuilder;

import java.io.IOException;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Aggregate IVF profiling data for a single KNN query: per-segment stats and
 * global query-centroid and cluster-size distribution metrics.
 */
public final class IVFProfile implements Writeable, ToXContentObject {

    public static final String SEGMENTS = "segments";
    public static final String TOTAL_VECTORS_VISITED = "total_vectors_visited";
    public static final String TOTAL_CLUSTERS_VISITED = "total_clusters_visited";
    public static final String CENTROID_SCORE_MIN = "centroid_score_min";
    public static final String CENTROID_SCORE_MAX = "centroid_score_max";
    public static final String CENTROID_SCORE_MEAN = "centroid_score_mean";
    public static final String CENTROID_SCORE_STD = "centroid_score_std";
    public static final String CLUSTER_SIZE_MIN = "cluster_size_min";
    public static final String CLUSTER_SIZE_MAX = "cluster_size_max";
    public static final String CLUSTER_SIZE_MEAN = "cluster_size_mean";
    public static final String CLUSTER_SIZE_STD = "cluster_size_std";

    private final long totalVectorsVisited;
    private final long totalClustersVisited;
    private final List<IVFSegmentProfile> segments;
    private final float centroidScoreMin;
    private final float centroidScoreMax;
    private final float centroidScoreMean;
    private final float centroidScoreStd;
    private final int clusterSizeMin;
    private final int clusterSizeMax;
    private final float clusterSizeMean;
    private final float clusterSizeStd;

    public IVFProfile(
        long totalVectorsVisited,
        long totalClustersVisited,
        List<IVFSegmentProfile> segments,
        float centroidScoreMin,
        float centroidScoreMax,
        float centroidScoreMean,
        float centroidScoreStd,
        int clusterSizeMin,
        int clusterSizeMax,
        float clusterSizeMean,
        float clusterSizeStd
    ) {
        this.totalVectorsVisited = totalVectorsVisited;
        this.totalClustersVisited = totalClustersVisited;
        this.segments = segments == null ? List.of() : List.copyOf(segments);
        this.centroidScoreMin = centroidScoreMin;
        this.centroidScoreMax = centroidScoreMax;
        this.centroidScoreMean = centroidScoreMean;
        this.centroidScoreStd = centroidScoreStd;
        this.clusterSizeMin = clusterSizeMin;
        this.clusterSizeMax = clusterSizeMax;
        this.clusterSizeMean = clusterSizeMean;
        this.clusterSizeStd = clusterSizeStd;
    }

    public IVFProfile(StreamInput in) throws IOException {
        this.totalVectorsVisited = in.readVLong();
        this.totalClustersVisited = in.readVLong();
        this.segments = in.readCollectionAsList(IVFSegmentProfile::new);
        this.centroidScoreMin = in.readFloat();
        this.centroidScoreMax = in.readFloat();
        this.centroidScoreMean = in.readFloat();
        this.centroidScoreStd = in.readFloat();
        this.clusterSizeMin = in.readVInt();
        this.clusterSizeMax = in.readVInt();
        this.clusterSizeMean = in.readFloat();
        this.clusterSizeStd = in.readFloat();
    }

    public long getTotalVectorsVisited() {
        return totalVectorsVisited;
    }

    public long getTotalClustersVisited() {
        return totalClustersVisited;
    }

    public List<IVFSegmentProfile> getSegments() {
        return segments;
    }

    public float getCentroidScoreMin() {
        return centroidScoreMin;
    }

    public float getCentroidScoreMax() {
        return centroidScoreMax;
    }

    public float getCentroidScoreMean() {
        return centroidScoreMean;
    }

    public float getCentroidScoreStd() {
        return centroidScoreStd;
    }

    public int getClusterSizeMin() {
        return clusterSizeMin;
    }

    public int getClusterSizeMax() {
        return clusterSizeMax;
    }

    public float getClusterSizeMean() {
        return clusterSizeMean;
    }

    public float getClusterSizeStd() {
        return clusterSizeStd;
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        out.writeVLong(totalVectorsVisited);
        out.writeVLong(totalClustersVisited);
        out.writeCollection(segments);
        out.writeFloat(centroidScoreMin);
        out.writeFloat(centroidScoreMax);
        out.writeFloat(centroidScoreMean);
        out.writeFloat(centroidScoreStd);
        out.writeVInt(clusterSizeMin);
        out.writeVInt(clusterSizeMax);
        out.writeFloat(clusterSizeMean);
        out.writeFloat(clusterSizeStd);
    }

    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        builder.startObject();
        builder.field(TOTAL_VECTORS_VISITED, totalVectorsVisited);
        builder.field(TOTAL_CLUSTERS_VISITED, totalClustersVisited);
        builder.field(CENTROID_SCORE_MIN, centroidScoreMin);
        builder.field(CENTROID_SCORE_MAX, centroidScoreMax);
        builder.field(CENTROID_SCORE_MEAN, centroidScoreMean);
        builder.field(CENTROID_SCORE_STD, centroidScoreStd);
        builder.field(CLUSTER_SIZE_MIN, clusterSizeMin);
        builder.field(CLUSTER_SIZE_MAX, clusterSizeMax);
        builder.field(CLUSTER_SIZE_MEAN, clusterSizeMean);
        builder.field(CLUSTER_SIZE_STD, clusterSizeStd);
        builder.startArray(SEGMENTS);
        for (IVFSegmentProfile segment : segments) {
            segment.toXContent(builder, params);
        }
        builder.endArray();
        builder.endObject();
        return builder;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        IVFProfile that = (IVFProfile) obj;
        return totalVectorsVisited == that.totalVectorsVisited
            && totalClustersVisited == that.totalClustersVisited
            && Float.compare(that.centroidScoreMin, centroidScoreMin) == 0
            && Float.compare(that.centroidScoreMax, centroidScoreMax) == 0
            && Float.compare(that.centroidScoreMean, centroidScoreMean) == 0
            && Float.compare(that.centroidScoreStd, centroidScoreStd) == 0
            && clusterSizeMin == that.clusterSizeMin
            && clusterSizeMax == that.clusterSizeMax
            && Float.compare(that.clusterSizeMean, clusterSizeMean) == 0
            && Float.compare(that.clusterSizeStd, clusterSizeStd) == 0
            && segments.equals(that.segments);
    }

    @Override
    public int hashCode() {
        return Objects.hash(
            totalVectorsVisited,
            totalClustersVisited,
            segments,
            centroidScoreMin,
            centroidScoreMax,
            centroidScoreMean,
            centroidScoreStd,
            clusterSizeMin,
            clusterSizeMax,
            clusterSizeMean,
            clusterSizeStd
        );
    }

    @Override
    public String toString() {
        return Strings.toString(this);
    }
}
