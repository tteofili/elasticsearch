/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq.next.ash;

import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.packed.PackedInts;
import org.apache.lucene.util.packed.PackedLongValues;
import org.elasticsearch.index.codec.vectors.cluster.KMeansResult;
import org.elasticsearch.index.codec.vectors.diskbbq.CentroidSupplier;
import org.elasticsearch.index.codec.vectors.diskbbq.DocIdsWriter;
import org.elasticsearch.index.codec.vectors.diskbbq.IntSorter;
import org.elasticsearch.index.codec.vectors.diskbbq.next.IvfSegmentConfig;
import org.elasticsearch.logging.LogManager;
import org.elasticsearch.logging.Logger;
import org.elasticsearch.simdvec.ESVectorUtil;

import java.io.IOException;
import java.util.Arrays;

import static org.elasticsearch.simdvec.ES940OSQVectorsScorer.BULK_SIZE;

/**
 * Builds and writes ASH-encoded posting lists for IVF segments.
 * <p>
 * This class encapsulates the full ASH write pipeline:
 * <ol>
 *   <li>Collect vectors from the segment</li>
 *   <li>Run independent ASH k-means clustering (few coarse clusters for centering)</li>
 *   <li>Train the projection matrix W via the ASH optimization procedure</li>
 *   <li>Encode all vectors (project, center, scalar-quantize)</li>
 *   <li>Write posting lists grouped by IVF cluster assignment</li>
 * </ol>
 * <p>
 * The trained {@link AshProjectionMatrix} is retained after writing so the caller can
 * serialize it in the preconditioner slot of the segment file.
 */
public class AshPostingsListWriter {

    private static final Logger logger = LogManager.getLogger(AshPostingsListWriter.class);

    private AshProjectionMatrix ashProjectionMatrix;

    /**
     * Returns the projection matrix trained during the most recent
     * {@link #buildAndWrite} call, or null if not yet called.
     */
    public AshProjectionMatrix getAshProjectionMatrix() {
        return ashProjectionMatrix;
    }

    /**
     * Clears the stored projection matrix (call after writing it to disk).
     */
    public void clearProjectionMatrix() {
        ashProjectionMatrix = null;
    }

    /**
     * Result of writing posting lists: per-cluster offsets and lengths into the postings file.
     */
    public record PostingsOffsetAndLength(PackedLongValues offsets, PackedLongValues lengths) {}

    /**
     * Trains ASH, encodes vectors, and writes posting lists to the given output.
     *
     * @param fieldInfo         field metadata (dimensions, similarity function)
     * @param centroidSupplier  provides IVF cluster centroids and second-level clusters
     * @param floatVectorValues access to the segment's float vectors by ordinal
     * @param postingsOutput    output stream for the posting list data
     * @param fileOffset        base offset in the postings file (for relative addressing)
     * @param assignments       IVF cluster assignment per vector ordinal
     * @param segmentConfig     ASH configuration (totalBits, bitsPerDim, method, training iterations)
     * @return per-cluster offsets and lengths
     */
    public PostingsOffsetAndLength buildAndWrite(
        FieldInfo fieldInfo,
        CentroidSupplier centroidSupplier,
        FloatVectorValues floatVectorValues,
        IndexOutput postingsOutput,
        long fileOffset,
        int[] assignments,
        IvfSegmentConfig segmentConfig
    ) throws IOException {
        int nVectors = assignments.length;
        int originalDim = fieldInfo.getVectorDimension();
        int nClusters = centroidSupplier.size();

        // Collect all vectors into arrays for ASH training
        float[][] vectors = new float[nVectors][originalDim];
        for (int i = 0; i < nVectors; i++) {
            float[] v = floatVectorValues.vectorValue(i);
            System.arraycopy(v, 0, vectors[i], 0, originalDim);
        }

        // Create and train the ASH quantizer
        AsymmetricHashingQuantizer ashQuantizer = new AsymmetricHashingQuantizer(
            segmentConfig.ashTotalBits(),
            segmentConfig.ashBitsPerDim(),
            segmentConfig.ashMethod(),
            segmentConfig.ashTrainingIterations(),
            segmentConfig.ashTrainingFactor(),
            segmentConfig.ashSeed()
        );

        // Run ASH-specific k-means with few clusters (independent of IVF clustering).
        // Python uses n_clusters=16 by default; the IVF clusters are too fine-grained
        // for effective centroid subtraction in ASH.
        long tKmeans0 = System.currentTimeMillis();
        AsymmetricHashingQuantizer.AshKMeansResult ashKMeans = AsymmetricHashingQuantizer.runAshKMeans(
            vectors,
            segmentConfig.ashNumClusters(),
            segmentConfig.ashKMeansMaxIterations(),
            segmentConfig.ashSeed()
        );
        float[][] ashCentroids = ashKMeans.centroids();
        int[] ashAssignments = ashKMeans.assignments();
        long tKmeans1 = System.currentTimeMillis();
        logger.info("ASH k-means: {}ms, {} clusters", tKmeans1 - tKmeans0, ashCentroids.length);

        long t0 = System.currentTimeMillis();
        float[][] w = ashQuantizer.train(vectors, ashCentroids, ashAssignments);
        long t1 = System.currentTimeMillis();
        AsymmetricHashingResult ashResult = ashQuantizer.encode(vectors, ashCentroids, ashAssignments, w);
        long t2 = System.currentTimeMillis();
        logger.info("ASH train: {}ms, encode: {}ms, nDims={}", t1 - t0, t2 - t1, w[0].length);

        // Store the projection matrix + ASH centroids for later serialization
        this.ashProjectionMatrix = new AshProjectionMatrix(w, ashCentroids);

        // Build cluster-to-vector mappings (no SOAR/overspill for ASH)
        int[] centroidVectorCount = new int[nClusters];
        for (int i = 0; i < nVectors; i++) {
            centroidVectorCount[assignments[i]]++;
        }

        int maxPostingListSize = 0;
        int[][] assignmentsByCluster = new int[nClusters][];
        for (int c = 0; c < nClusters; c++) {
            int size = centroidVectorCount[c];
            maxPostingListSize = Math.max(maxPostingListSize, size);
            assignmentsByCluster[c] = new int[size];
        }
        Arrays.fill(centroidVectorCount, 0);

        for (int i = 0; i < nVectors; i++) {
            int c = assignments[i];
            assignmentsByCluster[c][centroidVectorCount[c]++] = i;
        }

        // Write posting lists
        final PackedLongValues.Builder offsets = PackedLongValues.monotonicBuilder(PackedInts.COMPACT);
        final PackedLongValues.Builder lengths = PackedLongValues.monotonicBuilder(PackedInts.COMPACT);
        final int nDims = ashResult.encodedVectors()[0].length;
        final int bitsPerDim = segmentConfig.ashBitsPerDim();
        final int packedCodeBytes = AsymmetricHashingScorer.packedByteLength(nDims, bitsPerDim);
        final int[] docIds = new int[maxPostingListSize];
        final int[] docDeltas = new int[maxPostingListSize];
        final int[] clusterOrds = new int[maxPostingListSize];
        DocIdsWriter idsWriter = new DocIdsWriter();
        KMeansResult centroidClusters = centroidSupplier.secondLevelClusters();

        for (int c = 0; c < nClusters; c++) {
            float[] centroid = centroidSupplier.centroid(c);
            int[] cluster = assignmentsByCluster[c];
            long offset = postingsOutput.alignFilePointer(Float.BYTES) - fileOffset;
            offsets.add(offset);
            // Write parent centroid distance
            postingsOutput.writeInt(Float.floatToIntBits(ESVectorUtil.squareDistance(centroid, centroidClusters.getCentroid(c))));
            int size = cluster.length;
            postingsOutput.writeVInt(size);

            // Sort by docId
            for (int j = 0; j < size; j++) {
                docIds[j] = floatVectorValues.ordToDoc(cluster[j]);
                clusterOrds[j] = j;
            }
            new IntSorter(clusterOrds, i -> docIds[i]).sort(0, size);
            for (int j = 0; j < size; j++) {
                docDeltas[j] = j == 0 ? docIds[clusterOrds[j]] : docIds[clusterOrds[j]] - docIds[clusterOrds[j - 1]];
            }

            byte encoding = idsWriter.calculateBlockEncoding(i -> docDeltas[i], size, BULK_SIZE);
            postingsOutput.writeByte(encoding);

            // Write vectors in bulk blocks: docIds first, then ASH encoded vectors
            int written = 0;
            while (written < size) {
                int blockSize = Math.min(BULK_SIZE, size - written);
                final int blockStart = written;
                idsWriter.writeDocIds(d -> docDeltas[blockStart + d], blockSize, encoding, postingsOutput);
                for (int j = 0; j < blockSize; j++) {
                    int vectorOrd = cluster[clusterOrds[written + j]];
                    float[] encVec = ashResult.encodedVectors()[vectorOrd];
                    float scale = ashResult.scales()[vectorOrd];
                    float off = ashResult.offsets()[vectorOrd];
                    // Write: ashClusterId (1 byte), scale (float16), offset (float16), packed codes
                    postingsOutput.writeByte((byte) ashAssignments[vectorOrd]);
                    postingsOutput.writeShort(Float.floatToFloat16(scale));
                    postingsOutput.writeShort(Float.floatToFloat16(off));
                    byte[] packed = bitsPerDim == 1
                        ? AsymmetricHashingScorer.packBinaryCodes(encVec)
                        : AsymmetricHashingScorer.packMultiBitCodes(encVec, bitsPerDim);
                    postingsOutput.writeBytes(packed, packed.length);
                }
                written += blockSize;
            }
            lengths.add(postingsOutput.getFilePointer() - fileOffset - offset);
        }

        if (logger.isDebugEnabled()) {
            printClusterQualityStatistics(assignmentsByCluster);
        }

        return new PostingsOffsetAndLength(offsets.build(), lengths.build());
    }

    private static void printClusterQualityStatistics(int[][] clusters) {
        float min = Float.MAX_VALUE;
        float max = Float.MIN_VALUE;
        float mean = 0;
        float m2 = 0;
        int count = 0;
        for (int[] cluster : clusters) {
            count += 1;
            if (cluster == null) {
                continue;
            }
            float delta = cluster.length - mean;
            mean += delta / count;
            m2 += delta * (cluster.length - mean);
            min = Math.min(min, cluster.length);
            max = Math.max(max, cluster.length);
        }
        float variance = m2 / (clusters.length - 1);
        logger.debug(
            "Centroid count: {} min: {} max: {} mean: {} stdDev: {} variance: {}",
            clusters.length,
            min,
            max,
            mean,
            Math.sqrt(variance),
            variance
        );
    }
}
