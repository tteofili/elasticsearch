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
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.store.IndexInput;
import org.elasticsearch.index.codec.vectors.cluster.KMeansFloatVectorValues;
import org.elasticsearch.index.codec.vectors.cluster.KMeansResult;
import org.elasticsearch.index.codec.vectors.diskbbq.CentroidIndex;
import org.elasticsearch.index.codec.vectors.diskbbq.CentroidSlices;
import org.elasticsearch.index.codec.vectors.diskbbq.CentroidSupplier;
import org.elasticsearch.index.codec.vectors.diskbbq.FlatCentroidClusters;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Reads the raw centroids of a segment being merged from the temporary centroid file, a centroid at a time, instead of
 * keeping them on the heap. Byte-backed fields store one byte per dimension and float-backed ones four.
 */
class OffHeapCentroidSupplier implements CentroidSupplier {
    private final IndexInput centroidsInput;
    private final int numCentroids;
    private final int dimension;
    private final float[] scratch;
    private final byte[] byteScratch; // null for FLOAT32
    private final KMeansResult<float[]> clusters;
    private final CentroidSlices centroidSlices;
    private final VectorEncoding encoding;
    private int currOrd = -1;

    OffHeapCentroidSupplier(
        IndexInput centroidsInput,
        int numCentroids,
        FieldInfo info,
        KMeansResult<float[]> clusters,
        CentroidSlices centroidSlices
    ) {
        this.centroidsInput = centroidsInput;
        this.numCentroids = numCentroids;
        this.dimension = info.getVectorDimension();
        this.scratch = new float[dimension];
        this.encoding = info.getVectorEncoding();
        this.byteScratch = encoding == VectorEncoding.BYTE ? new byte[dimension] : null;
        this.clusters = clusters;
        this.centroidSlices = centroidSlices;
    }

    @Override
    public int size() {
        return numCentroids;
    }

    private void readByteCentroid(int centroidOrdinal) throws IOException {
        if (centroidOrdinal != currOrd) {
            centroidsInput.seek((long) centroidOrdinal * dimension);
            centroidsInput.readBytes(byteScratch, 0, dimension);
            for (int d = 0; d < dimension; d++) {
                scratch[d] = byteScratch[d];
            }
            currOrd = centroidOrdinal;
        }
    }

    @Override
    public float[] centroid(int centroidOrdinal) throws IOException {
        if (encoding == VectorEncoding.BYTE) {
            readByteCentroid(centroidOrdinal);
            return scratch;
        }
        if (centroidOrdinal != currOrd) {
            centroidsInput.seek((long) centroidOrdinal * dimension * Float.BYTES);
            centroidsInput.readFloats(scratch, 0, dimension);
            currOrd = centroidOrdinal;
        }
        return scratch;
    }

    @Override
    public byte[] byteCentroid(int centroidOrdinal) throws IOException {
        if (encoding != VectorEncoding.BYTE) {
            return null;
        }
        readByteCentroid(centroidOrdinal);
        return byteScratch;
    }

    @Override
    public CentroidIndex centroidIndex() {
        return new FlatCentroidClusters(clusters);
    }

    @Override
    public CentroidSlices slices() throws IOException {
        return centroidSlices;
    }

    @Override
    public KMeansFloatVectorValues asKmeansFloatVectorValues() throws IOException {
        if (encoding == VectorEncoding.BYTE) {
            List<float[]> floatCentroids = new ArrayList<>(numCentroids);
            for (int i = 0; i < numCentroids; i++) {
                centroidsInput.seek((long) i * dimension);
                float[] fc = new float[dimension];
                for (int d = 0; d < dimension; d++) {
                    fc[d] = centroidsInput.readByte();
                }
                floatCentroids.add(fc);
            }
            return KMeansFloatVectorValues.build(floatCentroids, null, dimension);
        }
        return KMeansFloatVectorValues.build(centroidsInput, null, numCentroids, dimension);
    }
}
