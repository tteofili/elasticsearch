/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.ash;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.elasticsearch.index.codec.vectors.diskbbq.quantize.DocEncoder;
import org.elasticsearch.index.codec.vectors.diskbbq.quantize.DocLanes;
import org.elasticsearch.index.codec.vectors.diskbbq.quantize.ScoreLanes;
import org.elasticsearch.simdvec.ESVectorUtil;

/**
 * ASH as a {@link DocEncoder}: a block is projected into the latent space and quantized together, by the encoder that
 * segment writers use, and the levels are those of its centered codes.
 */
public final class AshDocEncoder implements DocEncoder {

    private final AshProjectionMatrix matrix;
    private final int bits;
    private final float centerOffset;
    private final AsymmetricHashingQuantizer.BlockEncoder encoder;
    private final int maxBlockSize;
    private final boolean euclidean;
    private final float[] vectorCentroidSqDists;
    private final int[] levelsScratch;
    private final DocLanes decodeScratch = new DocLanes();
    private float[] centroid;
    private AsymmetricHashingQuantizer.VectorAndNorm precomputed;

    /**
     * @param matrix the projection
     * @param quantizer the quantizer whose bits per dimension decide the encoding; it does not need to have made the matrix
     * @param w the matrix as the encoder wants it, row-major (dimension x latent dimension)
     */
    AshDocEncoder(
        AshProjectionMatrix matrix,
        AsymmetricHashingQuantizer quantizer,
        float[] w,
        VectorSimilarityFunction similarity,
        int bits,
        int maxBlockSize
    ) {
        this.matrix = matrix;
        this.bits = bits;
        this.centerOffset = ((1 << bits) - 1) / 2.0f;
        this.maxBlockSize = maxBlockSize;
        this.euclidean = similarity == VectorSimilarityFunction.EUCLIDEAN;
        this.vectorCentroidSqDists = new float[maxBlockSize];
        this.levelsScratch = new int[matrix.nDims()];
        this.encoder = quantizer.newBlockEncoder(w, matrix.originalDim(), maxBlockSize);
    }

    @Override
    public int dimension() {
        return matrix.originalDim();
    }

    @Override
    public int latentDimension() {
        return matrix.nDims();
    }

    @Override
    public int bits() {
        return bits;
    }

    @Override
    public int codeLength() {
        return matrix.nDims();
    }

    @Override
    public int maxBlockSize() {
        return maxBlockSize;
    }

    @Override
    public void reset(float[] centroid) {
        this.centroid = centroid;
        this.precomputed = AsymmetricHashingQuantizer.precomputeCentroid(centroid, matrix.wT());
        encoder.reset();
    }

    @Override
    public void add(float[] vector) {
        if (euclidean) {
            vectorCentroidSqDists[encoder.size()] = ESVectorUtil.squareDistance(vector, centroid);
        }
        encoder.add(vector, centroid);
    }

    @Override
    public void encode() {
        encoder.encode(precomputed);
    }

    @Override
    public int size() {
        return encoder.size();
    }

    @Override
    public void read(int i, int[] levels, DocLanes lanes) {
        float[] code = encoder.code(i);
        int sum = 0;
        for (int d = 0; d < matrix.nDims(); d++) {
            levels[d] = Math.round(code[d] + centerOffset);
            sum += levels[d];
        }
        float scale = encoder.scale(i);
        lanes.scale = scale;
        lanes.lower = -scale * centerOffset;
        lanes.sum = sum;
        lanes.offset = encoder.offset(i);
    }

    @Override
    public int packedLength() {
        return bits * ((matrix.nDims() + 7) >>> 3);
    }

    @Override
    public void pack(int i, byte[] destination, int offset) {
        read(i, levelsScratch, decodeScratch);
        byte[] packed = ESVectorUtil.ashPackLevels(levelsScratch, bits);
        System.arraycopy(packed, 0, destination, offset, packed.length);
    }

    @Override
    public int nativeLaneCount() {
        return 5;
    }

    @Override
    public void nativeLanes(int i, int[] bits) {
        read(i, levelsScratch, decodeScratch);
        bits[0] = Float.floatToIntBits(encoder.scale(i));
        bits[1] = Float.floatToIntBits(encoder.offset(i));
        bits[2] = decodeScratch.sum;
        // the two lanes that only the Euclidean distance uses, which are zero for the others
        bits[3] = Float.floatToIntBits(euclidean ? encoder.vecCentroidDot(i) : 0f);
        bits[4] = Float.floatToIntBits(euclidean ? vectorCentroidSqDists[i] : 0f);
    }

    @Override
    public void scoreLanes(int i, ScoreLanes lanes) {
        read(i, levelsScratch, decodeScratch);
        lanes.s = encoder.scale(i);
        lanes.sigma = decodeScratch.sum;
        lanes.l = 0f;
        // the squared distance to the centroid and the inner product with it, which the Euclidean distance needs of a vector
        lanes.o = euclidean ? vectorCentroidSqDists[i] - 2 * encoder.offset(i) + 2 * encoder.vecCentroidDot(i) : encoder.offset(i);
    }

    @Override
    public void latent(int i, float[] latent) {
        // the encoder keeps the projection of the block, as it is proportional to the direction of each vector
        System.arraycopy(encoder.latent, i * latent.length, latent, 0, latent.length);
    }

    @Override
    public void close() {
        encoder.close();
    }
}
