/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq.quantize;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.elasticsearch.index.codec.vectors.OptimizedScalarQuantizer;
import org.elasticsearch.index.codec.vectors.diskbbq.QuantEncoding;
import org.elasticsearch.simdvec.ESVectorUtil;

/**
 * OSQ as a {@link DocEncoder}: each vector is quantized on its own, against the centroid of the block, with the
 * quantizer segment writers use. The latent space is the one of the vectors.
 * <p>
 * It optionally rotates what it is given first, for the callers that are not given vectors that are already in the
 * space that OSQ quantizes in, as a segment writer is.
 */
public final class OsqDocEncoder implements DocEncoder {

    /** Applies a rotation, which must be orthogonal. */
    @FunctionalInterface
    public interface Rotation {
        void apply(float[] in, float[] out);
    }

    private final OsqVectorQuantizer quantizer;
    private final int dimension;
    private final byte bits;
    private final float docScale;
    private final QuantEncoding encoding;
    private final byte[] packedScratch;
    private final Rotation rotation;
    private final float[] rotated;
    private final float[] rotatedCentroid;
    // the block
    private final int[][] levels;
    private final float[][] residuals;
    private final float[] scales;
    private final float[] uppers;
    private final float[] lowers;
    private final int[] sums;
    private final float[] offsets;
    private float[] centroid;
    private int size;

    /**
     * @param quantizer the quantizer to quantize with, which decides the lambda and the iterations
     * @param similarity how the vectors are compared
     * @param rotation rotates the vectors and the centroid before they are quantized, or {@code null} when they are in the
     *                 space that OSQ quantizes in already
     */
    public OsqDocEncoder(
        OptimizedScalarQuantizer quantizer,
        VectorSimilarityFunction similarity,
        int dimension,
        int bits,
        int maxBlockSize,
        Rotation rotation
    ) {
        this.quantizer = new OsqVectorQuantizer(quantizer, similarity, dimension, bits);
        this.dimension = dimension;
        this.bits = (byte) bits;
        this.docScale = 1f / ((1 << bits) - 1);
        this.encoding = QuantEncoding.fromBits((byte) bits);
        this.packedScratch = new byte[encoding.getDocPackedLength(dimension)];
        this.rotation = rotation;
        this.rotated = new float[dimension];
        this.rotatedCentroid = new float[dimension];
        this.levels = new int[maxBlockSize][encoding.discretizedDimensions(dimension)];
        this.residuals = new float[maxBlockSize][dimension];
        this.scales = new float[maxBlockSize];
        this.uppers = new float[maxBlockSize];
        this.lowers = new float[maxBlockSize];
        this.sums = new int[maxBlockSize];
        this.offsets = new float[maxBlockSize];
    }

    /** The encoder of a segment writer: OSQ with the similarity of the field, and no rotation of its own. */
    public static OsqDocEncoder forField(VectorSimilarityFunction similarity, int dimension, int bits, int maxBlockSize) {
        return new OsqDocEncoder(new OptimizedScalarQuantizer(similarity), similarity, dimension, bits, maxBlockSize, null);
    }

    @Override
    public int dimension() {
        return dimension;
    }

    @Override
    public int latentDimension() {
        return dimension;
    }

    @Override
    public int bits() {
        return bits;
    }

    @Override
    public int codeLength() {
        return dimension;
    }

    @Override
    public int maxBlockSize() {
        return levels.length;
    }

    @Override
    public void reset(float[] centroid) {
        if (rotation == null) {
            this.centroid = centroid;
        } else {
            rotation.apply(centroid, rotatedCentroid);
            this.centroid = rotatedCentroid;
        }
        size = 0;
    }

    @Override
    public void add(float[] vector) {
        assert size < levels.length : "block is full";
        float[] v = vector;
        if (rotation != null) {
            rotation.apply(vector, rotated);
            v = rotated;
        }
        OptimizedScalarQuantizer.QuantizationResult result = quantizer.quantize(v, centroid, null, levels[size]);
        System.arraycopy(v, 0, residuals[size], 0, dimension);
        ESVectorUtil.linearCombination(-1f, centroid, 1f, residuals[size]);
        lowers[size] = result.lowerInterval();
        uppers[size] = result.upperInterval();
        scales[size] = (result.upperInterval() - result.lowerInterval()) / ((1 << bits) - 1);
        sums[size] = result.quantizedComponentSum();
        offsets[size] = result.additionalCorrection();
        size++;
    }

    @Override
    public void encode() {
        // each vector is quantized as it is added
    }

    @Override
    public int size() {
        return size;
    }

    @Override
    public void read(int i, int[] levels, DocLanes lanes) {
        System.arraycopy(this.levels[i], 0, levels, 0, dimension);
        lanes.scale = scales[i];
        lanes.lower = lowers[i];
        lanes.sum = sums[i];
        lanes.offset = offsets[i];
    }

    @Override
    public int packedLength() {
        return packedScratch.length;
    }

    @Override
    public void pack(int i, byte[] destination, int offset) {
        encoding.pack(levels[i], packedScratch);
        System.arraycopy(packedScratch, 0, destination, offset, packedScratch.length);
    }

    @Override
    public int nativeLaneCount() {
        return 4;
    }

    @Override
    public void nativeLanes(int i, int[] bits) {
        bits[0] = Float.floatToIntBits(lowers[i]);
        bits[1] = Float.floatToIntBits(uppers[i]);
        bits[2] = Float.floatToIntBits(offsets[i]);
        bits[3] = sums[i];
    }

    @Override
    public void scoreLanes(int i, ScoreLanes lanes) {
        lanes.s = (uppers[i] - lowers[i]) * docScale;
        lanes.sigma = sums[i];
        lanes.l = lowers[i];
        lanes.o = offsets[i];
    }

    @Override
    public void latent(int i, float[] latent) {
        System.arraycopy(residuals[i], 0, latent, 0, dimension);
    }

    @Override
    public void close() {}
}
