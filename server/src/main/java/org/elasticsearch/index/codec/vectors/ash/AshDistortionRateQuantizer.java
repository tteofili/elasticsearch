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
import org.elasticsearch.index.codec.vectors.diskbbq.IvfSegmentConfig;
import org.elasticsearch.index.codec.vectors.diskbbq.calibrate.ClusteredSample;
import org.elasticsearch.index.codec.vectors.diskbbq.calibrate.DistortionRateQuantizer;
import org.elasticsearch.index.codec.vectors.diskbbq.calibrate.EncoderDistortionRateQuantizer;
import org.elasticsearch.index.codec.vectors.diskbbq.calibrate.RotationKind;
import org.elasticsearch.index.codec.vectors.diskbbq.quantize.AffineDocDecoder;
import org.elasticsearch.index.codec.vectors.diskbbq.quantize.DocDecoder;
import org.elasticsearch.index.codec.vectors.diskbbq.quantize.DocEncoder;
import org.elasticsearch.index.codec.vectors.diskbbq.quantize.PreparedQuery;
import org.elasticsearch.simdvec.ESVectorUtil;

import java.io.IOException;
import java.util.Random;

/**
 * ASH, asymmetric hashing, as a {@link DistortionRateQuantizer}: the projection matrix W takes residuals to the latent
 * space (a learned subspace of the data, rotated), where the direction is quantized onto a uniform grid, and the stored
 * gain is the residual's norm, the stored-norm gain the distortion-rate law is stated for. It works on a trained matrix,
 * as stored in a segment, or on a random one.
 */
public final class AshDistortionRateQuantizer extends EncoderDistortionRateQuantizer {

    /** The encoder is deterministic: its constructor takes a seed only for the projection it is not asked to make here. */
    private static final long ENCODER_SEED = 0L;

    private final AshProjectionMatrix matrix;
    private final int bits;
    private final int queryBits;
    private final RotationKind kind;
    private final AsymmetricHashingQuantizer quantizer;
    private final float[] w; // W, row-major (dimension x latent dimension)

    /**
     * @param matrix the projection matrix, W transposed, as a segment stores it
     * @param bitsPerDim bits per latent dimension of the documents
     * @param queryBitsPerDim bits per latent dimension of the query, or 0 when the query is not quantized
     */
    public AshDistortionRateQuantizer(AshProjectionMatrix matrix, int bitsPerDim, int queryBitsPerDim, RotationKind kind) {
        this.matrix = matrix;
        this.bits = bitsPerDim;
        this.queryBits = queryBitsPerDim;
        this.kind = kind;
        // only used to encode, which depends on the bits per dimension and not on how the matrix was made
        this.quantizer = new AsymmetricHashingQuantizer(
            1.0f,
            Integer.MAX_VALUE,
            bitsPerDim,
            AsymmetricHashingQuantizer.Method.RANDOM,
            0,
            1,
            ENCODER_SEED,
            null
        );
        int dimension = matrix.originalDim();
        int latent = matrix.nDims();
        this.w = ESVectorUtil.transposeMatrix(matrix.wT(), latent, dimension);
    }

    /** Learns the projection matrix from a sample of the corpus the way a segment writer does, from its configuration. */
    public static AshDistortionRateQuantizer train(IvfSegmentConfig.AshConfig config, ClusteredSample sample, long seed)
        throws IOException {
        int dimension = sample.dimension();
        AsymmetricHashingQuantizer trainer = new AsymmetricHashingQuantizer(
            config.projectedDimsFraction(),
            config.maxProjectedDims(),
            config.bitsPerDim(),
            AsymmetricHashingQuantizer.Method.LEARNED,
            config.trainingIterations(),
            config.trainingFactor(),
            seed,
            null
        );
        var trained = trainer.train(sample::vector, sample.size(), dimension, sample::centroid);
        return new AshDistortionRateQuantizer(trained.matrix(), config.bitsPerDim(), config.queryBitsPerDim(), RotationKind.LEARNED);
    }

    /** A random orthogonal projection of the whole space, for which the law is exact. */
    public static AshDistortionRateQuantizer random(int dimension, int bitsPerDim, int queryBitsPerDim, long seed) throws IOException {
        AsymmetricHashingQuantizer trainer = new AsymmetricHashingQuantizer(
            1.0f,
            Integer.MAX_VALUE,
            bitsPerDim,
            AsymmetricHashingQuantizer.Method.RANDOM,
            0,
            1,
            seed,
            null
        );
        var trained = trainer.train(ord -> new float[dimension], 1, dimension, ord -> new float[dimension]);
        return new AshDistortionRateQuantizer(trained.matrix(), bitsPerDim, queryBitsPerDim, RotationKind.HAAR);
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
    public RotationKind rotation() {
        return kind;
    }

    @Override
    public void project(float[] vector, float[] latent) {
        ESVectorUtil.matrixVectorMultiply(matrix.wT(), matrix.nDims(), matrix.originalDim(), vector, latent);
    }

    @Override
    public float[] sampleResidual(Random random) {
        float[] direction = new float[matrix.nDims()];
        for (int i = 0; i < direction.length; i++) {
            direction[i] = (float) random.nextGaussian();
        }
        ESVectorUtil.l2Normalize(direction);
        float[] residual = new float[matrix.originalDim()];
        ESVectorUtil.matrixVectorMultiply(w, residual.length, direction.length, direction, residual);
        return residual;
    }

    @Override
    protected DocEncoder newEncoder(int maxBlockSize) {
        return newDocEncoder(maxBlockSize);
    }

    @Override
    protected DocDecoder decoder() {
        return AffineDocDecoder.INSTANCE;
    }

    /** The encoder of this quantizer, the one segment writers use with its projection. */
    public DocEncoder newDocEncoder(int maxBlockSize) {
        return new AshDocEncoder(matrix, quantizer, w, VectorSimilarityFunction.DOT_PRODUCT, bits, maxBlockSize);
    }

    @Override
    protected PreparedQuery prepareQuery(float[] query, float[] centroid) {
        // the query is the same for every list, whatever its centroid
        return new AshQueryEncoder(matrix, bits, queryBits, VectorSimilarityFunction.DOT_PRODUCT, ordinal -> centroid).prepare(query);
    }
}
