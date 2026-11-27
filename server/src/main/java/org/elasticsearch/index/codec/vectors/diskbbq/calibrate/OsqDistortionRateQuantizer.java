/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq.calibrate;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.elasticsearch.index.codec.vectors.OptimizedScalarQuantizer;
import org.elasticsearch.index.codec.vectors.diskbbq.Preconditioner;
import org.elasticsearch.index.codec.vectors.diskbbq.QuantEncoding;
import org.elasticsearch.index.codec.vectors.diskbbq.next.ESNextDiskBBQVectorsFormat;
import org.elasticsearch.index.codec.vectors.diskbbq.quantize.AffineDocDecoder;
import org.elasticsearch.index.codec.vectors.diskbbq.quantize.DocDecoder;
import org.elasticsearch.index.codec.vectors.diskbbq.quantize.DocEncoder;
import org.elasticsearch.index.codec.vectors.diskbbq.quantize.OsqDocEncoder;
import org.elasticsearch.index.codec.vectors.diskbbq.quantize.OsqQueryEncoder;
import org.elasticsearch.index.codec.vectors.diskbbq.quantize.PreparedQuery;

import java.util.Random;

/**
 * OSQ, optimized scalar quantization, as a {@link DistortionRateQuantizer}. The rotation is applied in the index by the
 * preconditioner, or not at all, and the quantizer itself works in the rotated space, so the latent space is the whole
 * space. Its gain is not the residual's norm but the interval OSQ fits to each vector, controlled by lambda: after a
 * rotation that makes the interval symmetric it is, in closed form, {@code C / (lambda + (1 - lambda) C^2)} times the
 * norm, where {@code C} is the cosine between the direction and its code.
 */
public final class OsqDistortionRateQuantizer extends EncoderDistortionRateQuantizer {

    /** Applies a rotation, which must be orthogonal. */
    @FunctionalInterface
    public interface Rotation {
        void apply(float[] in, float[] out);
    }

    /** The number of coordinate-descent iterations of {@link OptimizedScalarQuantizer}'s default. */
    private static final int ITERATIONS = 5;

    private final int dimension;
    private final int bits;
    private final int queryBits;
    private final float lambda;
    private final RotationKind kind;
    private final Rotation rotation;

    private final QuantEncoding queryEncoding;
    // the query and the centroid of the list as the quantizer sees them, rotated
    private final float[] rotatedQuery;
    private final float[] rotatedCentroid;

    public OsqDistortionRateQuantizer(int dimension, int bits, int queryBits, float lambda, RotationKind kind, Rotation rotation) {
        this.dimension = dimension;
        this.bits = bits;
        this.queryBits = queryBits;
        this.lambda = lambda;
        this.kind = kind;
        this.rotation = rotation;
        this.queryEncoding = encodingOf(bits, queryBits);
        this.rotatedQuery = new float[dimension];
        this.rotatedCentroid = new float[dimension];
    }

    /** The quantizer an OSQ field of the given encoding is written with, with the preconditioner on or off. */
    public static OsqDistortionRateQuantizer forIndex(int dimension, QuantEncoding encoding, boolean precondition) {
        if (precondition) {
            Preconditioner preconditioner = Preconditioner.createPreconditioner(
                dimension,
                ESNextDiskBBQVectorsFormat.DEFAULT_PRECONDITIONING_BLOCK_DIMENSION
            );
            return new OsqDistortionRateQuantizer(
                dimension,
                encoding.bits(),
                encoding.queryBits(),
                OptimizedScalarQuantizer.DEFAULT_LAMBDA,
                RotationKind.BLOCK_DIAGONAL,
                preconditioner::applyTransform
            );
        }
        return new OsqDistortionRateQuantizer(
            dimension,
            encoding.bits(),
            encoding.queryBits(),
            OptimizedScalarQuantizer.DEFAULT_LAMBDA,
            RotationKind.NONE,
            (in, out) -> System.arraycopy(in, 0, out, 0, in.length)
        );
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
    public RotationKind rotation() {
        return kind;
    }

    @Override
    public void project(float[] vector, float[] latent) {
        rotation.apply(vector, latent);
    }

    @Override
    public float[] sampleResidual(Random random) {
        float[] v = new float[dimension];
        double norm = 0;
        for (int i = 0; i < dimension; i++) {
            v[i] = (float) random.nextGaussian();
            norm += v[i] * v[i];
        }
        float inverse = (float) (1 / Math.sqrt(norm));
        for (int i = 0; i < dimension; i++) {
            v[i] *= inverse;
        }
        return v;
    }

    @Override
    protected DocEncoder newEncoder(int maxBlockSize) {
        return newDocEncoder(maxBlockSize);
    }

    @Override
    protected DocDecoder decoder() {
        return AffineDocDecoder.INSTANCE;
    }

    /** The encoder of this quantizer, which rotates what it is given as the quantizer does. */
    public DocEncoder newDocEncoder(int maxBlockSize) {
        return new OsqDocEncoder(
            new OptimizedScalarQuantizer(VectorSimilarityFunction.DOT_PRODUCT, lambda, ITERATIONS),
            VectorSimilarityFunction.DOT_PRODUCT,
            dimension,
            bits,
            maxBlockSize,
            rotation::apply
        );
    }

    @Override
    protected PreparedQuery prepareQuery(float[] query, float[] centroid) {
        rotation.apply(query, rotatedQuery);
        rotation.apply(centroid, rotatedCentroid);
        return new OsqQueryEncoder(VectorSimilarityFunction.DOT_PRODUCT, queryEncoding, dimension, ordinal -> rotatedCentroid).prepare(
            rotatedQuery
        );
    }

    private static QuantEncoding encodingOf(int bits, int queryBits) {
        for (QuantEncoding encoding : QuantEncoding.values()) {
            if (encoding.bits() == bits && encoding.queryBits() == queryBits) {
                return encoding;
            }
        }
        throw new IllegalArgumentException("no OSQ encoding with " + bits + " bits for documents and " + queryBits + " for queries");
    }
}
