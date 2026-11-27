/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq.quantize;

/**
 * Quantizes the vectors of a posting list, a block at a time, whatever the scheme: from vectors that share a centroid to
 * a level for each coordinate of the latent space and the {@link DocLanes} that go with them.
 * <p>
 * It is the seam between the schemes (OSQ, ASH, and the ones to come) and what is common to them: laying the codes out,
 * writing them, and characterizing the quantizer, because everything the quantizer does to a vector is in what it
 * returns. Packing the levels into bytes and laying out a block on disk are separate, as they depend on the format and
 * not on the scheme.
 * <p>
 * An encoder is not thread safe: it holds the block, and scratch for it.
 */
public interface DocEncoder extends AutoCloseable {

    /** The dimension of the vectors. */
    int dimension();

    /** The dimension of the space the levels are in, at most {@link #dimension()}; smaller when the scheme projects. */
    int latentDimension();

    int bits();

    /**
     * The number of symbols in the code of a vector: one for each coordinate of the latent space for the schemes that
     * quantize coordinates (the code is then a level in {@code [0, 2^bits)} for each), fewer for the ones that quantize
     * blocks of coordinates to an entry of a codebook.
     */
    int codeLength();

    /** The most vectors a block can hold. */
    int maxBlockSize();

    /** Starts a block of vectors that belong to the posting list of {@code centroid}. */
    void reset(float[] centroid);

    /** Adds a vector to the block. The array is the caller's again when this returns. */
    void add(float[] vector);

    /** Quantizes the vectors that were added. */
    void encode();

    /** The number of vectors in the block. */
    int size();

    /**
     * Reads what was made of the {@code i}-th vector.
     *
     * @param levels receives the {@link #codeLength()} symbols of the code; for a scheme that quantizes coordinates, the
     *               level in {@code [0, 2^bits)} of each of the {@link #latentDimension()} coordinates
     * @param lanes receives the corrections
     */
    void read(int i, int[] levels, DocLanes lanes);

    /** The number of bytes of the packed code of a vector. */
    int packedLength();

    /**
     * Packs the code of the {@code i}-th vector into {@code destination}, starting at {@code offset}, in the layout that the
     * scheme's scorers read. What a layout writer does with a code, whatever the scheme.
     */
    void pack(int i, byte[] destination, int offset);

    /** The number of corrections the scheme stores beside the code of each vector. */
    int nativeLaneCount();

    /**
     * Writes the corrections of the {@code i}-th vector, as the scheme stores them and in the order it stores them: the
     * bits of a float for a float lane, the value of an int lane. Not {@link DocLanes}, which is the same for every
     * scheme but not lossless: a writer that has to reproduce what is on disk needs these.
     *
     * @param bits receives {@link #nativeLaneCount()} lanes
     */
    void nativeLanes(int i, int[] bits);

    /**
     * Writes the corrections of the {@code i}-th vector in the form that every scheme scores them in, see
     * {@link CanonicalScorer}: what scoring needs of the similarity of the field included.
     */
    void scoreLanes(int i, ScoreLanes lanes);

    /**
     * Writes into {@code latent} the vector that the {@code i}-th one was quantized from, in the latent space and without its
     * centroid, up to a positive factor: its direction, which is what quantizing keeps.
     */
    void latent(int i, float[] latent);

    @Override
    void close();
}
