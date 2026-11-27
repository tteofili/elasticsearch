/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq.calibrate;

import org.elasticsearch.index.codec.vectors.diskbbq.quantize.DocDecoder;
import org.elasticsearch.index.codec.vectors.diskbbq.quantize.DocEncoder;
import org.elasticsearch.index.codec.vectors.diskbbq.quantize.DocLanes;
import org.elasticsearch.index.codec.vectors.diskbbq.quantize.PreparedQuery;
import org.elasticsearch.simdvec.ESVectorUtil;

import java.io.IOException;
import java.io.UncheckedIOException;

/**
 * A {@link DistortionRateQuantizer} whose reconstruction is what its {@link DocEncoder} and {@link DocDecoder} make of
 * residuals, so that what is characterized is what a segment writer writes and what a scorer reads, for any scheme that has
 * an encoder and a decoder, and cannot drift from them.
 */
public abstract class EncoderDistortionRateQuantizer implements DistortionRateQuantizer {

    private static final int BLOCK = 256;

    /** A new encoder, which this class closes, for blocks of at most {@code maxBlockSize} vectors. */
    protected abstract DocEncoder newEncoder(int maxBlockSize);

    protected abstract DocDecoder decoder();

    /**
     * Prepares {@code query} for scoring the posting list of {@code centroid}, with the query encoder that serves queries
     * for the scheme. Both are in the space of the original vectors, so a scheme that rotates them does it here.
     */
    protected abstract PreparedQuery prepareQuery(float[] query, float[] centroid) throws IOException;

    // scratch of queryError, which the estimator calls once per query and posting list; makes this class unsafe to share
    // between threads
    private float[] exactQuery;
    private float[] servedQuery;

    /**
     * The distance between the query the scorer would like to use and the one it ends up using, as the query encoder makes
     * it, so that it is the error of the encoder that serves queries.
     */
    @Override
    public final double queryError(float[] query, float[] centroid) {
        if (exactQuery == null) {
            exactQuery = new float[latentDimension()];
            servedQuery = new float[latentDimension()];
        }
        try {
            PreparedQuery prepared = prepareQuery(query, centroid);
            prepared.forList(0);
            prepared.exactLatent(exactQuery);
            prepared.servedLatent(servedQuery);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        ESVectorUtil.linearCombination(-1f, servedQuery, 1f, exactQuery);
        return Math.sqrt(ESVectorUtil.dotProduct(exactQuery, exactQuery));
    }

    @Override
    public final Reconstruction[] reconstruct(Residuals residuals) throws IOException {
        int n = residuals.size();
        int nDims = latentDimension();
        Reconstruction[] out = new Reconstruction[n];
        float[] noCentroid = new float[dimension()];
        DocDecoder decoder = decoder();
        try (DocEncoder encoder = newEncoder(BLOCK)) {
            int[] indices = new int[BLOCK];
            double[] gains = new double[BLOCK];
            int[] levels = new int[encoder.codeLength()];
            DocLanes lanes = new DocLanes();
            float[] direction = new float[nDims];
            float[] reconstruction = new float[nDims];
            for (int start = 0; start < n; start += BLOCK) {
                int end = Math.min(n, start + BLOCK);
                int count = 0;
                encoder.reset(noCentroid);
                for (int i = start; i < end; i++) {
                    float[] residual = residuals.residual(i);
                    double normSq = ESVectorUtil.dotProduct(residual, residual);
                    if (normSq == 0) {
                        out[i] = Reconstruction.EMPTY;
                    } else {
                        encoder.add(residual);
                        gains[count] = Math.sqrt(normSq);
                        indices[count++] = i;
                    }
                }
                if (count == 0) {
                    continue;
                }
                encoder.encode();
                for (int j = 0; j < count; j++) {
                    encoder.read(j, levels, lanes);
                    decoder.decode(levels, lanes, reconstruction);
                    encoder.latent(j, direction);
                    double directionNorm = Math.sqrt(ESVectorUtil.dotProduct(direction, direction));
                    double gain = gains[j];
                    double a = ESVectorUtil.dotProduct(direction, reconstruction) / (directionNorm * gain);
                    double normSq = ESVectorUtil.dotProduct(reconstruction, reconstruction) / (gain * gain);
                    out[indices[j]] = new Reconstruction(a, normSq);
                }
            }
        }
        return out;
    }
}
