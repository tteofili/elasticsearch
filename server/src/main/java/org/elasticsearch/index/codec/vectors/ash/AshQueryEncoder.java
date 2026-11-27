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
import org.elasticsearch.index.codec.vectors.diskbbq.quantize.CentroidReader;
import org.elasticsearch.index.codec.vectors.diskbbq.quantize.PreparedQuery;
import org.elasticsearch.index.codec.vectors.diskbbq.quantize.QueryEncoder;
import org.elasticsearch.index.codec.vectors.diskbbq.quantize.QueryScalars;
import org.elasticsearch.simdvec.ESVectorUtil;

import java.io.IOException;

/**
 * The query side of ASH: the query is projected into the latent space once, with the int8 product of
 * {@link AshProjectionMatrix#projectQuery}, and, for the integer scorer, quantized once to {@code queryBitsPerDim} bits
 * with a min-max grid. Unlike OSQ nothing is done to it per posting list: what changes from one list to the next is the
 * exact inner product with its centroid, which the visitor computes.
 */
public final class AshQueryEncoder implements QueryEncoder {

    private final AshProjectionMatrix matrix;
    private final int bitsPerDim;
    private final int queryBitsPerDim;
    private final VectorSimilarityFunction similarity;
    private final CentroidReader centroids;

    /**
     * @param bitsPerDim bits per latent dimension of the documents
     * @param queryBitsPerDim bits per latent dimension of the query, or 0 when the scorer takes the projected query as it is
     * @param centroids reads the centroid of a posting list, for what the query needs of it
     */
    public AshQueryEncoder(
        AshProjectionMatrix matrix,
        int bitsPerDim,
        int queryBitsPerDim,
        VectorSimilarityFunction similarity,
        CentroidReader centroids
    ) {
        this.matrix = matrix;
        this.bitsPerDim = bitsPerDim;
        this.queryBitsPerDim = queryBitsPerDim;
        this.similarity = similarity;
        this.centroids = centroids;
    }

    @Override
    public int latentDimension() {
        return matrix.nDims();
    }

    @Override
    public PreparedQuery prepare(float[] query) {
        return new Prepared(query);
    }

    /**
     * The min-max grid that a projected query is quantized to: level {@code l} stands for {@code min + l * inverse}.
     *
     * @param scale the factor from a value to a level, {@code 1} when the query is constant
     */
    record Grid(float min, float scale, float inverse) {}

    /**
     * Quantizes a projected query to {@code bits} bits on its min-max grid.
     *
     * @param levels receives the level of each of the {@code nDims} coordinates
     */
    static Grid quantizeToLevels(float[] projected, int nDims, int bits, int[] levels) {
        int numQueryLevels = 1 << bits;

        float qMin = Float.MAX_VALUE, qMax = -Float.MAX_VALUE;
        for (int j = 0; j < nDims; j++) {
            qMin = Math.min(qMin, projected[j]);
            qMax = Math.max(qMax, projected[j]);
        }
        float range = qMax - qMin;
        float qScale = range > 0 ? (numQueryLevels - 1) / range : 1.0f;
        float invQScale = range > 0 ? range / (numQueryLevels - 1) : 0f;

        for (int j = 0; j < nDims; j++) {
            levels[j] = Math.clamp(Math.round((projected[j] - qMin) * qScale), 0, numQueryLevels - 1);
        }
        return new Grid(qMin, qScale, invQScale);
    }

    private final class Prepared implements PreparedQuery {
        private final float[] query;
        // what the scorer starts from, once for the search: the int8 projection of the query, quantized on its grid
        private final float[] projected;
        private final int[] levels;
        private final Grid grid;
        private final float[] centeredCode;
        private int currentCentroidOrdinal = -1;

        private Prepared(float[] query) {
            this.query = query;
            this.projected = matrix.projectQuery(query);
            this.levels = new int[matrix.nDims()];
            this.centeredCode = new float[matrix.nDims()];
            this.grid = queryBitsPerDim > 0 ? quantizeToLevels(projected, projected.length, queryBitsPerDim, levels) : null;
        }

        @Override
        public void forList(int centroidOrdinal) {
            // the query is the same for every list: only what the scorer adds for the centroid of the list changes
            currentCentroidOrdinal = centroidOrdinal;
        }

        @Override
        public void exactLatent(float[] latent) {
            ESVectorUtil.matrixVectorMultiply(matrix.wT(), matrix.nDims(), matrix.originalDim(), query, latent);
        }

        @Override
        public void servedLatent(float[] latent) {
            for (int j = 0; j < projected.length; j++) {
                latent[j] = grid == null ? projected[j] : grid.min() + levels[j] * grid.inverse();
            }
        }

        @Override
        public QueryScalars scalars() throws IOException {
            float[] centroid = centroids.read(currentCentroidOrdinal);
            float queryDotCentroid = ESVectorUtil.dotProduct(query, centroid, query.length);
            float centroidNormSq = ESVectorUtil.dotProduct(centroid, centroid);
            QueryScalars dot;
            if (grid == null) {
                dot = new QueryScalars(1f, 0f, 0f, 0f, queryDotCentroid);
            } else {
                int unsignedSum = 0;
                for (int level : levels) {
                    unsignedSum += level;
                }
                float center = ((1 << bitsPerDim) - 1) / 2.0f;
                float constantCorrection = center * (unsignedSum * grid.inverse() + grid.min() * matrix.nDims());
                dot = new QueryScalars(grid.inverse(), grid.min(), -constantCorrection, 0f, queryDotCentroid);
            }
            if (similarity != VectorSimilarityFunction.EUCLIDEAN) {
                return dot;
            }
            // ||q-x||^2 = ||q-c||^2 + ||x-c||^2 - 2 (dot - <x,c> - <q,c> + ||c||^2): <q,c> cancels and the -2 moves to the query
            // scalars, so that what is stored for a vector stays independent of the query
            return new QueryScalars(
                -2 * dot.alpha(),
                -2 * dot.beta(),
                -2 * dot.gamma(),
                0f,
                ESVectorUtil.squareDistance(query, centroid) - 2 * centroidNormSq
            );
        }

        @Override
        public float rawDot(int[] docLevels) {
            int nDims = matrix.nDims();
            if (grid != null) {
                long raw = 0;
                for (int j = 0; j < nDims; j++) {
                    raw += (long) levels[j] * docLevels[j];
                }
                return raw;
            }
            // the float scorer takes the centered codes
            float center = ((1 << bitsPerDim) - 1) / 2.0f;
            for (int j = 0; j < nDims; j++) {
                centeredCode[j] = docLevels[j] - center;
            }
            return ESVectorUtil.dotProduct(projected, centeredCode, nDims);
        }
    }
}
