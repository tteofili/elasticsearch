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
import org.elasticsearch.simdvec.ScoringQuery;

import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

import static org.elasticsearch.index.codec.vectors.OptimizedScalarQuantizer.DEFAULT_LAMBDA;

/**
 * The query side of OSQ: the query is quantized against the centroid of each posting list that is visited, to the bits
 * of the query of its encoding, and the quantized queries of the last few lists are kept, as search comes back to them.
 */
public final class OsqQueryEncoder implements QueryEncoder {

    private static final int QUERY_CACHE_SIZE = 16;

    private final VectorSimilarityFunction similarity;
    private final QuantEncoding quantEncoding;
    private final int dimension;
    private final CentroidReader centroids;

    public OsqQueryEncoder(VectorSimilarityFunction similarity, QuantEncoding quantEncoding, int dimension, CentroidReader centroids) {
        this.similarity = similarity;
        this.quantEncoding = quantEncoding;
        this.dimension = dimension;
        this.centroids = centroids;
    }

    @Override
    public int latentDimension() {
        return dimension;
    }

    @Override
    public Prepared prepare(float[] query) {
        return new Prepared(query);
    }

    private record QueryQuantizerResult(OptimizedScalarQuantizer.QuantizationResult queryCorrections, ScoringQuery quantizedTarget) {}

    /**
     * A query for a search. A visitor follows the lists in two steps, {@link #reset} when it moves to a list and
     * {@link #quantizeQueryIfNecessary} when it first has to score one, so that a list that turns out to have nothing to
     * score does not cost a quantization.
     */
    public final class Prepared implements PreparedQuery {
        private final LinkedHashMap<Integer, QueryQuantizerResult> cache;
        private final float[] target;
        private final float[] scratch;
        private final int[] quantizationScratch;
        private final OptimizedScalarQuantizer quantizer;

        private int currentCentroidOrdinal = -2;
        private int nextCentroidOrdinal = -1;
        private byte[] evictedQuantizedQuery = null;
        private QueryQuantizerResult result = null;
        // what scoring and characterization read of the current list, quantized once for it
        private int scoredCentroidOrdinal = -2;
        private OptimizedScalarQuantizer.QuantizationResult scoredCorrections;
        private int[] scoredLevels;

        private Prepared(float[] target) {
            this.target = target;
            this.scratch = new float[dimension];
            this.quantizationScratch = new int[quantEncoding.discretizedDimensions(dimension)];
            this.quantizer = new OptimizedScalarQuantizer(similarity, DEFAULT_LAMBDA, 1);

            this.cache = new LinkedHashMap<>(QUERY_CACHE_SIZE, 0.75f, true) {
                @Override
                protected boolean removeEldestEntry(Map.Entry<Integer, QueryQuantizerResult> eldest) {
                    if (size() > QUERY_CACHE_SIZE) {
                        evictedQuantizedQuery = eldest.getValue().quantizedTarget().bytes();
                        return true;
                    }
                    return false;
                }
            };
        }

        public void reset(int centroidOrdinal) {
            this.nextCentroidOrdinal = centroidOrdinal;
        }

        public void quantizeQueryIfNecessary() throws IOException {
            if (nextCentroidOrdinal != currentCentroidOrdinal) {
                var quantized = cache.get(nextCentroidOrdinal);
                if (quantized != null) {
                    result = quantized;
                    currentCentroidOrdinal = nextCentroidOrdinal;
                    return;
                }
                // reuse the evicted byte array to reduce allocations
                final byte[] quantizedQuery = Objects.requireNonNullElseGet(
                    evictedQuantizedQuery,
                    () -> new byte[quantEncoding.getQueryPackedLength(target.length)]
                );
                OptimizedScalarQuantizer.QuantizationResult queryCorrections = quantize(nextCentroidOrdinal);
                quantEncoding.packQuery(quantizationScratch, quantizedQuery);
                currentCentroidOrdinal = nextCentroidOrdinal;
                result = new QueryQuantizerResult(queryCorrections, new ScoringQuery(quantizedQuery));
                cache.put(nextCentroidOrdinal, result);
            }
        }

        /** Quantizes the query against the centroid with this ordinal, leaving the levels in the scratch. */
        private OptimizedScalarQuantizer.QuantizationResult quantize(int centroidOrdinal) throws IOException {
            return quantizer.scalarQuantize(
                target,
                scratch,
                quantizationScratch,
                quantEncoding.queryBits(),
                centroids.read(centroidOrdinal)
            );
        }

        public OptimizedScalarQuantizer.QuantizationResult getQueryCorrections() {
            return result.queryCorrections();
        }

        public ScoringQuery getQuantizedTarget() {
            return result.quantizedTarget();
        }

        @Override
        public void forList(int centroidOrdinal) throws IOException {
            reset(centroidOrdinal);
            quantizeQueryIfNecessary();
        }

        @Override
        public void exactLatent(float[] latent) throws IOException {
            float[] centroid = centroids.read(currentCentroidOrdinal);
            for (int d = 0; d < dimension; d++) {
                latent[d] = target[d] - centroid[d];
            }
        }

        /** Quantizes the query for the current list, unless that is already done, and keeps what it made. */
        private OptimizedScalarQuantizer.QuantizationResult scored() throws IOException {
            if (scoredCentroidOrdinal != currentCentroidOrdinal) {
                scoredCorrections = quantize(currentCentroidOrdinal);
                if (scoredLevels == null) {
                    scoredLevels = new int[dimension];
                }
                System.arraycopy(quantizationScratch, 0, scoredLevels, 0, dimension);
                scoredCentroidOrdinal = currentCentroidOrdinal;
            }
            return scoredCorrections;
        }

        @Override
        public QueryScalars scalars() throws IOException {
            float[] centroid = centroids.read(currentCentroidOrdinal);
            OptimizedScalarQuantizer.QuantizationResult qr = scored();
            float queryScale = 1f / ((1 << quantEncoding.queryBits()) - 1);
            float ay = qr.lowerInterval();
            float ly = (qr.upperInterval() - ay) * queryScale;
            float y1 = qr.quantizedComponentSum();
            // for the Euclidean distance the estimate is a squared distance, addQ + addDoc - 2 * dot, so the factor -2 goes to
            // the query scalars and what is stored for a vector stays independent of the query
            float sign = similarity == VectorSimilarityFunction.EUCLIDEAN ? -2f : 1f;
            float omega = similarity == VectorSimilarityFunction.EUCLIDEAN
                ? qr.additionalCorrection()
                : qr.additionalCorrection() - ESVectorUtil.dotProduct(centroid, centroid);
            return new QueryScalars(sign * ly, sign * ay, 0f, sign * (ay * dimension + ly * y1), omega);
        }

        @Override
        public float rawDot(int[] levels) throws IOException {
            scored();
            long raw = 0;
            for (int d = 0; d < dimension; d++) {
                raw += (long) levels[d] * scoredLevels[d];
            }
            return raw;
        }

        @Override
        public void servedLatent(float[] latent) throws IOException {
            OptimizedScalarQuantizer.QuantizationResult corrections = scored();
            float step = (corrections.upperInterval() - corrections.lowerInterval()) / ((1 << quantEncoding.queryBits()) - 1);
            for (int d = 0; d < dimension; d++) {
                latent[d] = corrections.lowerInterval() + step * scoredLevels[d];
            }
        }
    }
}
