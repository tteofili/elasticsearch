/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq.calibrate;

import org.apache.lucene.index.FloatVectorValues;
import org.elasticsearch.simdvec.ESVectorUtil;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Predicts the recall of a quantizer without quantizing the corpus: Algorithm 2 of "Configuring Project-then-Quantize
 * Embedding Compression", with the codebook entering through a {@link CodebookProfile} and the quantizer through
 * {@link DistortionRateQuantizer}.
 * <p>
 * For each sample query the candidates that can reach the top after compression are the ones with the highest exact
 * scores, its {@link Pool}. Their compressed scores are drawn from the law of the paper's Lemma 3,
 * {@code centroidScore + gain * (alpha * A + r * B * xi)} with {@code (A, B)} from the codebook and {@code xi} one
 * coordinate of a uniform point of the sphere, and, if the query is quantized, an independent term for the error of the
 * query. The share of the true nearest neighbors among the top {@code eta * k} compressed scores is the recall after
 * reranking.
 * <p>
 * The estimate is only to be trusted when the quantizer's rotation passes {@link IsotropyCheck}. It assumes candidates'
 * errors are independent and ignores candidates outside the pool, which can only make it optimistic.
 */
public final class RecallEstimator {

    public static final int DEFAULT_DRAWS = 32;

    private RecallEstimator() {}

    /**
     * The candidates of one query, in no particular order.
     *
     * @param centroidScore {@code <query, centroid>} of each candidate's posting list, exact
     * @param gain the norm of each candidate's residual
     * @param alpha the component of the latent query along the latent direction of the candidate
     * @param r the norm of the rest of the latent query
     * @param isTrue whether the candidate is among the true nearest neighbors of the query in the whole corpus
     * @param queryError the norm of the quantization error of the query for the candidate's posting list, or {@code null}
     *                   when the query is not quantized
     */
    public record Pool(float[] centroidScore, float[] gain, float[] alpha, float[] r, boolean[] isTrue, float[] queryError) {
        public int size() {
            return gain.length;
        }
    }

    /**
     * Builds the pools from exact scores, which is the estimator's only pass over the corpus.
     * <p>
     * The corpus is read, never copied: every sampled vector is read once for the scan, which scores it against all
     * the queries, and once more if it lands in a pool. Only the queries, a few hundred at most, are held on heap.
     *
     * @param corpus the corpus sample, its landmarks and the landmark of each sampled vector
     * @param queries the vectors that hold the sample queries, and {@code queryOrdinals} the ones to use
     * @param k the number of nearest neighbors that count
     * @param poolSize how many candidates, with the highest exact scores, to keep per query
     * @param visitedClusters for an IVF search, which clusters each query visits (the pool then holds what they contain,
     *                        and true neighbors outside them are misses), or {@code null} for an exhaustive scan
     * @param quantizedQuery whether to include the error of quantizing each query for each candidate's posting list
     */
    public static List<Pool> pools(
        DistortionRateQuantizer quantizer,
        ClusteredSample corpus,
        FloatVectorValues queries,
        int[] queryOrdinals,
        int k,
        int poolSize,
        boolean[][] visitedClusters,
        boolean quantizedQuery
    ) throws IOException {
        int nQueries = queryOrdinals.length;
        int dimension = corpus.dimension();
        // the queries as one row-major matrix, so that a document is scored against all of them with one matrix-vector product
        float[] queryMatrix = new float[nQueries * dimension];
        float[][] projectedQuery = new float[nQueries][];
        double[] projectedNormSq = new double[nQueries];
        for (int q = 0; q < nQueries; q++) {
            float[] query = queries.vectorValue(queryOrdinals[q]);
            System.arraycopy(query, 0, queryMatrix, q * dimension, dimension);
            projectedQuery[q] = quantizer.project(query);
            projectedNormSq[q] = ESVectorUtil.dotProduct(projectedQuery[q], projectedQuery[q]);
        }
        Top[] truth = new Top[nQueries];
        Top[] pool = new Top[nQueries];
        for (int q = 0; q < nQueries; q++) {
            truth[q] = new Top(k);
            pool[q] = new Top(poolSize);
        }
        // one read of each sampled vector, scored against every query
        float[] scores = new float[nQueries];
        for (int i = 0; i < corpus.size(); i++) {
            ESVectorUtil.matrixVectorMultiply(queryMatrix, nQueries, dimension, corpus.vector(i), scores);
            for (int q = 0; q < nQueries; q++) {
                float score = scores[q];
                truth[q].offer(score, i);
                if (visitedClusters == null || visitedClusters[q][corpus.assignment()[i]]) {
                    pool[q].offer(score, i);
                }
            }
        }
        List<Pool> pools = new ArrayList<>(nQueries);
        float[] query = new float[dimension];
        float[] residual = new float[dimension];
        float[] latent = new float[quantizer.latentDimension()];
        for (int q = 0; q < nQueries; q++) {
            int[] ids = pool[q].ids();
            int m = ids.length;
            float[] centroidScore = new float[m];
            float[] gain = new float[m];
            float[] alpha = new float[m];
            float[] r = new float[m];
            boolean[] isTrue = new boolean[m];
            float[] queryError = quantizedQuery ? new float[m] : null;
            Map<Integer, Float> errors = new HashMap<>();
            int[] truthIds = truth[q].ids();
            System.arraycopy(queryMatrix, q * dimension, query, 0, dimension);
            for (int p = 0; p < m; p++) {
                int i = ids[p];
                float[] centroid = corpus.centroid(i);
                centroidScore[p] = ESVectorUtil.dotProduct(query, centroid);
                System.arraycopy(corpus.vector(i), 0, residual, 0, dimension);
                ESVectorUtil.linearCombination(-1f, centroid, 1f, residual);
                float g = (float) Math.sqrt(ESVectorUtil.dotProduct(residual, residual));
                gain[p] = g;
                if (g > 0) {
                    // alpha does not depend on the scale of the latent direction, so the residual is not normalized
                    quantizer.project(residual, latent);
                    double rho = Math.sqrt(ESVectorUtil.dotProduct(latent, latent));
                    if (rho > 0) {
                        alpha[p] = (float) (ESVectorUtil.dotProduct(projectedQuery[q], latent) / rho);
                    }
                }
                r[p] = (float) Math.sqrt(Math.max(0, projectedNormSq[q] - (double) alpha[p] * alpha[p]));
                for (int t : truthIds) {
                    if (t == i) {
                        isTrue[p] = true;
                    }
                }
                if (quantizedQuery) {
                    int cluster = corpus.assignment()[i];
                    Float known = errors.get(cluster);
                    if (known == null) {
                        known = (float) quantizer.queryError(query, corpus.centroids().centroid(cluster));
                        errors.put(cluster, known);
                    }
                    queryError[p] = known;
                }
            }
            pools.add(new Pool(centroidScore, gain, alpha, r, isTrue, queryError));
        }
        return pools;
    }

    /**
     * Estimates recall@k after reranking the top {@code eta * k} compressed candidates.
     *
     * @param etas the oversampling factors to estimate for
     * @param latentDimension {@link DistortionRateQuantizer#latentDimension()}
     * @return the estimated recall for each of {@code etas}, averaged over the pools and {@code draws} draws
     */
    public static double[] estimate(
        CodebookProfile profile,
        List<Pool> pools,
        int[] etas,
        int k,
        int latentDimension,
        int draws,
        long seed
    ) {
        double[] recall = new double[etas.length];
        double sqrtD = Math.sqrt(latentDimension - 1.0);
        for (int q = 0; q < pools.size(); q++) {
            Pool pool = pools.get(q);
            Random random = new Random(seed + q);
            int m = pool.size();
            long[] keys = new long[m];
            double[] hits = new double[etas.length];
            for (int t = 0; t < draws; t++) {
                for (int p = 0; p < m; p++) {
                    int s = random.nextInt(profile.size());
                    double a = profile.a()[s];
                    double b = profile.b()[s];
                    double xi = random.nextGaussian() / sqrtD; // one coordinate of a uniform point of the sphere
                    double g = pool.gain()[p];
                    double score = pool.centroidScore()[p] + g * (pool.alpha()[p] * a + pool.r()[p] * b * xi);
                    if (pool.queryError() != null) {
                        score += g * Math.sqrt(a * a + b * b) * pool.queryError()[p] / Math.sqrt(latentDimension) * random.nextGaussian();
                    }
                    keys[p] = rankKey((float) score, p);
                }
                Arrays.sort(keys);
                for (int e = 0; e < etas.length; e++) {
                    int limit = Math.min(etas[e] * k, m);
                    for (int j = 0; j < limit; j++) {
                        if (pool.isTrue()[(int) keys[j]]) {
                            hits[e]++;
                        }
                    }
                }
            }
            for (int e = 0; e < etas.length; e++) {
                recall[e] += hits[e] / (draws * (double) k);
            }
        }
        for (int e = 0; e < etas.length; e++) {
            recall[e] /= pools.size();
        }
        return recall;
    }

    /** A key that sorts ascending from the highest score to the lowest, and carries {@code index} in its low bits. */
    private static long rankKey(float score, int index) {
        int bits = Float.floatToIntBits(score);
        int sortable = bits >= 0 ? bits : bits ^ 0x7FFFFFFF;
        return ((long) ~sortable << 32) | index;
    }

    /** The best {@code capacity} scores seen with their ids, which is a tiny pool of candidates next to a corpus. */
    private static final class Top {
        private final float[] scores;
        private final int[] ids;
        private int size;
        private int minAt;

        Top(int capacity) {
            scores = new float[capacity];
            ids = new int[capacity];
        }

        void offer(float score, int id) {
            if (size < scores.length) {
                scores[size] = score;
                ids[size++] = id;
                if (size == scores.length) {
                    findMin();
                }
            } else if (score > scores[minAt]) {
                scores[minAt] = score;
                ids[minAt] = id;
                findMin();
            }
        }

        private void findMin() {
            minAt = 0;
            for (int i = 1; i < size; i++) {
                if (scores[i] < scores[minAt]) {
                    minAt = i;
                }
            }
        }

        /** The ids, most similar first. */
        int[] ids() {
            long[] keys = new long[size];
            for (int i = 0; i < size; i++) {
                keys[i] = rankKey(scores[i], i);
            }
            Arrays.sort(keys);
            int[] out = new int[size];
            for (int i = 0; i < size; i++) {
                out[i] = ids[(int) keys[i]];
            }
            return out;
        }
    }
}
