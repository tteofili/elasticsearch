/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq.next.calibrate;

import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;

import java.io.IOException;

/**
 * Manifold model for distance/similarity as a function of rank and corpus size.
 * Fits a log-linear model: log(distance at rank k) ~ alpha + invDim * (log(k) - log(N)).
 * Used in calibration to predict expected distances and compute expected recall@k.
 */
public final class ManifoldModel {

    /**
     * multipliers for the manifold rank sweep:
     * rank = floor(multiplier * k / 5), multipliers descending 29..5.
     */
    private static final int[] RANK_MULTIPLIERS = {
        29,
        28,
        27,
        26,
        25,
        24,
        23,
        22,
        21,
        20,
        19,
        18,
        17,
        16,
        15,
        14,
        13,
        12,
        11,
        10,
        9,
        8,
        7,
        6,
        5 };

    private static final int[] SAMPLE_SIZES = {
        4096,
        4608,
        5120,
        5632,
        6144,
        6656,
        7168,
        7680,
        8192,
        8704,
        9216,
        9728,
        10240,
        10752,
        11264,
        11776,
        12288,
        12800,
        13312,
        13824,
        14336,
        14848,
        15360,
        15872,
        16384 };

    static final int[] RANKS_FOR_K = { 29, 25, 21, 17, 13, 9, 7, 5 };

    static final int[] SAMPLE_SIZES_FAST = { 4096, 5632, 7168, 8704, 10240, 11776, 13312, 16384 };

    private ManifoldModel() {}

    /**
     * Estimate manifold parameters (alpha, invDim) using default (full) sample sizes.
     */
    public static double[] estimateManifoldParameters(
        VectorSimilarityFunction similarityFunction,
        int dim,
        float[][] queries,
        FloatVectorValues fvv,
        int[] corpusOrdinals,
        boolean cosine,
        int k
    ) throws IOException {
        return estimateManifoldParameters(similarityFunction, dim, queries, fvv, corpusOrdinals, cosine, k, ranksForK(k), SAMPLE_SIZES);
    }

    /**
     * Estimate manifold parameters (alpha, invDim) using reduced sample sizes for faster execution.
     */
    public static double[] estimateManifoldParametersFast(
        VectorSimilarityFunction similarityFunction,
        int dim,
        float[][] queries,
        FloatVectorValues fvv,
        int[] corpusOrdinals,
        boolean cosine,
        int k
    ) throws IOException {
        return estimateManifoldParameters(
            similarityFunction,
            dim,
            queries,
            fvv,
            corpusOrdinals,
            cosine,
            k,
            ranksForKFast(k),
            SAMPLE_SIZES_FAST
        );
    }

    static int[] ranksForK(int k) {
        return ranksFromMultipliers(RANK_MULTIPLIERS, k);
    }

    static int[] ranksForKFast(int k) {
        return ranksFromMultipliers(RANKS_FOR_K, k);
    }

    private static int[] ranksFromMultipliers(int[] multipliers, int k) {
        int[] ranks = new int[multipliers.length];
        for (int i = 0; i < multipliers.length; i++) {
            ranks[i] = Math.max(1, (multipliers[i] * k) / 5);
        }
        return ranks;
    }

    /**
     * Estimate manifold parameters (alpha, invDim) from query-corpus distances at various
     * ranks and sample sizes. Corpus vectors are accessed lazily via {@code fvv} and
     * {@code corpusOrdinals} rather than materializing a {@code float[][]}.
     *
     * @param fvv the underlying vector values source
     * @param corpusOrdinals ordinal indices into {@code fvv} for the corpus subset
     * @param cosine if true, corpus vectors are L2-normalized into a scratch buffer before distance computation
     * @param ranksForK the rank values to sweep
     * @param sampleSizes the corpus sample sizes to sweep (must be same length as ranksForK)
     * @return double[2] containing {alpha, invDim}
     */
    static double[] estimateManifoldParameters(
        VectorSimilarityFunction similarityFunction,
        int dim,
        float[][] queries,
        FloatVectorValues fvv,
        int[] corpusOrdinals,
        boolean cosine,
        int k,
        int[] ranksForK,
        int[] sampleSizes
    ) throws IOException {
        int nQueries = queries.length;
        int nDocsTotal = corpusOrdinals.length;
        int m = Math.min(ranksForK.length, sampleSizes.length);

        int logCount = 0;
        double[] logRanks = new double[m];
        double[] logSampleSizes = new double[m];
        double[] logDistances = new double[m];

        float[] scratch = cosine ? new float[dim] : null;
        int sampleStart = 0;
        for (int i = 0; i < m; i++) {
            int rank = ranksForK[i];
            int sampleEnd = sampleSizes[i];
            if (sampleEnd > nDocsTotal) break;
            double avgDist = 0;
            for (float[] query : queries) {
                double d = ithDistance(similarityFunction, dim, rank, query, fvv, corpusOrdinals, sampleStart, sampleEnd, cosine, scratch);
                avgDist += d;
            }
            avgDist /= nQueries;
            logRanks[logCount] = Math.log(rank);
            logSampleSizes[logCount] = Math.log(sampleSizes[i]);
            logDistances[logCount] = Math.log(Math.max(avgDist, 1e-38));
            logCount++;
            sampleStart = sampleEnd;
        }
        if (logCount < 2) {
            return new double[] { 0, 0 };
        }
        double[] x = new double[logCount];
        for (int i = 0; i < logCount; i++) {
            x[i] = logRanks[i] - logSampleSizes[i];
        }
        double[] y = new double[logCount];
        System.arraycopy(logDistances, 0, y, 0, logCount);
        Regression.OLSResult res = Regression.fitOls(x, y);
        return new double[] { res.beta0(), res.beta1() };
    }

    private static double ithDistance(
        VectorSimilarityFunction similarityFunction,
        int dim,
        int rank,
        float[] query,
        FloatVectorValues fvv,
        int[] corpusOrdinals,
        int start,
        int end,
        boolean cosine,
        float[] scratch
    ) throws IOException {
        int count = end - start;
        double[] dists = new double[count];
        boolean isDot = isDotLike(similarityFunction);
        for (int i = 0; i < count; i++) {
            float[] doc = fvv.vectorValue(corpusOrdinals[start + i]);
            if (cosine) {
                doc = CalibrationUtils.copyAndNormalize(doc, scratch);
            }
            if (isDot) {
                dists[i] = -CalibrationUtils.dot(dim, query, doc);
            } else {
                dists[i] = CalibrationUtils.euclideanSq(dim, query, doc);
            }
        }
        int idx = Math.min(rank - 1, dists.length - 1);
        if (idx < 0) return 0;
        quickSelect(dists, 0, dists.length - 1, idx);
        return isDot ? -dists[idx] : dists[idx];
    }

    /**
     * Partition-based selection (Hoare's quickselect) that rearranges {@code a} so that
     * {@code a[k]} holds the value that would be at index {@code k} in a sorted array.
     * Average O(n), worst-case O(n^2) but median-of-3 pivot makes worst case unlikely.
     */
    private static void quickSelect(double[] a, int lo, int hi, int k) {
        while (lo < hi) {
            int pivotIdx = medianOfThree(a, lo, lo + (hi - lo) / 2, hi);
            double pivot = a[pivotIdx];
            a[pivotIdx] = a[hi];
            a[hi] = pivot;

            int store = lo;
            for (int i = lo; i < hi; i++) {
                if (a[i] < pivot) {
                    double tmp = a[store];
                    a[store] = a[i];
                    a[i] = tmp;
                    store++;
                }
            }
            a[hi] = a[store];
            a[store] = pivot;

            if (store == k) {
                return;
            } else if (k < store) {
                hi = store - 1;
            } else {
                lo = store + 1;
            }
        }
    }

    private static int medianOfThree(double[] a, int i, int j, int k) {
        if (a[i] > a[j]) {
            if (a[j] > a[k]) return j;
            return a[i] > a[k] ? k : i;
        } else {
            if (a[i] > a[k]) return i;
            return a[j] > a[k] ? k : j;
        }
    }

    /**
     * Expected distance/similarity value at rank k in a corpus of size N from the manifold model.
     */
    public static double expectedRankDistance(VectorSimilarityFunction similarityFunction, double alpha, double invDim, int N, int k) {
        double logK = Math.log(k);
        double logN = Math.log(N);
        if (isDotLike(similarityFunction)) {
            return -Math.exp(alpha + (logK - logN) * invDim);
        }
        return Math.exp(alpha + (logK - logN) * invDim);
    }

    static boolean isDotLike(VectorSimilarityFunction similarityFunction) {
        return similarityFunction == VectorSimilarityFunction.DOT_PRODUCT
            || similarityFunction == VectorSimilarityFunction.COSINE
            || similarityFunction == VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT;
    }
}
