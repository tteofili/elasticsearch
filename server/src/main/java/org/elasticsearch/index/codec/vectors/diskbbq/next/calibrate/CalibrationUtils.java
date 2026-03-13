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

import java.io.IOException;
import java.util.Random;

/**
 * Utility methods for quantization calibration: vector math and sampling from
 * {@link FloatVectorValues}.
 */
public final class CalibrationUtils {

    static final int MAX_QUERY_SAMPLE = 128;
    static final int MAX_CORPUS_SAMPLE = 16384;
    static final int MAX_QUERY_SAMPLE_FAST = 48;
    static final int MAX_CORPUS_SAMPLE_FAST = 8192;
    static final long CALIBRATION_SEED = 215873873L;

    private CalibrationUtils() {}

    /**
     * Sampled data from a {@link FloatVectorValues}: materialized query vectors and
     * ordinal indices into the original {@code FloatVectorValues} for the corpus.
     */
    public record SampledData(float[][] queries, int[] corpusOrdinals) {}

    /**
     * Dot product of two float arrays of length {@code dim}.
     */
    public static double dot(int dim, float[] x, float[] y) {
        double sum = 0;
        for (int i = 0; i < dim; i++) {
            sum += (double) x[i] * y[i];
        }
        return sum;
    }

    /**
     * Squared Euclidean distance between two float arrays of length {@code dim}.
     */
    public static double euclideanSq(int dim, float[] x, float[] y) {
        double sum = 0;
        for (int i = 0; i < dim; i++) {
            double d = x[i] - y[i];
            sum += d * d;
        }
        return sum;
    }

    /**
     * L2-normalize each row of the matrix in place.
     */
    public static void normalize(float[][] vectors) {
        for (float[] v : vectors) {
            normalizeVector(v);
        }
    }

    /**
     * L2-normalize a single vector in place.
     */
    public static void normalizeVector(float[] v) {
        double norm = 0;
        for (float f : v) {
            norm += (double) f * f;
        }
        norm = Math.sqrt(norm);
        if (norm == 0) norm = 1;
        for (int j = 0; j < v.length; j++) {
            v[j] = (float) (v[j] / norm);
        }
    }

    /**
     * Copy a vector into a scratch buffer and L2-normalize the copy.
     * Returns the scratch buffer for convenience.
     */
    public static float[] copyAndNormalize(float[] src, float[] scratch) {
        System.arraycopy(src, 0, scratch, 0, src.length);
        normalizeVector(scratch);
        return scratch;
    }

    /**
     * Sample random, disjoint query and corpus subsets from {@link FloatVectorValues}
     * using default (full) sample sizes.
     */
    public static SampledData sampleData(FloatVectorValues vectorValues, int dim) throws IOException {
        return sampleData(vectorValues, dim, MAX_QUERY_SAMPLE, MAX_CORPUS_SAMPLE);
    }

    /**
     * Sample random, disjoint query and corpus subsets using reduced sample sizes
     * for faster calibration (e.g., during merge re-calibration).
     */
    public static SampledData sampleDataFast(FloatVectorValues vectorValues, int dim) throws IOException {
        return sampleData(vectorValues, dim, MAX_QUERY_SAMPLE_FAST, MAX_CORPUS_SAMPLE_FAST);
    }

    /**
     * Sample random, disjoint query and corpus subsets from {@link FloatVectorValues}.
     * Queries are materialized (cloned); corpus vectors are represented as ordinal indices
     * into the original {@code vectorValues}, avoiding bulk materialization.
     */
    static SampledData sampleData(FloatVectorValues vectorValues, int dim, int maxQuerySample, int maxCorpusSample) throws IOException {
        int n = vectorValues.size();
        Random rng = new Random(CALIBRATION_SEED);
        int nQueries = Math.min(maxQuerySample, n / 2);
        int nDocs = Math.min(maxCorpusSample, n - nQueries);

        int[] indices = new int[n];
        for (int i = 0; i < n; i++) {
            indices[i] = i;
        }
        fisherYatesShuffle(indices, rng);

        float[][] queries = new float[nQueries][];
        for (int i = 0; i < nQueries; i++) {
            queries[i] = vectorValues.vectorValue(indices[i]).clone();
        }
        int[] corpusOrdinals = new int[nDocs];
        System.arraycopy(indices, nQueries, corpusOrdinals, 0, nDocs);
        return new SampledData(queries, corpusOrdinals);
    }

    private static void fisherYatesShuffle(int[] a, Random rng) {
        for (int i = a.length - 1; i > 0; i--) {
            int j = rng.nextInt(i + 1);
            int t = a[i];
            a[i] = a[j];
            a[j] = t;
        }
    }
}
