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
 * Utility methods for quantization calibration: vector math, sampling from
 * {@link FloatVectorValues}, and the Neyshabur-Srebro transform.
 */
public final class CalibrationUtils {

    static final int MAX_QUERY_SAMPLE = 128;
    static final int MAX_CORPUS_SAMPLE = 16384;
    static final long CALIBRATION_SEED = 215873873L;

    private CalibrationUtils() {}

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
    }

    /**
     * Sample random, disjoint query and corpus subsets from {@link FloatVectorValues}.
     * Returns {@code float[][][2]} where index 0 is queries and index 1 is corpus.
     */
    public static float[][][] sampleQueriesAndCorpus(FloatVectorValues vectorValues, int dim) throws IOException {
        int n = vectorValues.size();
        Random rng = new Random(CALIBRATION_SEED);
        int nQueries = Math.min(MAX_QUERY_SAMPLE, n / 2);
        int nDocs = Math.min(MAX_CORPUS_SAMPLE, n - nQueries);

        int[] indices = new int[n];
        for (int i = 0; i < n; i++) {
            indices[i] = i;
        }
        fisherYatesShuffle(indices, rng);

        float[][] queries = new float[nQueries][];
        for (int i = 0; i < nQueries; i++) {
            queries[i] = vectorValues.vectorValue(indices[i]).clone();
        }
        float[][] corpus = new float[nDocs][];
        for (int i = 0; i < nDocs; i++) {
            corpus[i] = vectorValues.vectorValue(indices[nQueries + i]).clone();
        }
        return new float[][][] { queries, corpus };
    }

    /**
     * Sample random, disjoint query and corpus subsets from {@link FloatVectorValues}.
     * Returns {@code float[][][2]} where index 0 is queries and index 1 is corpus.
     */
    public static int[] sampleQueryIndices(FloatVectorValues vectorValues, int dim) throws IOException {
        int n = vectorValues.size();
        Random rng = new Random(CALIBRATION_SEED);
        int nQueries = Math.min(MAX_QUERY_SAMPLE, n / 2);
        int nDocs = Math.min(MAX_CORPUS_SAMPLE, n - nQueries);

        int[] indices = new int[n];
        for (int i = 0; i < n; i++) {
            indices[i] = i;
        }
        fisherYatesShuffle(indices, rng);

        return indices;
    }

    /**
     * Neyshabur-Srebro transform: converts dot-product similarity to Euclidean distance
     * by appending an extra coordinate. Returns {@code float[][][2]} where index 0 is
     * transformed queries and index 1 is transformed corpus; both have dimension dim+1.
     */
    public static float[][][] neyshaburSrebroTransform(int dim, float[][] queries, float[][] corpus) {
        double maxNormSq = 0;
        for (float[] v : corpus) {
            double normSq = 0;
            for (int j = 0; j < dim; j++) {
                normSq += (double) v[j] * v[j];
            }
            if (normSq > maxNormSq) maxNormSq = normSq;
        }
        int newDim = dim + 1;
        float[][] tq = new float[queries.length][newDim];
        for (int i = 0; i < queries.length; i++) {
            System.arraycopy(queries[i], 0, tq[i], 0, dim);
            tq[i][dim] = 0f;
        }
        float[][] tc = new float[corpus.length][newDim];
        for (int i = 0; i < corpus.length; i++) {
            double normSq = 0;
            for (int j = 0; j < dim; j++) {
                normSq += (double) corpus[i][j] * corpus[i][j];
            }
            System.arraycopy(corpus[i], 0, tc[i], 0, dim);
            tc[i][dim] = (float) Math.sqrt(Math.max(0, maxNormSq - normSq));
        }
        return new float[][][] { tq, tc };
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
