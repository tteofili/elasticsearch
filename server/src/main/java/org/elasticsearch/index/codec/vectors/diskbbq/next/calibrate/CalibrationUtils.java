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
import org.apache.lucene.index.KnnVectorValues.DocIndexIterator;
import org.apache.lucene.index.VectorSimilarityFunction;

import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

/**
 * Utility methods for quantization calibration: vector math and sampling from
 * {@link FloatVectorValues}.
 */
public final class CalibrationUtils {

    static final int MAX_QUERY_SAMPLE = 1024;
    static final int MAX_CORPUS_SAMPLE = 16384;
    static final int MAX_QUERY_SAMPLE_FAST = 48;
    static final int MAX_CORPUS_SAMPLE_FAST = 8192;
    static final long CALIBRATION_SEED = 215873873L;

    private CalibrationUtils() {}

    /**
     * Sampled data from a {@link FloatVectorValues}: materialized query vectors and
     * ordinal indices into the original {@code floatVectorValues} for the corpus.
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
     * Whether to apply the Neyshabur–Srebro lift (dot product → Euclidean in one higher dimension)
     * before calibration, matching {@code auto_osq} for inner-product metrics.
     */
    public static boolean needsNeyshaburSrebroLift(VectorSimilarityFunction similarityFunction) {
        return similarityFunction == VectorSimilarityFunction.DOT_PRODUCT
            || similarityFunction == VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT;
    }

    /**
     * Maximum squared L2 norm over the sampled corpus vectors (same statistic as reference
     * {@code neyshaburSrebroTransform} over the calibration corpus subset).
     */
    public static double maxSquaredNormOverCorpusSample(FloatVectorValues vectorValues, int[] corpusOrdinals, int dim) throws IOException {
        double maxNormSq = 0;
        for (int ord : corpusOrdinals) {
            float[] v = vectorValues.vectorValue(ord);
            double normSq = dot(dim, v, v);
            if (normSq > maxNormSq) {
                maxNormSq = normSq;
            }
        }
        return maxNormSq;
    }

    /**
     * Lift query rows to {@code dim+1}: {@code [q, 0]} (reference: queries get a zero last coordinate).
     */
    public static float[][] liftQueriesForDotProduct(float[][] queries, int dim) {
        float[][] lifted = new float[queries.length][dim + 1];
        for (int i = 0; i < queries.length; i++) {
            System.arraycopy(queries[i], 0, lifted[i], 0, dim);
            lifted[i][dim] = 0f;
        }
        return lifted;
    }

    /**
     * Corpus view that maps each vector {@code x} to {@code [x, sqrt(M - ||x||^2)]} with
     * {@code M = maxNormSq} over the calibration corpus sample, per Neyshabur and Srebro (ICML 2015).
     */
    public static final class NeyshaburCorpusFloatVectorValues extends FloatVectorValues {
        private final FloatVectorValues delegate;
        private final int dim;
        private final double maxNormSq;
        private final float[] buffer;

        public NeyshaburCorpusFloatVectorValues(FloatVectorValues delegate, int dim, double maxNormSq) {
            this.delegate = delegate;
            this.dim = dim;
            this.maxNormSq = maxNormSq;
            this.buffer = new float[dim + 1];
        }

        @Override
        public float[] vectorValue(int ord) throws IOException {
            float[] v = delegate.vectorValue(ord);
            double normSq = 0;
            for (int j = 0; j < dim; j++) {
                float t = v[j];
                buffer[j] = t;
                normSq += (double) t * t;
            }
            buffer[dim] = (float) Math.sqrt(Math.max(0.0, maxNormSq - normSq));
            return buffer;
        }

        @Override
        public FloatVectorValues copy() throws IOException {
            return new NeyshaburCorpusFloatVectorValues(delegate.copy(), dim, maxNormSq);
        }

        @Override
        public int dimension() {
            return dim + 1;
        }

        @Override
        public int size() {
            return delegate.size();
        }

        @Override
        public DocIndexIterator iterator() {
            return delegate.iterator();
        }
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

        int[] queryOrdinals = Arrays.copyOfRange(indices, 0, nQueries);
        Arrays.sort(queryOrdinals);
        float[][] queries = new float[nQueries][];
        for (int i = 0; i < nQueries; i++) {
            queries[i] = vectorValues.vectorValue(queryOrdinals[i]).clone();
        }
        int[] corpusOrdinals = new int[nDocs];
        System.arraycopy(indices, nQueries, corpusOrdinals, 0, nDocs);
        Arrays.sort(corpusOrdinals);
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
