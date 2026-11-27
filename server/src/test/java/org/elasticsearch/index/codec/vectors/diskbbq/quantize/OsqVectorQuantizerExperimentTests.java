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
import org.elasticsearch.simdvec.ESVectorUtil;
import org.elasticsearch.test.ESTestCase;

import java.util.Arrays;
import java.util.Locale;
import java.util.Random;

/**
 * EXPERIMENT (not a regression test): times the quantization sequence a segment writer used to carry inline against
 * {@link OsqVectorQuantizer}, which replaced it, on the same vectors, alternating the two so that neither gets a warmer
 * JIT. Runs only with {@code -Dexp.osqperf=true}.
 */
public class OsqVectorQuantizerExperimentTests extends ESTestCase {

    public void testTimings() {
        assumeTrue("exp.osqperf is not set", Boolean.getBoolean("exp.osqperf"));
        int dim = Integer.getInteger("exp.dim", 768);
        int n = Integer.getInteger("exp.n", 20_000);
        int reps = Integer.getInteger("exp.reps", 15);
        Random random = new Random(1);
        float[][] vectors = new float[n][dim];
        for (float[] v : vectors) {
            for (int d = 0; d < dim; d++) {
                v[d] = (float) random.nextGaussian();
            }
        }
        float[][] centroids = new float[16][dim];
        for (float[] c : centroids) {
            for (int d = 0; d < dim; d++) {
                c[d] = (float) random.nextGaussian() * 0.1f;
            }
        }
        float[] parent = new float[dim];
        for (VectorSimilarityFunction similarity : new VectorSimilarityFunction[] {
            VectorSimilarityFunction.EUCLIDEAN,
            VectorSimilarityFunction.DOT_PRODUCT }) {
            for (int bits : new int[] { 1, 4 }) {
                for (boolean withParent : new boolean[] { false, true }) {
                    OptimizedScalarQuantizer inline = new OptimizedScalarQuantizer(similarity);
                    OsqVectorQuantizer extracted = new OsqVectorQuantizer(similarity, dim, bits);
                    int[] levels = new int[dim];
                    float[] scratch = new float[dim];
                    float[] parentCentroid = withParent ? parent : null;
                    long[] oldNanos = new long[reps];
                    long[] newNanos = new long[reps];
                    double sink = 0;
                    for (int rep = 0; rep < reps; rep++) {
                        boolean oldFirst = rep % 2 == 0;
                        for (int pass = 0; pass < 2; pass++) {
                            boolean old = (pass == 0) == oldFirst;
                            long start = System.nanoTime();
                            for (int i = 0; i < n; i++) {
                                float[] v = vectors[i];
                                float[] c = centroids[i & 15];
                                OptimizedScalarQuantizer.QuantizationResult r;
                                if (old) {
                                    r = inline.scalarQuantize(v, scratch, levels, (byte) bits, c);
                                    if (parentCentroid != null) {
                                        float additional = switch (similarity) {
                                            case EUCLIDEAN -> ESVectorUtil.squareDistance(v, parentCentroid);
                                            case DOT_PRODUCT, MAXIMUM_INNER_PRODUCT -> ESVectorUtil.dotProduct(scratch, parentCentroid);
                                            default -> throw new AssertionError(similarity);
                                        };
                                        r = new OptimizedScalarQuantizer.QuantizationResult(
                                            r.lowerInterval(),
                                            r.upperInterval(),
                                            additional,
                                            r.quantizedComponentSum()
                                        );
                                    }
                                } else {
                                    r = extracted.quantize(v, c, parentCentroid, levels);
                                }
                                sink += r.lowerInterval() + r.additionalCorrection() + levels[i % dim];
                            }
                            long elapsed = System.nanoTime() - start;
                            if (old) {
                                oldNanos[rep] = elapsed;
                            } else {
                                newNanos[rep] = elapsed;
                            }
                        }
                    }
                    Arrays.sort(oldNanos);
                    Arrays.sort(newNanos);
                    // skip the first third: warm-up
                    int from = reps / 3;
                    double oldMedian = oldNanos[(from + reps) / 2] / 1e6;
                    double newMedian = newNanos[(from + reps) / 2] / 1e6;
                    logger.info(
                        "EXP7 {} bits={} parent={} old={}ms new={}ms ratio(new/old)={} sink={}",
                        similarity,
                        bits,
                        withParent,
                        String.format(Locale.ROOT, "%.2f", oldMedian),
                        String.format(Locale.ROOT, "%.2f", newMedian),
                        String.format(Locale.ROOT, "%.4f", newMedian / oldMedian),
                        sink > 0
                    );
                }
            }
        }
    }
}
