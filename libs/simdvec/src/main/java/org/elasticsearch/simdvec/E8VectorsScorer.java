/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.simdvec;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.VectorUtil;
import org.elasticsearch.logging.LogManager;
import org.elasticsearch.logging.Logger;
import org.elasticsearch.simdvec.internal.vectorization.E8Kernels;

import java.io.IOException;
import java.util.Locale;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;
import static org.apache.lucene.index.VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT;

/**
 * Scores bulks of E8 block codes (see {@link E8Lattice}) stored transposed: for a bulk of {@code count} documents with
 * {@code blocks} code bytes each, byte {@code b * count + j} is the code of document {@code j} for block {@code b}.
 * <p>
 * The integer part, {@code sum_b ±table[b][pair]}, is computed by one of three kernels, chosen once per JVM: the native
 * library (NEON {@code tbl}/{@code tbx}, AVX2 {@code pshufb}, AVX-512 VBMI {@code vpermi2b}), the Java Vector API, or a
 * scalar loop. The system property {@value #KERNEL_PROPERTY} ({@code auto}, {@code native}, {@code panama} or
 * {@code scalar}) forces one, for benchmarking; a kernel that is not available falls back to the next one.
 * <p>
 * For the DiskBBQ layout, a bulk of codes is followed by the four correction arrays of the OSQ layout; the E8 code
 * stores its ratio scale {@code alpha = |r|^2 / <r, r̂>} in place of the lower interval, and the same additional
 * correction as OSQ, so the final score uses OSQ's formulas with {@code <q - p, r> ≈ alpha * tableScale * sum}.
 */
public class E8VectorsScorer {

    /** System property forcing the E8 kernel: {@code auto} (default), {@code native}, {@code panama} or {@code scalar}. */
    public static final String KERNEL_PROPERTY = "es.vectors.e8.kernel";

    public enum Kernel {
        NATIVE,
        PANAMA,
        SCALAR
    }

    protected final IndexInput in;
    protected final int blocks;
    protected final int bulkSize;
    protected final int[] sums;
    private final byte[] codeScratch;
    protected final float[] alphas;
    protected final float[] additional;

    protected E8VectorsScorer(IndexInput in, int blocks, int bulkSize) {
        this.in = in;
        this.blocks = blocks;
        this.bulkSize = bulkSize;
        this.sums = new int[bulkSize];
        this.codeScratch = new byte[blocks * bulkSize];
        this.alphas = new float[bulkSize];
        this.additional = new float[bulkSize];
    }

    /** The kernel requested by {@value #KERNEL_PROPERTY}, or {@code null} for automatic selection. */
    static Kernel requestedKernel() {
        String value = System.getProperty(KERNEL_PROPERTY, "auto").toLowerCase(Locale.ROOT);
        return switch (value) {
            case "auto" -> null;
            case "native" -> Kernel.NATIVE;
            case "panama" -> Kernel.PANAMA;
            case "scalar" -> Kernel.SCALAR;
            default -> throw new IllegalArgumentException("Unknown " + KERNEL_PROPERTY + ": " + value);
        };
    }

    /**
     * Creates a scorer reading transposed bulks of {@code blocks} code bytes per document from {@code in}, with the kernel
     * requested by {@value #KERNEL_PROPERTY} (by default the fastest available one).
     */
    public static E8VectorsScorer create(IndexInput in, int blocks, int bulkSize) {
        return create(in, blocks, bulkSize, requestedKernel());
    }

    /**
     * Creates a scorer with the given kernel, or the fastest available one if {@code kernel} is {@code null}. A kernel
     * that is not available here (no Vector API, no native library, or an input that cannot be read as memory segments)
     * falls back to the next one: native, then Panama, then scalar.
     */
    public static E8VectorsScorer create(IndexInput in, int blocks, int bulkSize, Kernel kernel) {
        ESVectorizationProvider provider = ESVectorizationProvider.getInstance();
        boolean panama = provider instanceof DefaultESVectorizationProvider == false;
        boolean nativeLib = provider instanceof Native22ESVectorizationProvider;
        E8VectorsScorer scorer = null;
        if (kernel != Kernel.SCALAR && panama) {
            scorer = E8Kernels.create(in, blocks, bulkSize, nativeLib && kernel != Kernel.PANAMA);
        }
        if (scorer == null) {
            scorer = new E8VectorsScorer(in, blocks, bulkSize);
        }
        if (LOGGED_KERNEL.compareAndSet(false, true)) {
            LOGGER.info("E8 scan kernel: {} (requested: {})", scorer.kernel(), kernel == null ? "auto" : kernel);
        }
        return scorer;
    }

    private static final Logger LOGGER = LogManager.getLogger(E8VectorsScorer.class);
    private static final AtomicBoolean LOGGED_KERNEL = new AtomicBoolean();

    /** The kernel this scorer uses. */
    public Kernel kernel() {
        return Kernel.SCALAR;
    }

    public int blocks() {
        return blocks;
    }

    /**
     * Reads {@code count * blocks} transposed code bytes and writes the integer sums into {@code sums[0 .. count)}.
     */
    public void lookupBulk(byte[] table8, int count, int[] sums) throws IOException {
        in.readBytes(codeScratch, 0, blocks * count);
        for (int j = 0; j < count; j++) {
            sums[j] = E8Lattice.lookupOne(codeScratch, 0, table8, blocks, count, j);
        }
    }

    /**
     * Scores a bulk in the DiskBBQ layout: {@code count * blocks} transposed codes, then the four correction arrays
     * (alpha, unused float, unused int, additional correction). Documents outside {@code mask} get a score of zero.
     *
     * @param table8 the int8 query table of {@code (q - p)}, see {@link E8Lattice#buildTable}
     * @param tableScale the scale returned with the table
     * @param queryAdditionalCorrection the centroid term, as for OSQ
     * @return the maximum score over the selected documents
     */
    public float scoreBulk(
        byte[] table8,
        float tableScale,
        float queryAdditionalCorrection,
        VectorSimilarityFunction similarityFunction,
        int mask,
        float[] scores,
        int count
    ) throws IOException {
        lookupBulk(table8, count, sums);
        in.readFloats(alphas, 0, count);
        in.skipBytes((long) count * (Float.BYTES + Integer.BYTES));
        in.readFloats(additional, 0, count);
        return applyCorrections(tableScale, queryAdditionalCorrection, similarityFunction, mask, scores, count);
    }

    /**
     * Turns {@code sums}, {@code alphas} and {@code additional} into scores for the selected documents (others get 0).
     */
    protected float applyCorrections(
        float tableScale,
        float queryAdditionalCorrection,
        VectorSimilarityFunction similarityFunction,
        int mask,
        float[] scores,
        int count
    ) {
        float maxScore = Float.NEGATIVE_INFINITY;
        for (int j = 0; j < count; j++) {
            if (BulkMask.isSelected(mask, j)) {
                float estimate = alphas[j] * tableScale * sums[j];
                float s = applyCorrection(estimate, queryAdditionalCorrection, additional[j], similarityFunction);
                scores[j] = s;
                if (s > maxScore) {
                    maxScore = s;
                }
            } else {
                scores[j] = 0f;
            }
        }
        return maxScore;
    }

    /**
     * The OSQ score transform applied to an estimate of {@code <q - p, x - c>}; see
     * {@link ES940OSQVectorsScorer#applyCorrectionsIndividually}.
     */
    public static float applyCorrection(
        float estimate,
        float queryAdditionalCorrection,
        float additionalCorrection,
        VectorSimilarityFunction similarityFunction
    ) {
        if (similarityFunction == EUCLIDEAN) {
            float score = queryAdditionalCorrection + additionalCorrection - 2 * estimate;
            return Math.max(1 / (1f + score), 0);
        }
        float score = estimate + queryAdditionalCorrection + additionalCorrection;
        if (similarityFunction == MAXIMUM_INNER_PRODUCT) {
            return VectorUtil.scaleMaxInnerProductScore(score);
        }
        return Math.max((1f + score) / 2f, 0);
    }
}
