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
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.VectorUtil;
import org.elasticsearch.index.codec.vectors.OptimizedScalarQuantizer;
import org.elasticsearch.simdvec.ES940OSQVectorsScorer;
import org.elasticsearch.simdvec.ESVectorUtil;
import org.elasticsearch.test.ESTestCase;

import java.io.IOException;
import java.util.List;

import static org.apache.lucene.index.VectorSimilarityFunction.DOT_PRODUCT;
import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;

/**
 * Design check for making DiskBBQ quantizer-agnostic: verifies that the scoring rules of OSQ and ASH are one
 * formula with different bookkeeping. Every quantizer scores a document {@code i} as
 * <pre>
 *   score_i = s_i * (alpha * r_i + beta * sigma_i + gamma) + l_i * kappa + o_i + omega
 * </pre>
 * where {@code r_i} is the raw dot product between the document code and the query code, {@code (s, sigma, l, o)}
 * are per-document correction lanes and {@code (alpha, beta, gamma, kappa, omega)} are per-query scalars
 * (they depend on the query and on the posting list's centroid, never on the document).
 * <p>
 * The OSQ expectation comes from the real {@link ES940OSQVectorsScorer#applyCorrectionsIndividually}. The ASH
 * expectation is a verbatim copy of the lambdas in {@code AshPostingsVisitor}, which cannot be called in isolation;
 * keep both in sync if those change. The estimates are also checked against the exact answer, so that a wrong lane
 * mapping which both formulas happened to share would still be caught.
 */
public class CanonicalScoreFormTests extends ESTestCase {

    private static final int DIM = 128;
    private static final int N_DOCS = 64;

    /** Per-document correction lanes. */
    private record DocLanes(float s, float sigma, float l, float o) {}

    /** Per-query (per posting list) scalars. */
    private record QueryScalars(float alpha, float beta, float gamma, float kappa, float omega) {}

    private static float canonical(DocLanes d, QueryScalars q, float rawDot) {
        return d.s() * (q.alpha() * rawDot + q.beta() * d.sigma() + q.gamma()) + d.l() * q.kappa() + d.o() + q.omega();
    }

    private record OsqConfig(byte docBits, byte queryBits, double maxRelError) {}

    public void testOsqIsCanonical() throws IOException {
        // error thresholds are about 2x the typical relative RMS error, as a sanity check on the lane mapping
        for (OsqConfig cfg : List.of(
            new OsqConfig((byte) 1, (byte) 4, 0.12),
            new OsqConfig((byte) 2, (byte) 4, 0.07),
            new OsqConfig((byte) 4, (byte) 4, 0.03)
        )) {
            for (VectorSimilarityFunction sim : List.of(DOT_PRODUCT, EUCLIDEAN)) {
                checkOsq(cfg, sim);
            }
        }
    }

    private void checkOsq(OsqConfig cfg, VectorSimilarityFunction sim) throws IOException {
        // documents and query share a common direction, as the members of one IVF cluster do
        float[] common = AshUtils.randomGaussians(random(), DIM);
        float[] query = AshUtils.randomGaussians(random(), DIM);
        float[][] docs = new float[N_DOCS][];
        for (int i = 0; i < N_DOCS; i++) {
            docs[i] = AshUtils.randomGaussians(random(), DIM);
        }
        for (int d = 0; d < DIM; d++) {
            query[d] += common[d];
            for (float[] doc : docs) {
                doc[d] += common[d];
            }
        }
        if (sim == DOT_PRODUCT) {
            VectorUtil.l2normalize(query);
            for (float[] doc : docs) {
                VectorUtil.l2normalize(doc);
            }
        }
        // the centroid is the mean of the documents, so residuals are small relative to the vectors
        float[] centroid = new float[DIM];
        for (float[] doc : docs) {
            for (int d = 0; d < DIM; d++) {
                centroid[d] += doc[d] / N_DOCS;
            }
        }

        OptimizedScalarQuantizer osq = new OptimizedScalarQuantizer(sim);
        float[] residual = new float[DIM];
        int[] queryCodes = new int[DIM];
        var qr = osq.scalarQuantize(query.clone(), residual, queryCodes, cfg.queryBits(), centroid);
        float centroidDp = ESVectorUtil.dotProduct(centroid, centroid);

        float docScale = 1f / ((1 << cfg.docBits()) - 1);
        float queryScale = 1f / ((1 << cfg.queryBits()) - 1);
        float ay = qr.lowerInterval();
        float ly = (qr.upperInterval() - ay) * queryScale;
        float y1 = qr.quantizedComponentSum();
        // For euclidean the estimate is a squared distance = addQ + addDoc - 2 * dot, so the factor -2 is applied to
        // the query scalars and the document lanes stay independent of the query.
        float sign = sim == EUCLIDEAN ? -2f : 1f;
        QueryScalars q = new QueryScalars(
            sign * ly,
            sign * ay,
            0f,
            sign * (ay * DIM + ly * y1),
            sim == EUCLIDEAN ? qr.additionalCorrection() : qr.additionalCorrection() - centroidDp
        );

        try (ByteBuffersDirectory dir = new ByteBuffersDirectory(); IndexOutput out = dir.createOutput("empty", IOContext.DEFAULT)) {
            out.writeBytes(new byte[1024], 1024);
            out.close();
            try (IndexInput in = dir.openInput("empty", IOContext.DEFAULT)) {
                ES940OSQVectorsScorer scorer = ESVectorUtil.getES940OSQVectorsScorer(
                    in,
                    cfg.queryBits(),
                    cfg.docBits(),
                    DIM,
                    (DIM * cfg.docBits() + 7) / 8,
                    32,
                    ES940OSQVectorsScorer.BitEncoding.STRIPED
                );
                double sumSqErr = 0;
                double sumSqTrue = 0;
                int[] docCodes = new int[DIM];
                for (float[] doc : docs) {
                    var dr = osq.scalarQuantize(doc.clone(), residual, docCodes, cfg.docBits(), centroid);
                    long raw = 0;
                    for (int k = 0; k < DIM; k++) {
                        raw += (long) docCodes[k] * queryCodes[k];
                    }
                    DocLanes d = new DocLanes(
                        (dr.upperInterval() - dr.lowerInterval()) * docScale,
                        dr.quantizedComponentSum(),
                        dr.lowerInterval(),
                        dr.additionalCorrection()
                    );
                    float estimate = canonical(d, q, raw);

                    float expected = scorer.applyCorrectionsIndividually(
                        qr.lowerInterval(),
                        qr.upperInterval(),
                        qr.quantizedComponentSum(),
                        qr.additionalCorrection(),
                        sim,
                        centroidDp,
                        dr.lowerInterval(),
                        dr.upperInterval(),
                        dr.quantizedComponentSum(),
                        dr.additionalCorrection(),
                        raw
                    );
                    // compare before the clamping and inversion of the similarity transform, which would hide errors
                    assertTrue("score clamped, comparison would be vacuous: " + expected, expected > 0f);
                    float expectedEstimate = sim == EUCLIDEAN ? 1 / expected - 1 : 2 * expected - 1;
                    assertEquals("osq " + cfg + " " + sim, expectedEstimate, estimate, 1e-3f * Math.max(1f, Math.abs(expectedEstimate)));

                    double truth = sim == EUCLIDEAN ? squareDistance(query, doc) : ESVectorUtil.dotProduct(query, doc);
                    sumSqErr += (estimate - truth) * (estimate - truth);
                    sumSqTrue += normalizerSq(sim, query, doc, truth);
                }
                assertTrue(
                    "osq " + cfg + " " + sim + " rel error " + Math.sqrt(sumSqErr / sumSqTrue),
                    Math.sqrt(sumSqErr / sumSqTrue) < cfg.maxRelError()
                );
            }
        }
    }

    public void testAshIsCanonical() throws IOException {
        for (int bits : new int[] { 2, 4 }) {
            for (VectorSimilarityFunction sim : List.of(DOT_PRODUCT, EUCLIDEAN)) {
                for (boolean integerQuery : new boolean[] { false, true }) {
                    checkAsh(bits, sim, integerQuery);
                }
            }
        }
    }

    private void checkAsh(int bits, VectorSimilarityFunction sim, boolean integerQuery) throws IOException {
        AsymmetricHashingQuantizer quantizer = new AsymmetricHashingQuantizer(
            1.0f,
            Integer.MAX_VALUE,
            bits,
            AsymmetricHashingQuantizer.Method.RANDOM,
            0,
            1,
            42L,
            null
        );
        var w = quantizer.train(ord -> new float[DIM], 1, DIM, i -> new float[DIM]);
        int nDims = quantizer.nDims(DIM);

        float[] centroid = AshUtils.randomGaussians(random(), DIM);
        float[] query = AshUtils.randomGaussians(random(), DIM);
        float[][] docs = new float[N_DOCS][];
        for (int i = 0; i < N_DOCS; i++) {
            docs[i] = AshUtils.randomGaussians(random(), DIM);
            for (int d = 0; d < DIM; d++) {
                docs[i][d] += centroid[d];
            }
        }

        float[] qt = ESVectorUtil.matrixVectorMultiply(w.wT(), nDims, DIM, query);
        float queryDotCentroid = ESVectorUtil.dotProduct(query, centroid, DIM);
        float centroidNormSq = ESVectorUtil.dotProduct(centroid, centroid);
        float queryCentroidSqDist = ESVectorUtil.squareDistance(query, centroid);
        float center = ((1 << bits) - 1) / 2.0f;

        // Query quantization, copied from AshPostingsVisitor.quantizeQuery (arithmetic only).
        int queryBits = 4;
        int levels = 1 << queryBits;
        float qMin = Float.MAX_VALUE, qMax = -Float.MAX_VALUE;
        for (float v : qt) {
            qMin = Math.min(qMin, v);
            qMax = Math.max(qMax, v);
        }
        float range = qMax - qMin;
        float qScale = range > 0 ? (levels - 1) / range : 1.0f;
        float invQScale = range > 0 ? range / (levels - 1) : 0f;
        int[] rounded = new int[nDims];
        int unsignedSum = 0;
        for (int j = 0; j < nDims; j++) {
            rounded[j] = Math.clamp(Math.round((qt[j] - qMin) * qScale), 0, levels - 1);
            unsignedSum += rounded[j];
        }
        float constantCorrection = center * (unsignedSum * invQScale + qMin * nDims);

        // per-query scalars for the dot-product-like estimate of <q, x>
        QueryScalars dotQuery = integerQuery
            ? new QueryScalars(invQScale, qMin, -constantCorrection, 0f, queryDotCentroid)
            : new QueryScalars(1f, 0f, 0f, 0f, queryDotCentroid);
        // euclidean: ||q-x||^2 = ||q-c||^2 + ||x-c||^2 - 2 (dot - <x,c> - <q,c> + ||c||^2); <q,c> cancels, and the
        // -2 moves to the query scalars so that document lanes stay query independent.
        QueryScalars euclideanQuery = new QueryScalars(
            -2 * dotQuery.alpha(),
            -2 * dotQuery.beta(),
            -2 * dotQuery.gamma(),
            0f,
            queryCentroidSqDist - 2 * centroidNormSq
        );

        AsymmetricHashingQuantizer.VectorAndNorm precomputed = AsymmetricHashingQuantizer.precomputeCentroid(centroid, w.wT());
        double sumSqErr = 0;
        double sumSqTrue = 0;
        try (AsymmetricHashingQuantizer.BlockEncoder encoder = quantizer.newBlockEncoder(w.w(), DIM, N_DOCS)) {
            for (float[] doc : docs) {
                encoder.add(doc, centroid);
            }
            encoder.encode(precomputed);
            for (int j = 0; j < N_DOCS; j++) {
                float[] code = encoder.code(j); // centered levels
                float scale = encoder.scale(j);
                float offset = encoder.offset(j);
                float vecCentroidDot = encoder.vecCentroidDot(j);
                float vecCentroidSqDist = ESVectorUtil.squareDistance(docs[j], centroid);

                float raw;
                float docSum = 0;
                if (integerQuery) {
                    raw = 0;
                    for (int k = 0; k < nDims; k++) {
                        float unsigned = code[k] + center;
                        docSum += unsigned;
                        raw += rounded[k] * unsigned;
                    }
                } else {
                    raw = ESVectorUtil.dotProduct(qt, code, nDims);
                }

                // expectation: verbatim AshPostingsVisitor correctionApplier + similarityConverter
                float dot;
                if (integerQuery) {
                    float floatDot = Math.fma(invQScale, raw, Math.fma(qMin, docSum, -constantCorrection));
                    dot = Math.fma(floatDot, scale, queryDotCentroid + offset);
                } else {
                    dot = Math.fma(raw, scale, queryDotCentroid + offset);
                }
                float expected = sim == EUCLIDEAN
                    ? queryCentroidSqDist + vecCentroidSqDist - 2 * (dot - vecCentroidDot - queryDotCentroid + centroidNormSq)
                    : dot;

                // canonical form: lanes are s=scale, sigma=docSum, l unused, o similarity specific
                DocLanes d = sim == EUCLIDEAN
                    ? new DocLanes(scale, docSum, 0f, vecCentroidSqDist - 2 * offset + 2 * vecCentroidDot)
                    : new DocLanes(scale, docSum, 0f, offset);
                float estimate = canonical(d, sim == EUCLIDEAN ? euclideanQuery : dotQuery, raw);
                assertEquals(
                    "ash bits=" + bits + " " + sim + " int=" + integerQuery,
                    expected,
                    estimate,
                    1e-3f * Math.max(1f, Math.abs(expected))
                );

                double truth = sim == EUCLIDEAN ? squareDistance(query, docs[j]) : ESVectorUtil.dotProduct(query, docs[j]);
                sumSqErr += (estimate - truth) * (estimate - truth);
                sumSqTrue += normalizerSq(sim, query, docs[j], truth);
            }
        }
        double maxRel = bits == 2 ? 0.07 : 0.025;
        assertTrue(
            "ash bits=" + bits + " " + sim + " int=" + integerQuery + " rel error " + Math.sqrt(sumSqErr / sumSqTrue),
            Math.sqrt(sumSqErr / sumSqTrue) < maxRel
        );
    }

    /** Squared scale that an estimation error is measured against: |q||x| for dot products, the distance for euclidean. */
    private static double normalizerSq(VectorSimilarityFunction sim, float[] q, float[] x, double truth) {
        if (sim == EUCLIDEAN) {
            return truth * truth;
        }
        double nq = Math.sqrt(ESVectorUtil.dotProduct(q, q));
        double nx = Math.sqrt(ESVectorUtil.dotProduct(x, x));
        return nq * nq * nx * nx;
    }

    private static double squareDistance(float[] a, float[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) {
            double d = a[i] - b[i];
            s += d * d;
        }
        return s;
    }
}
