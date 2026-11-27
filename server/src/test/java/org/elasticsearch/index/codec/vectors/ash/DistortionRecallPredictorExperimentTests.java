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
import org.elasticsearch.index.codec.vectors.OptimizedScalarQuantizer;
import org.elasticsearch.index.codec.vectors.diskbbq.Preconditioner;
import org.elasticsearch.index.codec.vectors.diskbbq.calibrate.CodebookProfile;
import org.elasticsearch.index.codec.vectors.diskbbq.calibrate.DistortionRateQuantizer;
import org.elasticsearch.index.codec.vectors.diskbbq.calibrate.DistortionRateTestUtils;
import org.elasticsearch.index.codec.vectors.diskbbq.calibrate.OsqDistortionRateQuantizer;
import org.elasticsearch.index.codec.vectors.diskbbq.calibrate.RecallEstimator;
import org.elasticsearch.index.codec.vectors.diskbbq.calibrate.RotationKind;
import org.elasticsearch.simdvec.ESVectorUtil;
import org.elasticsearch.test.ESTestCase;

import java.io.DataInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.function.IntConsumer;

/**
 * EXPERIMENT (not a regression test): does the recall estimator of "Configuring Project-then-Quantize Embedding
 * Compression" (Algorithm 2) predict the recall of OSQ and ASH, and does it reproduce the effect of OSQ's anisotropy
 * weight lambda?
 * <p>
 * Measured side: the real quantizers encode a corpus sample (landmarks are k-means centroids, residuals are rotated),
 * every document is scored against a full-precision query exhaustively, and recall@10 after reranking the top
 * {@code eta * 10} candidates is measured. Predicted side: no document is quantized. Only the codebook is sampled, by
 * quantizing random unit directions with the same quantizer, and the score errors of the exact top candidates are
 * drawn from the law of the paper's Lemma 3, generalized to any gain by sampling the pair
 * {@code (A, B) = (<w, w_hat>, |w_hat - A w|)} instead of the cosine {@code C} (for a stored-norm gain A = C and
 * B = sqrt(1 - C^2)). Runs only when {@code -Dlaw.corpus} and {@code -Dlaw.queries} name fvec files.
 */
public class DistortionRecallPredictorExperimentTests extends ESTestCase {

    private static final int K = 10;
    private static final int[] ETAS = { 1, 3 };
    private static final int POOL = 600; // 20 * eta * k at the largest eta
    private static final int DRAWS = 32;
    private static final int SAMPLES = 3000;
    private static final int MAX_KEEP = 30; // eta * k at the largest eta

    private interface Rotation {
        void apply(float[] in, float[] out);
    }

    private record Codebook(
        float[] a,
        float[] b,
        double meanC,
        double meanA,
        double meanClosedFormA,
        double varC,
        double prGap,
        double maxShare
    ) {}

    private static final boolean SWEEP = Boolean.getBoolean("law.sweep");
    private static final boolean PRODUCTION = Boolean.getBoolean("law.production");

    public void testPredictedVersusMeasuredRecall() throws Exception {
        String corpusPath = System.getProperty("law.corpus");
        assumeTrue("set -Dlaw.corpus and -Dlaw.queries to run this experiment", corpusPath != null);
        String name = System.getProperty("law.name", "data");
        float[][] docs = readFvec(Path.of(corpusPath));
        float[][] queries = readFvec(Path.of(System.getProperty("law.queries")));
        int n = Math.min(docs.length, Integer.getInteger("law.n", docs.length));
        docs = Arrays.copyOf(docs, n);
        queries = Arrays.copyOf(queries, Math.min(queries.length, Integer.getInteger("law.q", 500)));
        if (Boolean.getBoolean("law.normalize")) {
            normalize(docs);
            normalize(queries);
        }
        int dim = docs[0].length;
        ExecutorService ex = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
        try {
            run(name, docs, queries, dim, ex);
        } finally {
            ex.shutdown();
            ex.awaitTermination(1, TimeUnit.MINUTES);
        }
    }

    private enum QueryMode {
        FP,
        Q4_OSQ,
        Q4_ASH
    }

    /**
     * The candidates the estimator works on, for one way of searching: exhaustive, or restricted to the clusters an IVF
     * search visits. {@code isTrue} marks the candidates that are among the true nearest neighbors of the whole corpus.
     */
    private record Scope(
        String name,
        int[][] poolId,
        float[][] poolScore,
        float[][] poolG,
        float[][] poolAlpha,
        float[][] poolR,
        boolean[][] isTrue,
        boolean[][] visited,
        double partition
    ) {}

    /** What depends on the quantizer and rotation under test. */
    private record Cfg(float[][] recon, float[][] qFp, float[][] qAsh, Rotation rot, float[][] rotCents, float[][] dqQC, float[] dqQ) {}

    private void run(String name, float[][] docs, float[][] queries, int dim, ExecutorService ex) throws Exception {
        int n = docs.length;
        int nq = queries.length;
        long t0 = System.nanoTime();
        int k = (int) Math.sqrt(n);
        float[][] cents = new float[k][];
        int[] assign = new int[n];
        kmeans(docs, cents, assign, 6, ex);
        logger.info("EXP3 {} kmeans k={} done in {}s", name, k, (System.nanoTime() - t0) / 1_000_000_000);

        float[] gain = new float[n];
        for (int i = 0; i < n; i++) {
            gain[i] = (float) Math.sqrt(ESVectorUtil.squareDistance(docs[i], cents[assign[i]]));
        }
        int[][] docsByCluster = new int[k][];
        int[] counts = new int[k];
        for (int i = 0; i < n; i++) {
            counts[assign[i]]++;
        }
        for (int c = 0; c < k; c++) {
            docsByCluster[c] = new int[counts[c]];
            counts[c] = 0;
        }
        for (int i = 0; i < n; i++) {
            docsByCluster[assign[i]][counts[assign[i]]++] = i;
        }
        float[][] qm = new float[nq][k]; // <q, centroid>
        parallelFor(ex, nq, q -> {
            for (int c = 0; c < k; c++) {
                qm[q][c] = ESVectorUtil.dotProduct(queries[q], cents[c]);
            }
        });
        float[] qnorm2 = new float[nq];
        for (int q = 0; q < nq; q++) {
            qnorm2[q] = ESVectorUtil.dotProduct(queries[q], queries[q]);
        }

        // exact top candidates per query, most similar first: the first K are the true neighbors of the whole corpus
        int[][] globalId = new int[nq][];
        float[][] globalScore = new float[nq][];
        parallelFor(ex, nq, q -> {
            Top top = new Top(POOL);
            for (int i = 0; i < n; i++) {
                top.offer(ESVectorUtil.dotProduct(queries[q], docs[i]), i);
            }
            globalId[q] = top.ids();
            globalScore[q] = top.scores();
        });
        int[][] trueIds = new int[nq][];
        for (int q = 0; q < nq; q++) {
            trueIds[q] = Arrays.copyOf(globalId[q], K);
        }
        List<Scope> scopes = new ArrayList<>();
        scopes.add(buildScope("exh", globalId, globalScore, null, 0, trueIds, qm, assign, gain, qnorm2));
        for (double visit : new double[] { 0.02, 0.08 }) {
            scopes.add(
                ivfScope("ivf" + (int) Math.round(visit * 100), visit, docs, queries, qm, docsByCluster, assign, gain, qnorm2, trueIds, ex)
            );
        }
        Scope exh = scopes.get(0);
        Scope ivf2 = scopes.get(1);
        Scope ivf8 = scopes.get(2);
        logger.info(
            "EXP3 {} pools done in {}s; share of true neighbors in the visited clusters: ivf2={} ivf8={}",
            name,
            (System.nanoTime() - t0) / 1_000_000_000,
            fmt(ivf2.partition()),
            fmt(ivf8.partition())
        );

        Rotation haar = haarRotation(dim, new Random(5));
        Rotation block = blockRotation(dim, 32);
        Rotation none = (in, out) -> System.arraycopy(in, 0, out, 0, in.length);
        float[][] recon = new float[n][dim];
        Map<String, float[][]> dqCache = new HashMap<>();

        record OsqCfg(int bits, float lambda, String rotName, Rotation rot) {}
        List<OsqCfg> osq = new ArrayList<>();
        if (SWEEP) {
            for (int bits : new int[] { 1, 2 }) {
                osq.add(new OsqCfg(bits, 0.1f, "none", none));
                osq.add(new OsqCfg(bits, 0.1f, "block8", blockRotation(dim, 8)));
                osq.add(new OsqCfg(bits, 0.1f, "block32", block));
                osq.add(new OsqCfg(bits, 0.1f, "block128", blockRotation(dim, 128)));
                osq.add(new OsqCfg(bits, 0.1f, "haar", haar));
            }
        }
        for (float lambda : SWEEP ? new float[0] : new float[] { 0.1f, 1.0f }) {
            osq.add(new OsqCfg(1, lambda, "haar", haar));
            osq.add(new OsqCfg(1, lambda, "block32", block));
            osq.add(new OsqCfg(1, lambda, "none", none));
            osq.add(new OsqCfg(2, lambda, "haar", haar));
        }
        for (OsqCfg oc : osq) {
            Codebook iso = osqCodebook(dim, oc.bits(), oc.lambda(), ex);
            reconOsq(docs, cents, assign, oc.rot(), oc.lambda(), oc.bits(), recon, ex);
            Codebook data = dataCodebook(docs, cents, assign, oc.rot(), recon);
            float[][] qFp = new float[nq][dim];
            for (int q = 0; q < nq; q++) {
                oc.rot().apply(queries[q], qFp[q]);
            }
            float[][] rotCents = new float[k][dim];
            for (int c = 0; c < k; c++) {
                oc.rot().apply(cents[c], rotCents[c]);
            }
            float[][] dq = dqCache.get(oc.rotName());
            if (dq == null) {
                dq = dqOsq(queries, cents, oc.rot(), ex);
                dqCache.put(oc.rotName(), dq);
            }
            Cfg cfg = new Cfg(recon, qFp, null, oc.rot(), rotCents, dq, null);
            String label = "osq-b" + oc.bits() + "-l" + oc.lambda();
            scenarios(
                name,
                label,
                oc.rotName(),
                iso,
                data,
                cfg,
                QueryMode.Q4_OSQ,
                exh,
                ivf2,
                ivf8,
                queries,
                cents,
                qm,
                docsByCluster,
                assign,
                trueIds,
                dim,
                ex
            );
            if (PRODUCTION) {
                RotationKind kind = oc.rotName().equals("haar") ? RotationKind.HAAR
                    : oc.rotName().equals("none") ? RotationKind.NONE
                    : RotationKind.BLOCK_DIAGONAL;
                productionCheck(
                    name,
                    label,
                    oc.rotName(),
                    new OsqDistortionRateQuantizer(dim, oc.bits(), 4, oc.lambda(), kind, oc.rot()::apply),
                    docs,
                    cents,
                    assign,
                    queries,
                    ivf8
                );
            }
        }
        for (int bits : SWEEP ? new int[0] : new int[] { 1, 2, 4 }) {
            Codebook iso = ashCodebook(dim, bits);
            var w = reconAsh(docs, cents, assign, bits, recon, ex);
            Codebook data = dataCodebook(docs, cents, assign, null, recon);
            float[] dqQ = new float[nq];
            float[][] qAsh = ashQueries(queries, w, w.wT().length / dim, dim, dqQ);
            Cfg cfg = new Cfg(recon, queries, qAsh, null, null, null, dqQ);
            scenarios(
                name,
                "ash-b" + bits,
                "learned-random-W",
                iso,
                data,
                cfg,
                QueryMode.Q4_ASH,
                exh,
                ivf2,
                ivf8,
                queries,
                cents,
                qm,
                docsByCluster,
                assign,
                trueIds,
                dim,
                ex
            );
            if (PRODUCTION) {
                productionCheck(
                    name,
                    "ash-b" + bits,
                    "learned-random-W",
                    AshDistortionRateQuantizer.random(dim, bits, 4, 42L),
                    docs,
                    cents,
                    assign,
                    queries,
                    ivf8
                );
            }
        }
        logger.info("EXP3 {} finished in {}s", name, (System.nanoTime() - t0) / 1_000_000_000);
    }

    /** The estimate of the production classes, to be compared with the experiment's own in the EXP3 lines. */
    private void productionCheck(
        String name,
        String cfgLabel,
        String rot,
        DistortionRateQuantizer production,
        float[][] docs,
        float[][] cents,
        int[] assign,
        float[][] queries,
        Scope ivf8
    ) throws IOException {
        CodebookProfile response = CodebookProfile.sample(production, SAMPLES, 7);
        for (boolean ivf : new boolean[] { false, true }) {
            var pools = RecallEstimator.pools(
                production,
                DistortionRateTestUtils.sampleOf(docs, cents, assign),
                DistortionRateTestUtils.values(queries),
                DistortionRateTestUtils.ordinals(queries.length),
                K,
                POOL,
                ivf ? ivf8.visited() : null,
                ivf
            );
            double[] estimate = RecallEstimator.estimate(response, pools, ETAS, K, production.latentDimension(), DRAWS, 11);
            logger.info(
                "EXP5 {} cfg={} rot={} scope={} query={} cProd={} production@1={} production@3={}",
                name,
                cfgLabel,
                rot,
                ivf ? "ivf8" : "exh",
                ivf ? "q4" : "fp",
                fmt(response.angularError()),
                fmt(estimate[0]),
                fmt(estimate[1])
            );
        }
    }

    private void scenarios(
        String name,
        String cfgLabel,
        String rot,
        Codebook iso,
        Codebook data,
        Cfg cfg,
        QueryMode q4,
        Scope exh,
        Scope ivf2,
        Scope ivf8,
        float[][] queries,
        float[][] cents,
        float[][] qm,
        int[][] docsByCluster,
        int[] assign,
        int[][] trueIds,
        int dim,
        ExecutorService ex
    ) throws Exception {
        double cIso = 2 * (1 - iso.meanC());
        double cData = 2 * (1 - data.meanC());
        String gap = fmt((cData - cIso) / cIso);
        record Run(Scope scope, QueryMode mode) {}
        Run[] runs = SWEEP
            ? new Run[] { new Run(exh, QueryMode.FP), new Run(ivf8, q4) }
            : new Run[] { new Run(exh, QueryMode.FP), new Run(exh, q4), new Run(ivf2, q4), new Run(ivf8, QueryMode.FP), new Run(ivf8, q4) };
        for (Run run : runs) {
            boolean quant = run.mode() != QueryMode.FP;
            double[] measured = measured(cfg, run.scope(), run.mode(), queries, cents, qm, docsByCluster, trueIds, ex);
            double[] predIso = predict(iso, run.scope(), quant, cfg, assign, dim, ex);
            double[] predData = predict(data, run.scope(), quant, cfg, assign, dim, ex);
            if (SWEEP) {
                logger.info(
                    "EXP4 {} cfg={} rot={} cGap={} psiGap={} prGap={} maxShare={} scope={} query={} err1={} err3={} measured1={} measured3={}",
                    name,
                    cfgLabel,
                    rot,
                    fmt((cData - cIso) / cIso),
                    fmt((data.varC() - iso.varC()) / iso.varC()),
                    fmt(data.prGap()),
                    fmt(data.maxShare()),
                    run.scope().name(),
                    quant ? "q4" : "fp",
                    fmt(100 * (predIso[0] - measured[0])),
                    fmt(100 * (predIso[1] - measured[1])),
                    fmt(measured[0]),
                    fmt(measured[1])
                );
            }
            logger.info(
                "EXP3 {} cfg={} rot={} cIso={} cData={} isoGap={} scope={} query={} measured@1={} iso@1={} data@1={} measured@3={} iso@3={} data@3={}",
                name,
                cfgLabel,
                rot,
                fmt(cIso),
                fmt(cData),
                gap,
                run.scope().name(),
                quant ? "q4" : "fp",
                fmt(measured[0]),
                fmt(predIso[0]),
                fmt(predData[0]),
                fmt(measured[1]),
                fmt(predIso[1]),
                fmt(predData[1])
            );
        }
    }

    /** Exhaustive when {@code visited} is null, otherwise the pools hold what the visited clusters contain. */
    private static Scope buildScope(
        String name,
        int[][] poolId,
        float[][] poolScore,
        boolean[][] visited,
        double partition,
        int[][] trueIds,
        float[][] qm,
        int[] assign,
        float[] gain,
        float[] qnorm2
    ) {
        int nq = poolId.length;
        float[][] g = new float[nq][];
        float[][] alpha = new float[nq][];
        float[][] r = new float[nq][];
        boolean[][] isTrue = new boolean[nq][];
        for (int q = 0; q < nq; q++) {
            int m = poolId[q].length;
            g[q] = new float[m];
            alpha[q] = new float[m];
            r[q] = new float[m];
            isTrue[q] = new boolean[m];
            for (int p = 0; p < m; p++) {
                int i = poolId[q][p];
                g[q][p] = gain[i];
                alpha[q][p] = (poolScore[q][p] - qm[q][assign[i]]) / gain[i];
                r[q][p] = (float) Math.sqrt(Math.max(0, qnorm2[q] - alpha[q][p] * alpha[q][p]));
                for (int t : trueIds[q]) {
                    if (t == i) {
                        isTrue[q][p] = true;
                    }
                }
            }
        }
        return new Scope(name, poolId, poolScore, g, alpha, r, isTrue, visited, partition);
    }

    private static Scope ivfScope(
        String name,
        double visitFraction,
        float[][] docs,
        float[][] queries,
        float[][] qm,
        int[][] docsByCluster,
        int[] assign,
        float[] gain,
        float[] qnorm2,
        int[][] trueIds,
        ExecutorService ex
    ) throws Exception {
        int nq = queries.length;
        int k = docsByCluster.length;
        int need = (int) Math.ceil(visitFraction * docs.length);
        boolean[][] visited = new boolean[nq][k];
        int[][] poolId = new int[nq][];
        float[][] poolScore = new float[nq][];
        parallelFor(ex, nq, q -> {
            Integer[] order = new Integer[k];
            for (int c = 0; c < k; c++) {
                order[c] = c;
            }
            Arrays.sort(order, (x, y) -> Float.compare(qm[q][y], qm[q][x]));
            int seen = 0;
            Top top = new Top(POOL);
            for (int j = 0; j < k && seen < need; j++) {
                int c = order[j];
                visited[q][c] = true;
                seen += docsByCluster[c].length;
                for (int i : docsByCluster[c]) {
                    top.offer(ESVectorUtil.dotProduct(queries[q], docs[i]), i);
                }
            }
            poolId[q] = top.ids();
            poolScore[q] = top.scores();
        });
        long inside = 0;
        for (int q = 0; q < nq; q++) {
            for (int t : trueIds[q]) {
                if (visited[q][assign[t]]) {
                    inside++;
                }
            }
        }
        double partition = inside / (double) ((long) nq * K);
        return buildScope(name, poolId, poolScore, visited, partition, trueIds, qm, assign, gain, qnorm2);
    }

    /** Exhaustive or IVF search with compressed scores, returning recall@K after reranking the top eta * K. */
    private static double[] measured(
        Cfg cfg,
        Scope scope,
        QueryMode mode,
        float[][] queries,
        float[][] cents,
        float[][] qm,
        int[][] docsByCluster,
        int[][] trueIds,
        ExecutorService ex
    ) throws Exception {
        int nq = queries.length;
        int dim = queries[0].length;
        int k = cents.length;
        int[][] kept = new int[nq][];
        parallelFor(ex, nq, q -> {
            Top top = new Top(MAX_KEEP);
            float[] residual = new float[dim];
            float[] rotated = new float[dim];
            float[] scratch = new float[dim];
            float[] vec = new float[dim];
            float[] zero = new float[dim];
            int[] codes = new int[dim];
            OptimizedScalarQuantizer qq = mode == QueryMode.Q4_OSQ
                ? new OptimizedScalarQuantizer(VectorSimilarityFunction.DOT_PRODUCT)
                : null;
            for (int c = 0; c < k; c++) {
                if (scope.visited() != null && scope.visited()[q][c] == false) {
                    continue;
                }
                float[] qv;
                if (mode == QueryMode.FP) {
                    qv = cfg.qFp()[q];
                } else if (mode == QueryMode.Q4_ASH) {
                    qv = cfg.qAsh()[q];
                } else {
                    // the query's residual against the centroid is quantized to 4 bits; the centroid itself is exact
                    for (int d = 0; d < dim; d++) {
                        residual[d] = queries[q][d] - cents[c][d];
                    }
                    cfg.rot().apply(residual, rotated);
                    var res = qq.scalarQuantize(rotated, scratch, codes, (byte) 4, zero);
                    float step = (res.upperInterval() - res.lowerInterval()) / 15f;
                    for (int d = 0; d < dim; d++) {
                        vec[d] = cfg.rotCents()[c][d] + res.lowerInterval() + step * codes[d];
                    }
                    qv = vec;
                }
                for (int i : docsByCluster[c]) {
                    top.offer(qm[q][c] + ESVectorUtil.dotProduct(qv, cfg.recon()[i]), i);
                }
            }
            kept[q] = top.ids();
        });
        double[] out = new double[ETAS.length];
        for (int e = 0; e < ETAS.length; e++) {
            long hit = 0;
            for (int q = 0; q < nq; q++) {
                int limit = Math.min(ETAS[e] * K, kept[q].length);
                for (int j = 0; j < limit; j++) {
                    for (int t = 0; t < K; t++) {
                        if (kept[q][j] == trueIds[q][t]) {
                            hit++;
                        }
                    }
                }
            }
            out[e] = hit / (double) ((long) nq * K);
        }
        return out;
    }

    /**
     * The estimator: no document is quantized. The score error of each candidate is drawn from the law of the paper's
     * Lemma 3, with the codebook sample {@code cb}, plus, when the query is quantized, an independent term of variance
     * {@code (|w_hat| g_i)^2 |dq|^2 / d} where {@code dq} is the quantization error of the query.
     */
    private static double[] predict(Codebook cb, Scope sc, boolean quantQuery, Cfg cfg, int[] assign, int dim, ExecutorService ex)
        throws Exception {
        int nq = sc.poolId().length;
        double[][] perQuery = new double[nq][ETAS.length];
        parallelFor(ex, nq, q -> {
            Random rnd = new Random(1000 + q);
            int m = sc.poolId()[q].length;
            long[] keys = new long[m];
            double[] hits = new double[ETAS.length];
            for (int t = 0; t < DRAWS; t++) {
                for (int p = 0; p < m; p++) {
                    int s = rnd.nextInt(cb.a().length);
                    double xi = rnd.nextGaussian() / Math.sqrt(dim - 1); // one coordinate of a uniform point of S^{d-2}
                    double g = sc.poolG()[q][p];
                    double a = cb.a()[s];
                    double b = cb.b()[s];
                    double sig = sc.poolScore()[q][p] - g * sc.poolAlpha()[q][p] * (1 - a) + g * sc.poolR()[q][p] * b * xi;
                    if (quantQuery) {
                        double dq = cfg.dqQC() != null ? cfg.dqQC()[q][assign[sc.poolId()[q][p]]] : cfg.dqQ()[q];
                        sig += g * Math.sqrt(a * a + b * b) * dq / Math.sqrt(dim) * rnd.nextGaussian();
                    }
                    int bits = Float.floatToIntBits((float) sig);
                    int sortable = bits >= 0 ? bits : bits ^ 0x7FFFFFFF;
                    keys[p] = ((long) ~sortable << 32) | p; // ascending order = highest score first
                }
                Arrays.sort(keys);
                for (int e = 0; e < ETAS.length; e++) {
                    int limit = Math.min(ETAS[e] * K, m);
                    for (int j = 0; j < limit; j++) {
                        if (sc.isTrue()[q][(int) keys[j]]) {
                            hits[e]++;
                        }
                    }
                }
            }
            for (int e = 0; e < ETAS.length; e++) {
                perQuery[q][e] = hits[e] / (DRAWS * (double) K);
            }
        });
        double[] out = new double[ETAS.length];
        for (int e = 0; e < ETAS.length; e++) {
            double sum = 0;
            for (int q = 0; q < nq; q++) {
                sum += perQuery[q][e];
            }
            out[e] = sum / nq;
        }
        return out;
    }

    /** |error| of the 4-bit quantization of each query's rotated residual against each centroid, as OSQ does it. */
    private static float[][] dqOsq(float[][] queries, float[][] cents, Rotation rot, ExecutorService ex) throws Exception {
        int nq = queries.length;
        int dim = queries[0].length;
        float[][] out = new float[nq][cents.length];
        parallelFor(ex, nq, q -> {
            OptimizedScalarQuantizer qq = new OptimizedScalarQuantizer(VectorSimilarityFunction.DOT_PRODUCT);
            float[] residual = new float[dim];
            float[] rotated = new float[dim];
            float[] scratch = new float[dim];
            float[] zero = new float[dim];
            int[] codes = new int[dim];
            for (int c = 0; c < cents.length; c++) {
                for (int d = 0; d < dim; d++) {
                    residual[d] = queries[q][d] - cents[c][d];
                }
                rot.apply(residual, rotated);
                var res = qq.scalarQuantize(rotated, scratch, codes, (byte) 4, zero);
                float step = (res.upperInterval() - res.lowerInterval()) / 15f;
                double err = 0;
                for (int d = 0; d < dim; d++) {
                    double diff = rotated[d] - (res.lowerInterval() + step * codes[d]);
                    err += diff * diff;
                }
                out[q][c] = (float) Math.sqrt(err);
            }
        });
        return out;
    }

    /** ASH's query: the projection W^T q quantized to 4 bits (min-max, as the posting visitor does) and mapped back. */
    private static float[][] ashQueries(
        float[][] queries,
        AsymmetricHashingQuantizer.TrainedProjection w,
        int nDims,
        int dim,
        float[] dqOut
    ) {
        float[][] out = new float[queries.length][dim];
        int levels = 16;
        for (int q = 0; q < queries.length; q++) {
            float[] qt = ESVectorUtil.matrixVectorMultiply(w.wT(), nDims, dim, queries[q]);
            float min = Float.MAX_VALUE;
            float max = -Float.MAX_VALUE;
            for (float v : qt) {
                min = Math.min(min, v);
                max = Math.max(max, v);
            }
            float range = max - min;
            float scale = range > 0 ? (levels - 1) / range : 1f;
            float inv = range > 0 ? range / (levels - 1) : 0f;
            float[] deq = new float[nDims];
            double err = 0;
            for (int j = 0; j < nDims; j++) {
                int code = Math.clamp(Math.round((qt[j] - min) * scale), 0, levels - 1);
                deq[j] = min + code * inv;
                double diff = qt[j] - deq[j];
                err += diff * diff;
            }
            dqOut[q] = (float) Math.sqrt(err);
            for (int d = 0; d < dim; d++) {
                out[q][d] = ESVectorUtil.dotProduct(w.w(), d * nDims, deq, 0, nDims);
            }
        }
        return out;
    }

    /**
     * The same (A, B) pairs as the isotropic codebook sample, but taken from real rotated residuals of the corpus,
     * with the real quantizer's reconstruction. If the rotation spreads the energy of the residuals evenly, this
     * agrees with the isotropic sample; if not, it shows by how much it does not.
     */
    private static Codebook dataCodebook(float[][] docs, float[][] cents, int[] assign, Rotation rot, float[][] recon) {
        int n = docs.length;
        int dim = docs[0].length;
        int want = Math.min(n, 2000);
        int step = Math.max(1, n / want);
        float[] a = new float[want];
        float[] b = new float[want];
        double[] c = new double[want];
        float[] residual = new float[dim];
        float[] rotated = new float[dim];
        double[] energy = new double[dim];
        int got = 0;
        for (int i = 0; i < n && got < want; i += step) {
            for (int d = 0; d < dim; d++) {
                residual[d] = docs[i][d] - cents[assign[i]][d];
            }
            float[] r = residual;
            if (rot != null) {
                rot.apply(residual, rotated);
                r = rotated;
            }
            double rr = ESVectorUtil.dotProduct(r, r);
            double hh = ESVectorUtil.dotProduct(recon[i], recon[i]);
            if (rr == 0 || hh == 0) {
                continue; // a document equal to its centroid has no direction to quantize
            }
            double dot = ESVectorUtil.dotProduct(r, recon[i]);
            double aa = dot / rr;
            a[got] = (float) aa;
            b[got] = (float) Math.sqrt(Math.max(0, hh / rr - aa * aa));
            c[got] = dot / Math.sqrt(rr * hh);
            for (int d = 0; d < dim; d++) {
                energy[d] += (double) r[d] * r[d] / rr;
            }
            got++;
        }
        return new Codebook(
            Arrays.copyOf(a, got),
            Arrays.copyOf(b, got),
            mean(Arrays.copyOf(c, got)),
            mean(Arrays.copyOf(a, got)),
            Double.NaN,
            variance(Arrays.copyOf(c, got)),
            prGap(energy, got),
            maxShare(energy, got)
        );
    }

    // ---------------------------------------------------------------- codebooks

    private static float[] randomUnit(Random rnd, int dim) {
        float[] w = new float[dim];
        double norm = 0;
        for (int i = 0; i < dim; i++) {
            w[i] = (float) rnd.nextGaussian();
            norm += w[i] * w[i];
        }
        float inv = (float) (1 / Math.sqrt(norm));
        for (int i = 0; i < dim; i++) {
            w[i] *= inv;
        }
        return w;
    }

    /** Samples (A, B) of the real OSQ quantizer on random directions: A = {@code <w, w_hat>}, B = {@code |w_hat - A w|}. */
    private static Codebook osqCodebook(int dim, int bits, float lambda, ExecutorService ex) throws Exception {
        float[] a = new float[SAMPLES];
        float[] b = new float[SAMPLES];
        double[] c = new double[SAMPLES];
        double[] closed = new double[SAMPLES];
        parallelFor(ex, SAMPLES, s -> {
            Random rnd = new Random(77L * 31 + s);
            OptimizedScalarQuantizer q = new OptimizedScalarQuantizer(VectorSimilarityFunction.DOT_PRODUCT, lambda, 5);
            float[] w = randomUnit(rnd, dim);
            int[] codes = new int[dim];
            var res = q.scalarQuantize(w, new float[dim], codes, (byte) bits, new float[dim]);
            float step = (res.upperInterval() - res.lowerInterval()) / ((1 << bits) - 1);
            double dot = 0, nh = 0;
            for (int i = 0; i < dim; i++) {
                double h = res.lowerInterval() + step * codes[i];
                dot += w[i] * h;
                nh += h * h;
            }
            a[s] = (float) dot;
            b[s] = (float) Math.sqrt(Math.max(0, nh - dot * dot));
            c[s] = dot / Math.sqrt(nh);
            closed[s] = c[s] * c[s] / (lambda + (1 - lambda) * c[s] * c[s]); // the note's h(C) * C
        });
        return new Codebook(a, b, mean(c), mean(a), mean(closed), variance(c), 0, 1);
    }

    /** ASH's stored-norm gain: A = cos(w, code), B = sin. Measured from the real quantizer, random W, d = D. */
    private static Codebook ashCodebook(int dim, int bits) throws Exception {
        AsymmetricHashingQuantizer aq = new AsymmetricHashingQuantizer(
            1.0f,
            Integer.MAX_VALUE,
            bits,
            AsymmetricHashingQuantizer.Method.RANDOM,
            0,
            1,
            42L,
            null
        );
        var w = aq.train(ord -> new float[dim], 1, dim, i -> new float[dim]);
        int nDims = aq.nDims(dim);
        float[] zero = new float[dim];
        var pre = AsymmetricHashingQuantizer.precomputeCentroid(zero, w.wT());
        Random rnd = new Random(99);
        float[] a = new float[SAMPLES];
        float[] b = new float[SAMPLES];
        double[] c = new double[SAMPLES];
        try (var enc = aq.newBlockEncoder(w.w(), dim, 250)) {
            for (int start = 0; start < SAMPLES; start += 250) {
                float[][] ws = new float[250][];
                enc.reset();
                for (int j = 0; j < 250; j++) {
                    ws[j] = randomUnit(rnd, dim);
                    enc.add(ws[j], zero);
                }
                enc.encode(pre);
                for (int j = 0; j < 250; j++) {
                    float[] hat = reconstructAsh(w.w(), nDims, dim, enc.code(j), enc.scale(j));
                    double dot = ESVectorUtil.dotProduct(ws[j], hat);
                    double nh = ESVectorUtil.dotProduct(hat, hat);
                    a[start + j] = (float) dot;
                    b[start + j] = (float) Math.sqrt(Math.max(0, nh - dot * dot));
                    c[start + j] = dot / Math.sqrt(nh);
                }
            }
        }
        return new Codebook(a, b, mean(c), mean(a), mean(c), variance(c), 0, 1);
    }

    private static float[] reconstructAsh(float[] w, int nDims, int dim, float[] code, float scale) {
        float[] out = new float[dim];
        for (int d = 0; d < dim; d++) {
            out[d] = scale * ESVectorUtil.dotProduct(w, d * nDims, code, 0, nDims);
        }
        return out;
    }

    // ---------------------------------------------------------------- reconstructions of the corpus

    private static void reconOsq(
        float[][] docs,
        float[][] cents,
        int[] assign,
        Rotation rot,
        float lambda,
        int bits,
        float[][] recon,
        ExecutorService ex
    ) throws Exception {
        int dim = docs[0].length;
        parallelChunks(ex, docs.length, 1024, (from, to) -> {
            OptimizedScalarQuantizer q = new OptimizedScalarQuantizer(VectorSimilarityFunction.DOT_PRODUCT, lambda, 5);
            float[] residual = new float[dim];
            float[] rotated = new float[dim];
            float[] scratch = new float[dim];
            float[] zero = new float[dim];
            int[] codes = new int[dim];
            for (int i = from; i < to; i++) {
                float[] c = cents[assign[i]];
                for (int d = 0; d < dim; d++) {
                    residual[d] = docs[i][d] - c[d];
                }
                rot.apply(residual, rotated);
                var res = q.scalarQuantize(rotated, scratch, codes, (byte) bits, zero);
                float step = (res.upperInterval() - res.lowerInterval()) / ((1 << bits) - 1);
                for (int d = 0; d < dim; d++) {
                    recon[i][d] = res.lowerInterval() + step * codes[d];
                }
            }
        });
    }

    private static AsymmetricHashingQuantizer.TrainedProjection reconAsh(
        float[][] docs,
        float[][] cents,
        int[] assign,
        int bits,
        float[][] recon,
        ExecutorService ex
    ) throws Exception {
        int dim = docs[0].length;
        AsymmetricHashingQuantizer base = new AsymmetricHashingQuantizer(
            1.0f,
            Integer.MAX_VALUE,
            bits,
            AsymmetricHashingQuantizer.Method.RANDOM,
            0,
            1,
            42L,
            null
        );
        var w = base.train(ord -> new float[dim], 1, dim, i -> new float[dim]);
        int nDims = base.nDims(dim);
        List<List<Integer>> byCluster = new ArrayList<>();
        for (int c = 0; c < cents.length; c++) {
            byCluster.add(new ArrayList<>());
        }
        for (int i = 0; i < docs.length; i++) {
            byCluster.get(assign[i]).add(i);
        }
        parallelFor(ex, cents.length, c -> {
            List<Integer> ids = byCluster.get(c);
            if (ids.isEmpty()) {
                return;
            }
            AsymmetricHashingQuantizer aq = new AsymmetricHashingQuantizer(
                1.0f,
                Integer.MAX_VALUE,
                bits,
                AsymmetricHashingQuantizer.Method.RANDOM,
                0,
                1,
                42L,
                null
            );
            var pre = AsymmetricHashingQuantizer.precomputeCentroid(cents[c], w.wT());
            try (var enc = aq.newBlockEncoder(w.w(), dim, 256)) {
                for (int start = 0; start < ids.size(); start += 256) {
                    int end = Math.min(ids.size(), start + 256);
                    enc.reset();
                    for (int j = start; j < end; j++) {
                        enc.add(docs[ids.get(j)], cents[c]);
                    }
                    enc.encode(pre);
                    for (int j = start; j < end; j++) {
                        recon[ids.get(j)] = reconstructAsh(w.w(), nDims, dim, enc.code(j - start), enc.scale(j - start));
                    }
                }
            }
        });
        return w;
    }

    // ---------------------------------------------------------------- k-means, rotations, helpers

    private static void kmeans(float[][] docs, float[][] cents, int[] assign, int iters, ExecutorService ex) throws Exception {
        int n = docs.length;
        int k = cents.length;
        int dim = docs[0].length;
        Random rnd = new Random(1);
        for (int c = 0; c < k; c++) {
            cents[c] = docs[rnd.nextInt(n)].clone();
        }
        for (int it = 0; it <= iters; it++) {
            float[] half = new float[k];
            for (int c = 0; c < k; c++) {
                half[c] = 0.5f * ESVectorUtil.dotProduct(cents[c], cents[c]);
            }
            parallelFor(ex, n, i -> {
                int best = 0;
                float bestScore = Float.NEGATIVE_INFINITY;
                for (int c = 0; c < k; c++) {
                    float s = ESVectorUtil.dotProduct(docs[i], cents[c]) - half[c];
                    if (s > bestScore) {
                        bestScore = s;
                        best = c;
                    }
                }
                assign[i] = best;
            });
            if (it == iters) {
                break;
            }
            double[][] sum = new double[k][dim];
            int[] count = new int[k];
            for (int i = 0; i < n; i++) {
                count[assign[i]]++;
                double[] s = sum[assign[i]];
                for (int d = 0; d < dim; d++) {
                    s[d] += docs[i][d];
                }
            }
            for (int c = 0; c < k; c++) {
                if (count[c] > 0) {
                    for (int d = 0; d < dim; d++) {
                        cents[c][d] = (float) (sum[c][d] / count[c]);
                    }
                }
            }
        }
    }

    private static Rotation haarRotation(int dim, Random rnd) {
        double[][] m = new double[dim][dim];
        for (double[] row : m) {
            for (int j = 0; j < dim; j++) {
                row[j] = rnd.nextGaussian();
            }
        }
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < i; j++) {
                double dot = 0;
                for (int d = 0; d < dim; d++) {
                    dot += m[i][d] * m[j][d];
                }
                for (int d = 0; d < dim; d++) {
                    m[i][d] -= dot * m[j][d];
                }
            }
            double norm = 0;
            for (int d = 0; d < dim; d++) {
                norm += m[i][d] * m[i][d];
            }
            norm = Math.sqrt(norm);
            for (int d = 0; d < dim; d++) {
                m[i][d] /= norm;
            }
        }
        float[][] r = new float[dim][dim];
        for (int i = 0; i < dim; i++) {
            for (int d = 0; d < dim; d++) {
                r[i][d] = (float) m[i][d];
            }
        }
        return (in, out) -> {
            for (int i = 0; i < dim; i++) {
                out[i] = ESVectorUtil.dotProduct(r[i], in);
            }
        };
    }

    private static Rotation blockRotation(int dim, int blockDim) {
        Preconditioner pre = Preconditioner.createPreconditioner(dim, blockDim);
        return pre::applyTransform;
    }

    /** The best {@code capacity} scores seen, kept with their ids. */
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

        private Integer[] order() {
            Integer[] order = new Integer[size];
            for (int i = 0; i < size; i++) {
                order[i] = i;
            }
            Arrays.sort(order, (x, y) -> Float.compare(scores[y], scores[x]));
            return order;
        }

        int[] ids() {
            Integer[] order = order();
            int[] out = new int[size];
            for (int i = 0; i < size; i++) {
                out[i] = ids[order[i]];
            }
            return out;
        }

        float[] scores() {
            Integer[] order = order();
            float[] out = new float[size];
            for (int i = 0; i < size; i++) {
                out[i] = scores[order[i]];
            }
            return out;
        }
    }

    private static double variance(double[] v) {
        double m = mean(v);
        double s = 0;
        for (double x : v) {
            s += (x - m) * (x - m);
        }
        return s / v.length;
    }

    /** d * sum(sigma_j^4) - 1, where sigma_j^2 is the mean energy share of coordinate j: 0 when it is uniform. */
    private static double prGap(double[] energy, int samples) {
        double sum = 0;
        for (double e : energy) {
            double share = e / samples;
            sum += share * share;
        }
        return energy.length * sum - 1;
    }

    /** d * max_j sigma_j^2: how many times its fair share the busiest coordinate carries. */
    private static double maxShare(double[] energy, int samples) {
        double max = 0;
        for (double e : energy) {
            max = Math.max(max, e / samples);
        }
        return energy.length * max;
    }

    private static double mean(float[] v) {
        double s = 0;
        for (float x : v) {
            s += x;
        }
        return s / v.length;
    }

    private static double mean(double[] v) {
        double s = 0;
        for (double x : v) {
            s += x;
        }
        return s / v.length;
    }

    private static String fmt(double v) {
        return String.format(Locale.ROOT, "%.4f", v);
    }

    private interface Chunk {
        void run(int from, int to) throws Exception;
    }

    private static void parallelChunks(ExecutorService ex, int n, int chunk, Chunk body) throws Exception {
        List<Future<Void>> futures = new ArrayList<>();
        for (int from = 0; from < n; from += chunk) {
            final int f = from;
            final int t = Math.min(n, from + chunk);
            Callable<Void> task = () -> {
                body.run(f, t);
                return null;
            };
            futures.add(ex.submit(task));
        }
        for (Future<Void> f : futures) {
            f.get();
        }
    }

    private static void parallelFor(ExecutorService ex, int n, IntConsumer body) throws Exception {
        parallelChunks(ex, n, Math.max(1, n / (4 * Runtime.getRuntime().availableProcessors())), (from, to) -> {
            for (int i = from; i < to; i++) {
                body.accept(i);
            }
        });
    }

    private static void normalize(float[][] vectors) {
        for (float[] v : vectors) {
            double norm = Math.sqrt(ESVectorUtil.dotProduct(v, v));
            if (norm > 0) {
                for (int i = 0; i < v.length; i++) {
                    v[i] /= (float) norm;
                }
            }
        }
    }

    private static float[][] readFvec(Path path) throws IOException {
        try (DataInputStream in = new DataInputStream(Files.newInputStream(path))) {
            byte[] header = new byte[4];
            in.readFully(header);
            int dim = ByteBuffer.wrap(header).order(ByteOrder.LITTLE_ENDIAN).getInt();
            long rows = Files.size(path) / (4L + 4L * dim);
            float[][] out = new float[(int) rows][dim];
            byte[] buf = new byte[4 * dim];
            for (int r = 0; r < rows; r++) {
                if (r > 0) {
                    in.readFully(header);
                }
                in.readFully(buf);
                ByteBuffer.wrap(buf).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(out[r]);
            }
            return out;
        }
    }
}
