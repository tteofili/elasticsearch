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
import org.elasticsearch.index.codec.vectors.diskbbq.QuantEncoding;
import org.elasticsearch.index.codec.vectors.diskbbq.quantize.CanonicalScorer;
import org.elasticsearch.index.codec.vectors.diskbbq.quantize.DocEncoder;
import org.elasticsearch.index.codec.vectors.diskbbq.quantize.DocLanes;
import org.elasticsearch.index.codec.vectors.diskbbq.quantize.OsqDocEncoder;
import org.elasticsearch.index.codec.vectors.diskbbq.quantize.OsqQueryEncoder;
import org.elasticsearch.index.codec.vectors.diskbbq.quantize.PreparedQuery;
import org.elasticsearch.simdvec.ES940OSQVectorsScorer;
import org.elasticsearch.simdvec.ESVectorUtil;
import org.elasticsearch.test.ESTestCase;

import java.io.IOException;

import static org.apache.lucene.index.VectorSimilarityFunction.DOT_PRODUCT;
import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;
import static org.elasticsearch.index.codec.vectors.OptimizedScalarQuantizer.DEFAULT_LAMBDA;

/**
 * One scorer for every scheme: the codes and corrections that the generic {@link DocEncoder} makes and the scalars that the
 * generic {@link PreparedQuery} gives, put through {@link CanonicalScorer}, score a vector as the scheme's own kernel does.
 * <p>
 * The OSQ expectation is the real {@link ES940OSQVectorsScorer#applyCorrectionsIndividually}. The ASH expectation is a
 * verbatim copy of the lambdas in {@code AshPostingsVisitor}, which cannot be called in isolation; keep it in sync if those
 * change (as {@link CanonicalScoreFormTests} does, which checks the form itself).
 */
public class CanonicalScorerTests extends ESTestCase {

    private static final int DIM = 128;
    private static final int N_DOCS = 64;

    public void testOsqScoresLikeItsKernel() throws IOException {
        for (QuantEncoding encoding : new QuantEncoding[] {
            QuantEncoding.ONE_BIT_4BIT_QUERY,
            QuantEncoding.TWO_BIT_4BIT_QUERY,
            QuantEncoding.FOUR_BIT_SYMMETRIC }) {
            for (VectorSimilarityFunction similarity : new VectorSimilarityFunction[] { DOT_PRODUCT, EUCLIDEAN }) {
                checkOsq(encoding, similarity);
            }
        }
    }

    private void checkOsq(QuantEncoding encoding, VectorSimilarityFunction similarity) throws IOException {
        float[] common = AshUtils.randomGaussians(random(), DIM);
        float[] query = AshUtils.randomGaussians(random(), DIM);
        float[][] docs = new float[N_DOCS][];
        for (int i = 0; i < N_DOCS; i++) {
            docs[i] = AshUtils.randomGaussians(random(), DIM);
        }
        // documents and query share a common direction, as the members of one IVF cluster do
        for (int d = 0; d < DIM; d++) {
            query[d] += common[d];
            for (float[] doc : docs) {
                doc[d] += common[d];
            }
        }
        if (similarity == DOT_PRODUCT) {
            VectorUtil.l2normalize(query);
            for (float[] doc : docs) {
                VectorUtil.l2normalize(doc);
            }
        }
        float[] centroid = new float[DIM];
        for (float[] doc : docs) {
            for (int d = 0; d < DIM; d++) {
                centroid[d] += doc[d] / N_DOCS;
            }
        }
        String label = "osq " + encoding + " " + similarity;

        // the query as the reader quantizes it
        OptimizedScalarQuantizer queryQuantizer = new OptimizedScalarQuantizer(similarity, DEFAULT_LAMBDA, 1);
        var qr = queryQuantizer.scalarQuantize(query.clone(), new float[DIM], new int[DIM], encoding.queryBits(), centroid);
        float centroidDp = ESVectorUtil.dotProduct(centroid, centroid);

        try (
            DocEncoder encoder = OsqDocEncoder.forField(similarity, DIM, encoding.bits(), N_DOCS);
            ByteBuffersDirectory dir = new ByteBuffersDirectory();
            IndexOutput out = dir.createOutput("empty", IOContext.DEFAULT)
        ) {
            encoder.reset(centroid);
            for (float[] doc : docs) {
                encoder.add(doc.clone());
            }
            encoder.encode();
            PreparedQuery prepared = new OsqQueryEncoder(similarity, encoding, DIM, ordinal -> centroid).prepare(query.clone());
            prepared.forList(0);

            out.writeBytes(new byte[1024], 1024);
            out.close();
            try (IndexInput in = dir.openInput("empty", IOContext.DEFAULT)) {
                ES940OSQVectorsScorer kernel = ESVectorUtil.getES940OSQVectorsScorer(
                    in,
                    encoding.queryBits(),
                    encoding.bits(),
                    DIM,
                    (DIM * encoding.bits() + 7) / 8,
                    32,
                    ES940OSQVectorsScorer.BitEncoding.STRIPED
                );
                int[] docLevels = new int[encoder.codeLength()];
                int[] nativeLanes = new int[encoder.nativeLaneCount()];
                var decodeLanes = new DocLanes();
                for (int i = 0; i < N_DOCS; i++) {
                    encoder.read(i, docLevels, decodeLanes);
                    encoder.nativeLanes(i, nativeLanes);
                    long raw = 0;
                    int[] queryLevels = new int[DIM];
                    queryQuantizer.scalarQuantize(query.clone(), new float[DIM], queryLevels, encoding.queryBits(), centroid);
                    for (int k = 0; k < DIM; k++) {
                        raw += (long) docLevels[k] * queryLevels[k];
                    }
                    float expected = kernel.applyCorrectionsIndividually(
                        qr.lowerInterval(),
                        qr.upperInterval(),
                        qr.quantizedComponentSum(),
                        qr.additionalCorrection(),
                        similarity,
                        centroidDp,
                        Float.intBitsToFloat(nativeLanes[0]),
                        Float.intBitsToFloat(nativeLanes[1]),
                        nativeLanes[3],
                        Float.intBitsToFloat(nativeLanes[2]),
                        raw
                    );
                    // compare before the clamping and inversion of the similarity transform, which would hide errors
                    assertTrue(label + " score clamped, comparison would be vacuous: " + expected, expected > 0f);
                    float expectedEstimate = similarity == EUCLIDEAN ? 1 / expected - 1 : 2 * expected - 1;
                    float actual = CanonicalScorer.score(encoder, i, prepared);
                    assertEquals(label + " vector " + i, expectedEstimate, actual, 1e-3f * Math.max(1f, Math.abs(expectedEstimate)));
                }
            }
        }
    }

    public void testAshScoresLikeItsVisitor() throws IOException {
        for (int bits : new int[] { 2, 4 }) {
            for (VectorSimilarityFunction similarity : new VectorSimilarityFunction[] { DOT_PRODUCT, EUCLIDEAN }) {
                for (int queryBits : new int[] { 0, 4 }) {
                    checkAsh(bits, similarity, queryBits);
                }
            }
        }
    }

    private void checkAsh(int bits, VectorSimilarityFunction similarity, int queryBits) throws IOException {
        String label = "ash bits=" + bits + " " + similarity + " queryBits=" + queryBits;
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
        var trained = quantizer.train(ord -> new float[DIM], 1, DIM, i -> new float[DIM]);
        AshProjectionMatrix matrix = trained.matrix();
        int nDims = matrix.nDims();

        float[] centroid = AshUtils.randomGaussians(random(), DIM);
        float[] query = AshUtils.randomGaussians(random(), DIM);
        float[][] docs = new float[N_DOCS][];
        for (int i = 0; i < N_DOCS; i++) {
            docs[i] = AshUtils.randomGaussians(random(), DIM);
            for (int d = 0; d < DIM; d++) {
                docs[i][d] += centroid[d];
            }
        }
        boolean integerQuery = queryBits > 0;
        float center = ((1 << bits) - 1) / 2.0f;
        // what the visitor starts from: the int8 projection of the query, and on its grid for the integer scorer
        float[] qt = matrix.projectQuery(query);
        float queryDotCentroid = ESVectorUtil.dotProduct(query, centroid, DIM);
        float centroidNormSq = ESVectorUtil.dotProduct(centroid, centroid);
        float queryCentroidSqDist = ESVectorUtil.squareDistance(query, centroid);
        int[] rounded = new int[nDims];
        AshQueryEncoder.Grid grid = integerQuery ? AshQueryEncoder.quantizeToLevels(qt, nDims, queryBits, rounded) : null;
        int unsignedSum = 0;
        for (int level : rounded) {
            unsignedSum += level;
        }
        float constantCorrection = integerQuery ? center * (unsignedSum * grid.inverse() + grid.min() * nDims) : 0f;

        float[] w = ESVectorUtil.transposeMatrix(matrix.wT(), nDims, DIM);
        try (DocEncoder generic = new AshDocEncoder(matrix, quantizer, w, similarity, bits, N_DOCS)) {
            var precomputed = AsymmetricHashingQuantizer.precomputeCentroid(centroid, matrix.wT());
            AsymmetricHashingQuantizer.BlockEncoder production = quantizer.newBlockEncoder(w, DIM, N_DOCS);
            generic.reset(centroid);
            production.reset();
            for (float[] doc : docs) {
                generic.add(doc);
                production.add(doc, centroid);
            }
            generic.encode();
            production.encode(precomputed);
            PreparedQuery prepared = new AshQueryEncoder(matrix, bits, queryBits, similarity, ordinal -> centroid).prepare(query);
            prepared.forList(0);

            for (int j = 0; j < N_DOCS; j++) {
                float[] code = production.code(j);
                float scale = production.scale(j);
                float offset = production.offset(j);
                float vecCentroidDot = production.vecCentroidDot(j);
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
                    float floatDot = Math.fma(grid.inverse(), raw, Math.fma(grid.min(), docSum, -constantCorrection));
                    dot = Math.fma(floatDot, scale, queryDotCentroid + offset);
                } else {
                    dot = Math.fma(raw, scale, queryDotCentroid + offset);
                }
                float expected = similarity == EUCLIDEAN
                    ? queryCentroidSqDist + vecCentroidSqDist - 2 * (dot - vecCentroidDot - queryDotCentroid + centroidNormSq)
                    : dot;

                float actual = CanonicalScorer.score(generic, j, prepared);
                assertEquals(label + " vector " + j, expected, actual, 1e-3f * Math.max(1f, Math.abs(expected)));
            }
            production.close();
        }
    }
}
