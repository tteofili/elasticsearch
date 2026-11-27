/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq.calibrate;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.elasticsearch.index.codec.vectors.OptimizedScalarQuantizer;
import org.elasticsearch.index.codec.vectors.diskbbq.calibrate.DistortionRateTestUtils.Dataset;
import org.elasticsearch.simdvec.ESVectorUtil;
import org.elasticsearch.test.ESTestCase;

import java.io.IOException;
import java.util.List;

import static org.elasticsearch.index.codec.vectors.diskbbq.calibrate.DistortionRateTestUtils.clusteredDataset;
import static org.elasticsearch.index.codec.vectors.diskbbq.calibrate.DistortionRateTestUtils.haar;
import static org.elasticsearch.index.codec.vectors.diskbbq.calibrate.DistortionRateTestUtils.osq;
import static org.elasticsearch.index.codec.vectors.diskbbq.calibrate.DistortionRateTestUtils.recall;
import static org.hamcrest.Matchers.closeTo;
import static org.hamcrest.Matchers.greaterThan;
import static org.hamcrest.Matchers.greaterThanOrEqualTo;

/** The estimator predicts the recall that real OSQ quantization of a corpus gives, without quantizing the corpus. */
public class RecallEstimatorTests extends ESTestCase {

    private static final int DIM = 64;
    private static final int K = 10;
    private static final int[] ETAS = { 1, 3 };

    /** Compressed scores of every document for every query, the way OSQ scores them, with a full-precision query. */
    private static float[][] compressedScores(Dataset data, int bits, OsqDistortionRateQuantizer.Rotation rotation) {
        OptimizedScalarQuantizer quantizer = new OptimizedScalarQuantizer(VectorSimilarityFunction.DOT_PRODUCT, 0.1f, 5);
        float[][] recon = new float[data.docs().length][DIM];
        float[] residual = new float[DIM];
        float[] rotated = new float[DIM];
        int[] codes = new int[DIM];
        for (int i = 0; i < recon.length; i++) {
            float[] centroid = data.centroids()[data.assignment()[i]];
            for (int d = 0; d < DIM; d++) {
                residual[d] = data.docs()[i][d] - centroid[d];
            }
            rotation.apply(residual, rotated);
            var result = quantizer.scalarQuantize(rotated, new float[DIM], codes, (byte) bits, new float[DIM]);
            float step = (result.upperInterval() - result.lowerInterval()) / ((1 << bits) - 1);
            for (int d = 0; d < DIM; d++) {
                recon[i][d] = result.lowerInterval() + step * codes[d];
            }
        }
        float[][] scores = new float[data.queries().length][recon.length];
        float[] rotatedQuery = new float[DIM];
        for (int q = 0; q < scores.length; q++) {
            rotation.apply(data.queries()[q], rotatedQuery);
            for (int i = 0; i < recon.length; i++) {
                float[] centroid = data.centroids()[data.assignment()[i]];
                scores[q][i] = ESVectorUtil.dotProduct(data.queries()[q], centroid) + ESVectorUtil.dotProduct(rotatedQuery, recon[i]);
            }
        }
        return scores;
    }

    public void testEstimateTracksMeasuredRecall() throws IOException {
        Dataset data = clusteredDataset(100, 30, DIM, 60, 31);
        OsqDistortionRateQuantizer.Rotation rotation = haar(DIM, 5);
        double[] previous = null;
        for (int bits : new int[] { 1, 2 }) {
            OsqDistortionRateQuantizer quantizer = osq(DIM, bits, 0.1f, RotationKind.HAAR, rotation);
            CodebookProfile profile = CodebookProfile.sample(quantizer, CodebookProfile.DEFAULT_SAMPLES, 7);
            List<RecallEstimator.Pool> pools = RecallEstimator.pools(
                quantizer,
                data.sample(),
                data.queryValues(),
                data.queryOrdinals(),
                K,
                300,
                null,
                false
            );
            double[] estimated = RecallEstimator.estimate(
                profile,
                pools,
                ETAS,
                K,
                quantizer.latentDimension(),
                RecallEstimator.DEFAULT_DRAWS,
                11
            );
            float[][] scores = compressedScores(data, bits, rotation);
            for (int e = 0; e < ETAS.length; e++) {
                double measured = recall(data.queries(), data.docs(), scores, K, ETAS[e]);
                assertThat("bits " + bits + " eta " + ETAS[e], estimated[e], closeTo(measured, 0.07));
            }
            assertThat("reranking more candidates cannot lower recall", estimated[1], greaterThanOrEqualTo(estimated[0]));
            if (previous != null) {
                assertThat("more bits, more recall", estimated[0], greaterThan(previous[0]));
            }
            previous = estimated;
        }
    }

    public void testQueryQuantizationAddsNoiseAndNeverRaisesRecall() throws IOException {
        Dataset data = clusteredDataset(100, 30, DIM, 60, 33);
        OsqDistortionRateQuantizer quantizer = osq(DIM, 1, 0.1f, RotationKind.HAAR, haar(DIM, 6));
        CodebookProfile profile = CodebookProfile.sample(quantizer, CodebookProfile.DEFAULT_SAMPLES, 7);
        double[] plain = RecallEstimator.estimate(
            profile,
            RecallEstimator.pools(quantizer, data.sample(), data.queryValues(), data.queryOrdinals(), K, 300, null, false),
            ETAS,
            K,
            DIM,
            RecallEstimator.DEFAULT_DRAWS,
            3
        );
        double[] quantized = RecallEstimator.estimate(
            profile,
            RecallEstimator.pools(quantizer, data.sample(), data.queryValues(), data.queryOrdinals(), K, 300, null, true),
            ETAS,
            K,
            DIM,
            RecallEstimator.DEFAULT_DRAWS,
            3
        );
        assertThat(quantized[0], org.hamcrest.Matchers.lessThanOrEqualTo(plain[0] + 0.01));
    }

    public void testVisitingFewerClustersLowersRecall() throws IOException {
        // clusters that overlap, so that the neighbors of a query are spread over several of them
        Dataset data = clusteredDataset(100, 30, DIM, 60, 35, 2.0f);
        OsqDistortionRateQuantizer quantizer = osq(DIM, 2, 0.1f, RotationKind.HAAR, haar(DIM, 8));
        CodebookProfile profile = CodebookProfile.sample(quantizer, CodebookProfile.DEFAULT_SAMPLES, 7);
        boolean[][] visited = new boolean[data.queries().length][data.centroids().length];
        for (int q = 0; q < visited.length; q++) {
            // visit the clusters whose centroid is among the 3 closest to the query
            float[] scores = new float[data.centroids().length];
            for (int c = 0; c < scores.length; c++) {
                scores[c] = ESVectorUtil.dotProduct(data.queries()[q], data.centroids()[c]);
            }
            for (int visit = 0; visit < 3; visit++) {
                int best = -1;
                for (int c = 0; c < scores.length; c++) {
                    if (visited[q][c] == false && (best < 0 || scores[c] > scores[best])) {
                        best = c;
                    }
                }
                visited[q][best] = true;
            }
        }
        double exhaustive = RecallEstimator.estimate(
            profile,
            RecallEstimator.pools(quantizer, data.sample(), data.queryValues(), data.queryOrdinals(), K, 300, null, false),
            ETAS,
            K,
            DIM,
            RecallEstimator.DEFAULT_DRAWS,
            3
        )[1];
        double ivf = RecallEstimator.estimate(
            profile,
            RecallEstimator.pools(quantizer, data.sample(), data.queryValues(), data.queryOrdinals(), K, 300, visited, false),
            ETAS,
            K,
            DIM,
            RecallEstimator.DEFAULT_DRAWS,
            3
        )[1];
        assertThat("visiting 3 of 30 clusters misses neighbors", ivf, org.hamcrest.Matchers.lessThan(exhaustive - 0.05));
    }
}
