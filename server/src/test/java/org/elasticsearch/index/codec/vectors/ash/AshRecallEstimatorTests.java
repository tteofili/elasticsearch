/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.ash;

import org.elasticsearch.index.codec.vectors.diskbbq.calibrate.CodebookProfile;
import org.elasticsearch.index.codec.vectors.diskbbq.calibrate.DistortionRateTestUtils.Dataset;
import org.elasticsearch.index.codec.vectors.diskbbq.calibrate.RecallEstimator;
import org.elasticsearch.simdvec.ESVectorUtil;
import org.elasticsearch.test.ESTestCase;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static org.elasticsearch.index.codec.vectors.diskbbq.calibrate.DistortionRateTestUtils.clusteredDataset;
import static org.elasticsearch.index.codec.vectors.diskbbq.calibrate.DistortionRateTestUtils.recall;
import static org.hamcrest.Matchers.closeTo;
import static org.hamcrest.Matchers.greaterThanOrEqualTo;

/** The estimator predicts the recall that real ASH quantization of a corpus gives, with a full-precision query. */
public class AshRecallEstimatorTests extends ESTestCase {

    private static final int DIM = 64;
    private static final int K = 10;
    private static final int[] ETAS = { 1, 3 };

    public void testEstimateTracksMeasuredRecall() throws IOException {
        Dataset data = clusteredDataset(100, 30, DIM, 60, 41);
        for (int bits : new int[] { 1, 2 }) {
            AsymmetricHashingQuantizer trainer = new AsymmetricHashingQuantizer(
                1.0f,
                Integer.MAX_VALUE,
                bits,
                AsymmetricHashingQuantizer.Method.RANDOM,
                0,
                1,
                17L,
                null
            );
            var trained = trainer.train(ord -> new float[DIM], 1, DIM, i -> new float[DIM]);
            AshDistortionRateQuantizer quantizer = new AshDistortionRateQuantizer(
                trained.matrix(),
                bits,
                0,
                org.elasticsearch.index.codec.vectors.diskbbq.calibrate.RotationKind.HAAR
            );

            // what ASH scores: the exact centroid term plus scale * <projected query, code>
            float[][] codes = new float[data.docs().length][];
            float[] scales = new float[data.docs().length];
            List<List<Integer>> byCluster = new ArrayList<>();
            for (int c = 0; c < data.centroids().length; c++) {
                byCluster.add(new ArrayList<>());
            }
            for (int i = 0; i < data.docs().length; i++) {
                byCluster.get(data.assignment()[i]).add(i);
            }
            for (int c = 0; c < data.centroids().length; c++) {
                var pre = AsymmetricHashingQuantizer.precomputeCentroid(data.centroids()[c], trained.wT());
                List<Integer> ids = byCluster.get(c);
                try (var encoder = trainer.newBlockEncoder(trained.w(), DIM, ids.size())) {
                    for (int id : ids) {
                        encoder.add(data.docs()[id], data.centroids()[c]);
                    }
                    encoder.encode(pre);
                    for (int j = 0; j < ids.size(); j++) {
                        codes[ids.get(j)] = encoder.code(j).clone();
                        scales[ids.get(j)] = encoder.scale(j);
                    }
                }
            }
            float[][] scores = new float[data.queries().length][data.docs().length];
            for (int q = 0; q < scores.length; q++) {
                float[] projected = quantizer.project(data.queries()[q]);
                for (int i = 0; i < data.docs().length; i++) {
                    float[] centroid = data.centroids()[data.assignment()[i]];
                    scores[q][i] = ESVectorUtil.dotProduct(data.queries()[q], centroid) + scales[i] * ESVectorUtil.dotProduct(
                        projected,
                        codes[i],
                        DIM
                    );
                }
            }

            CodebookProfile profile = CodebookProfile.sample(quantizer, CodebookProfile.DEFAULT_SAMPLES, 7);
            var pools = RecallEstimator.pools(quantizer, data.sample(), data.queryValues(), data.queryOrdinals(), K, 300, null, false);
            double[] estimated = RecallEstimator.estimate(
                profile,
                pools,
                ETAS,
                K,
                quantizer.latentDimension(),
                RecallEstimator.DEFAULT_DRAWS,
                11
            );
            for (int e = 0; e < ETAS.length; e++) {
                double measured = recall(data.queries(), data.docs(), scores, K, ETAS[e]);
                assertThat("bits " + bits + " eta " + ETAS[e], estimated[e], closeTo(measured, 0.07));
            }
            assertThat(estimated[1], greaterThanOrEqualTo(estimated[0]));
        }
    }
}
