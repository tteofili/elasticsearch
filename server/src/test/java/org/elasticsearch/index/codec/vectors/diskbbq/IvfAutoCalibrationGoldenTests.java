/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq;

import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.elasticsearch.test.ESTestCase;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static org.hamcrest.Matchers.equalTo;

/**
 * Pins the exact configuration auto-calibration selects on fixed-seed synthetic corpora, so refactors of the
 * calibration internals (e.g. abstracting the quantizer) can be shown to be behavior-preserving. Calibration is
 * deterministic for a given corpus, so any change in these outputs is a behavior change, not noise. Update the
 * expectations only when a change to the calibration algorithm itself is intended.
 */
public class IvfAutoCalibrationGoldenTests extends ESTestCase {

    private static final int VPC = 128;
    private static final int DIM = 32;
    private static final int BLOCK_DIM = 32;

    public void testSelectedConfigIsStable() throws IOException {
        List<String> actual = new ArrayList<>();
        for (VectorSimilarityFunction sim : VectorSimilarityFunction.values()) {
            for (double target : new double[] { 0.8, 0.9, 0.99 }) {
                for (int maxDocBits : new int[] { 7, 2, 1 }) {
                    if (sim == VectorSimilarityFunction.EUCLIDEAN && maxDocBits < 7) {
                        // Capped EUCLIDEAN cells sit on selection boundaries (a candidate just meets the target, otherwise a
                        // best-effort one wins), and which side they land on depends on JIT-dependent floating point
                        // noise: a cold JVM and a warmed-up one can disagree. Not stable goldens.
                        continue;
                    }
                    FloatVectorValues vectors = AutoCalibrationVectorFixtures.clusteredHeapVectors(3000, DIM, 24, 7L + sim.ordinal());
                    IvfAutoCalibration calibration = new IvfAutoCalibration(VPC, BLOCK_DIM, target, 10, maxDocBits);
                    IvfSegmentConfig config = calibration.calibrate(vectors, sim);
                    actual.add(
                        sim
                            + "|target="
                            + target
                            + "|maxDocBits="
                            + maxDocBits
                            + " -> "
                            + config.osqEncoding()
                            + ",precondition="
                            + config.usePrecondition()
                            + ",oversample="
                            + config.rescoreOversample()
                    );
                }
            }
        }
        String expected = ""
            + "EUCLIDEAN|target=0.8|maxDocBits=7 -> FOUR_BIT_SYMMETRIC,precondition=true,oversample=1.25\n"
            + "EUCLIDEAN|target=0.9|maxDocBits=7 -> FOUR_BIT_SYMMETRIC,precondition=true,oversample=1.25\n"
            + "EUCLIDEAN|target=0.99|maxDocBits=7 -> FOUR_BIT_SYMMETRIC,precondition=true,oversample=1.25\n"
            + "DOT_PRODUCT|target=0.8|maxDocBits=7 -> FOUR_BIT_SYMMETRIC,precondition=false,oversample=1.25\n"
            + "DOT_PRODUCT|target=0.8|maxDocBits=2 -> ONE_BIT_4BIT_QUERY,precondition=true,oversample=1.25\n"
            + "DOT_PRODUCT|target=0.8|maxDocBits=1 -> ONE_BIT_4BIT_QUERY,precondition=true,oversample=1.25\n"
            + "DOT_PRODUCT|target=0.9|maxDocBits=7 -> FOUR_BIT_SYMMETRIC,precondition=false,oversample=1.25\n"
            + "DOT_PRODUCT|target=0.9|maxDocBits=2 -> ONE_BIT_4BIT_QUERY,precondition=true,oversample=1.25\n"
            + "DOT_PRODUCT|target=0.9|maxDocBits=1 -> ONE_BIT_4BIT_QUERY,precondition=true,oversample=1.25\n"
            + "DOT_PRODUCT|target=0.99|maxDocBits=7 -> FOUR_BIT_SYMMETRIC,precondition=false,oversample=1.25\n"
            + "DOT_PRODUCT|target=0.99|maxDocBits=2 -> ONE_BIT_4BIT_QUERY,precondition=true,oversample=1.25\n"
            + "DOT_PRODUCT|target=0.99|maxDocBits=1 -> ONE_BIT_4BIT_QUERY,precondition=true,oversample=1.25\n"
            + "COSINE|target=0.8|maxDocBits=7 -> SEVEN_BIT_SYMMETRIC,precondition=true,oversample=1.25\n"
            + "COSINE|target=0.8|maxDocBits=2 -> TWO_BIT_4BIT_QUERY,precondition=false,oversample=1.25\n"
            + "COSINE|target=0.8|maxDocBits=1 -> ONE_BIT_1BIT_QUERY,precondition=true,oversample=1.25\n"
            + "COSINE|target=0.9|maxDocBits=7 -> SEVEN_BIT_SYMMETRIC,precondition=true,oversample=1.25\n"
            + "COSINE|target=0.9|maxDocBits=2 -> TWO_BIT_4BIT_QUERY,precondition=false,oversample=1.25\n"
            + "COSINE|target=0.9|maxDocBits=1 -> ONE_BIT_1BIT_QUERY,precondition=true,oversample=1.25\n"
            + "COSINE|target=0.99|maxDocBits=7 -> SEVEN_BIT_SYMMETRIC,precondition=true,oversample=1.25\n"
            + "COSINE|target=0.99|maxDocBits=2 -> TWO_BIT_4BIT_QUERY,precondition=false,oversample=1.25\n"
            + "COSINE|target=0.99|maxDocBits=1 -> ONE_BIT_1BIT_QUERY,precondition=true,oversample=1.25\n"
            + "MAXIMUM_INNER_PRODUCT|target=0.8|maxDocBits=7 -> FOUR_BIT_SYMMETRIC,precondition=false,oversample=1.25\n"
            + "MAXIMUM_INNER_PRODUCT|target=0.8|maxDocBits=2 -> TWO_BIT_4BIT_QUERY,precondition=false,oversample=1.25\n"
            + "MAXIMUM_INNER_PRODUCT|target=0.8|maxDocBits=1 -> ONE_BIT_1BIT_QUERY,precondition=true,oversample=1.25\n"
            + "MAXIMUM_INNER_PRODUCT|target=0.9|maxDocBits=7 -> FOUR_BIT_SYMMETRIC,precondition=false,oversample=1.25\n"
            + "MAXIMUM_INNER_PRODUCT|target=0.9|maxDocBits=2 -> TWO_BIT_4BIT_QUERY,precondition=false,oversample=1.25\n"
            + "MAXIMUM_INNER_PRODUCT|target=0.9|maxDocBits=1 -> ONE_BIT_1BIT_QUERY,precondition=true,oversample=1.25\n"
            + "MAXIMUM_INNER_PRODUCT|target=0.99|maxDocBits=7 -> FOUR_BIT_SYMMETRIC,precondition=false,oversample=1.25\n"
            + "MAXIMUM_INNER_PRODUCT|target=0.99|maxDocBits=2 -> TWO_BIT_4BIT_QUERY,precondition=false,oversample=1.25\n"
            + "MAXIMUM_INNER_PRODUCT|target=0.99|maxDocBits=1 -> ONE_BIT_1BIT_QUERY,precondition=true,oversample=1.25\n";
        assertThat(String.join("\n", actual) + "\n", equalTo(expected));
    }
}
