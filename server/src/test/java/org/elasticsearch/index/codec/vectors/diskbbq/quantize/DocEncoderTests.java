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
import org.elasticsearch.index.codec.vectors.ash.AshDistortionRateQuantizer;
import org.elasticsearch.simdvec.ESVectorUtil;
import org.elasticsearch.test.ESTestCase;

import java.io.IOException;
import java.util.function.IntFunction;

import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.greaterThan;
import static org.hamcrest.Matchers.greaterThanOrEqualTo;
import static org.hamcrest.Matchers.lessThan;

/**
 * What any {@link DocEncoder} and {@link DocDecoder} have to satisfy, which is all that the code that is written for every
 * scheme (laying out blocks, characterizing a quantizer) relies on. Each scheme is run through the same checks, by the
 * interface alone.
 */
public class DocEncoderTests extends ESTestCase {

    private static final int DIM = 64;
    private static final int BLOCK = 40;

    private record Scheme(String name, int bits, IntFunction<DocEncoder> encoders, double minCosine) {}

    private static Scheme[] schemes() throws IOException {
        return new Scheme[] {
            new Scheme("osq-1", 1, block -> OsqDocEncoder.forField(VectorSimilarityFunction.DOT_PRODUCT, DIM, 1, block), 0.75),
            new Scheme("osq-2", 2, block -> OsqDocEncoder.forField(VectorSimilarityFunction.DOT_PRODUCT, DIM, 2, block), 0.92),
            new Scheme("osq-4", 4, block -> OsqDocEncoder.forField(VectorSimilarityFunction.DOT_PRODUCT, DIM, 4, block), 0.985),
            ash(1, 0.75),
            ash(2, 0.92),
            ash(4, 0.985) };
    }

    private static Scheme ash(int bits, double minCosine) throws IOException {
        AshDistortionRateQuantizer quantizer = AshDistortionRateQuantizer.random(DIM, bits, 4, 7L);
        return new Scheme("ash-" + bits, bits, quantizer::newDocEncoder, minCosine);
    }

    private static float[][] vectors(int n) {
        float[][] vectors = new float[n][DIM];
        for (float[] vector : vectors) {
            for (int d = 0; d < DIM; d++) {
                vector[d] = (float) random().nextGaussian();
            }
        }
        return vectors;
    }

    private static void fill(DocEncoder encoder, float[] centroid, float[][] vectors) {
        encoder.reset(centroid);
        for (float[] vector : vectors) {
            encoder.add(vector);
        }
        encoder.encode();
    }

    public void testLevelsAreWithinTheBitsAndTheirSumIsALane() throws IOException {
        for (Scheme scheme : schemes()) {
            try (DocEncoder encoder = scheme.encoders().apply(BLOCK)) {
                assertThat(scheme.name(), encoder.bits(), equalTo(scheme.bits()));
                float[][] vectors = vectors(BLOCK);
                fill(encoder, vectors(1)[0], vectors);
                assertThat(scheme.name(), encoder.size(), equalTo(BLOCK));
                int[] levels = new int[encoder.latentDimension()];
                DocLanes lanes = new DocLanes();
                for (int i = 0; i < BLOCK; i++) {
                    encoder.read(i, levels, lanes);
                    int sum = 0;
                    for (int level : levels) {
                        assertThat(scheme.name(), level, greaterThanOrEqualTo(0));
                        assertThat(scheme.name(), level, lessThan(1 << scheme.bits()));
                        sum += level;
                    }
                    assertThat(scheme.name(), lanes.sum, equalTo(sum));
                    assertThat(scheme.name(), lanes.scale, greaterThan(0f));
                }
            }
        }
    }

    public void testDecodingGivesBackTheDirectionOfTheVector() throws IOException {
        DocDecoder decoder = AffineDocDecoder.INSTANCE;
        for (Scheme scheme : schemes()) {
            try (DocEncoder encoder = scheme.encoders().apply(BLOCK)) {
                fill(encoder, vectors(1)[0], vectors(BLOCK));
                int latent = encoder.latentDimension();
                int[] levels = new int[latent];
                DocLanes lanes = new DocLanes();
                float[] direction = new float[latent];
                float[] reconstruction = new float[latent];
                double sumOfCosines = 0;
                for (int i = 0; i < BLOCK; i++) {
                    encoder.read(i, levels, lanes);
                    encoder.latent(i, direction);
                    decoder.decode(levels, lanes, reconstruction);
                    double cosine = ESVectorUtil.dotProduct(direction, reconstruction) / Math.sqrt(
                        (double) ESVectorUtil.dotProduct(direction, direction) * ESVectorUtil.dotProduct(reconstruction, reconstruction)
                    );
                    sumOfCosines += cosine;
                }
                // a code is as close to its vector as the codebook can make it on average, not for every vector
                assertThat(scheme.name(), sumOfCosines / BLOCK, greaterThan(scheme.minCosine()));
            }
        }
    }

    public void testABlockCanBeRefilledAndGivesTheSameCodes() throws IOException {
        float[][] vectors = vectors(BLOCK);
        float[] centroid = vectors(1)[0];
        for (Scheme scheme : schemes()) {
            try (DocEncoder encoder = scheme.encoders().apply(BLOCK)) {
                int[] first = new int[encoder.codeLength()];
                int[] again = new int[encoder.codeLength()];
                DocLanes lanes = new DocLanes();
                fill(encoder, centroid, vectors);
                encoder.read(BLOCK - 1, first, lanes);
                float scale = lanes.scale;
                fill(encoder, vectors(1)[0], vectors(BLOCK));
                // a block of other vectors, and then the first block again
                fill(encoder, centroid, vectors);
                assertThat(scheme.name(), encoder.size(), equalTo(BLOCK));
                encoder.read(BLOCK - 1, again, lanes);
                assertArrayEquals(scheme.name(), first, again);
                assertEquals(scheme.name(), scale, lanes.scale, 1e-5f * Math.max(1f, scale));
            }
        }
    }

    public void testDifferentSchemesAreInterchangeableThroughTheInterface() throws IOException {
        // the same checks above, without knowing which scheme is behind an encoder: the dimensions that are exposed are all
        // that is needed to size the buffers
        for (Scheme scheme : schemes()) {
            try (DocEncoder encoder = scheme.encoders().apply(BLOCK)) {
                assertThat(scheme.name(), encoder.dimension(), equalTo(DIM));
                assertThat(scheme.name(), encoder.latentDimension(), lessThan(DIM + 1));
                assertThat(scheme.name(), encoder.maxBlockSize(), equalTo(BLOCK));
            }
        }
    }
}
