/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq.next;

import org.elasticsearch.Build;
import org.elasticsearch.index.IndexVersion;
import org.elasticsearch.index.codec.vectors.ash.AshDistortionRateQuantizer;
import org.elasticsearch.index.codec.vectors.diskbbq.QuantizerScheme;
import org.elasticsearch.index.codec.vectors.diskbbq.calibrate.ClusteredSample;
import org.elasticsearch.index.codec.vectors.diskbbq.calibrate.CodebookProfile;
import org.elasticsearch.index.codec.vectors.diskbbq.calibrate.DistortionRateQuantizer;
import org.elasticsearch.index.codec.vectors.diskbbq.calibrate.DistortionRateTestUtils.Dataset;
import org.elasticsearch.index.codec.vectors.diskbbq.calibrate.OsqDistortionRateQuantizer;
import org.elasticsearch.index.codec.vectors.diskbbq.calibrate.RotationKind;
import org.elasticsearch.index.mapper.vectors.DenseVectorFieldMapper.BBQIVFIndexOptions;
import org.elasticsearch.test.ESTestCase;

import java.io.IOException;
import java.util.Arrays;

import static org.elasticsearch.index.codec.vectors.diskbbq.calibrate.DistortionRateTestUtils.byteBackedSampleOf;
import static org.elasticsearch.index.codec.vectors.diskbbq.calibrate.DistortionRateTestUtils.clusteredDataset;
import static org.elasticsearch.index.codec.vectors.diskbbq.calibrate.DistortionRateTestUtils.roundedToBytes;
import static org.hamcrest.Matchers.closeTo;
import static org.hamcrest.Matchers.greaterThan;
import static org.hamcrest.Matchers.instanceOf;
import static org.hamcrest.Matchers.lessThanOrEqualTo;

/** A field's mapping options lead, through its scheme, to a quantizer the distortion-rate estimator can work with. */
public class QuantizerSchemesDistortionRateTests extends ESTestCase {

    private static final int DIM = 128;

    private static BBQIVFIndexOptions options(BBQIVFIndexOptions.QuantizationType type, boolean precondition, Integer bits) {
        return new BBQIVFIndexOptions(384, -1, 0.0d, false, null, IndexVersion.current(), precondition, bits, true, null, type, false);
    }

    public void testOsqSchemeFollowsThePreconditioner() throws IOException {
        DistortionRateQuantizer plain = QuantizerSchemes.OSQ.distortionRateQuantizer(
            options(BBQIVFIndexOptions.QuantizationType.OSQ, false, 1),
            DIM,
            null,
            randomLong()
        );
        assertThat(plain, instanceOf(OsqDistortionRateQuantizer.class));
        assertEquals(RotationKind.NONE, plain.rotation());
        assertEquals(1, plain.bits());
        assertEquals(DIM, plain.latentDimension());

        DistortionRateQuantizer rotated = QuantizerSchemes.OSQ.distortionRateQuantizer(
            options(BBQIVFIndexOptions.QuantizationType.OSQ, true, 4),
            DIM,
            null,
            randomLong()
        );
        assertEquals(RotationKind.BLOCK_DIAGONAL, rotated.rotation());
        assertEquals(4, rotated.bits());
    }

    public void testAshSchemeLearnsItsProjectionFromTheSample() throws IOException {
        assumeTrue("ASH is only available in snapshot builds", Build.current().isSnapshot());
        Dataset data = clusteredDataset(30, 12, DIM, 4, 51);
        ClusteredSample sample = data.sample();
        BBQIVFIndexOptions ash = options(BBQIVFIndexOptions.QuantizationType.ASH, false, 2);
        DistortionRateQuantizer quantizer = QuantizerSchemes.ASH.distortionRateQuantizer(ash, DIM, sample, randomLong());
        assertThat(quantizer, instanceOf(AshDistortionRateQuantizer.class));
        assertEquals(2, quantizer.bits());
        assertThat(quantizer.latentDimension(), lessThanOrEqualTo(DIM));
        expectThrows(NullPointerException.class, () -> QuantizerSchemes.ASH.distortionRateQuantizer(ash, DIM, null, randomLong()));
    }

    public void testAshSchemeLearnsTheSameFromByteVectorsAsFromTheFloatsTheyHold() throws IOException {
        assumeTrue("ASH is only available in snapshot builds", Build.current().isSnapshot());
        Dataset data = roundedToBytes(clusteredDataset(30, 12, DIM, 4, 55));
        BBQIVFIndexOptions ash = options(BBQIVFIndexOptions.QuantizationType.ASH, false, 2);
        long seed = randomLong();
        DistortionRateQuantizer fromFloats = QuantizerSchemes.ASH.distortionRateQuantizer(ash, DIM, data.sample(), seed);
        DistortionRateQuantizer fromBytes = QuantizerSchemes.ASH.distortionRateQuantizer(ash, DIM, byteBackedSampleOf(data), seed);
        assertEquals(fromFloats.latentDimension(), fromBytes.latentDimension());
        assertThat(
            CodebookProfile.sample(fromBytes, 500, 1).angularError(),
            closeTo(CodebookProfile.sample(fromFloats, 500, 1).angularError(), 0.02)
        );
    }

    public void testTheSeedReachesWhatAshLearns() throws IOException {
        assumeTrue("ASH is only available in snapshot builds", Build.current().isSnapshot());
        Dataset data = clusteredDataset(30, 12, DIM, 4, 57);
        BBQIVFIndexOptions ash = options(BBQIVFIndexOptions.QuantizationType.ASH, false, 2);
        long seed = randomLong();
        float[] probe = data.docs()[0];
        float[] first = QuantizerSchemes.ASH.distortionRateQuantizer(ash, DIM, data.sample(), seed).project(probe);
        float[] again = QuantizerSchemes.ASH.distortionRateQuantizer(ash, DIM, data.sample(), seed).project(probe);
        float[] other = QuantizerSchemes.ASH.distortionRateQuantizer(ash, DIM, data.sample(), seed + 1).project(probe);
        assertArrayEquals(first, again, 1e-2f);
        assertFalse("another seed learns another projection", Arrays.equals(first, other));
    }

    public void testEverySchemeGivesAQuantizerTheEstimatorCanCharacterize() throws IOException {
        assumeTrue("ASH is only available in snapshot builds", Build.current().isSnapshot());
        Dataset data = clusteredDataset(30, 12, DIM, 4, 53);
        ClusteredSample sample = data.sample();
        for (var scheme : new QuantizerScheme[] { QuantizerSchemes.OSQ, QuantizerSchemes.ASH }) {
            BBQIVFIndexOptions options = options(
                scheme == QuantizerSchemes.OSQ ? BBQIVFIndexOptions.QuantizationType.OSQ : BBQIVFIndexOptions.QuantizationType.ASH,
                true,
                scheme == QuantizerSchemes.OSQ ? 1 : 2
            );
            DistortionRateQuantizer quantizer = scheme.distortionRateQuantizer(options, DIM, sample, randomLong());
            CodebookProfile profile = CodebookProfile.sample(quantizer, 500, 1);
            // the cosine between a direction and its code is high, and the same abstraction yields it for both
            assertThat(scheme.name(), profile.meanCosine(), greaterThan(0.75));
            assertThat(scheme.name(), profile.angularError(), closeTo(2 * (1 - profile.meanCosine()), 1e-9));
        }
    }
}
