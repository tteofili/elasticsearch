/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq.calibrate;

import org.apache.lucene.index.FloatVectorValues;
import org.elasticsearch.index.codec.vectors.diskbbq.CentroidIndex;
import org.elasticsearch.index.codec.vectors.diskbbq.CentroidSupplier;
import org.elasticsearch.index.codec.vectors.diskbbq.calibrate.DistortionRateTestUtils.Dataset;
import org.elasticsearch.test.ESTestCase;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import static org.elasticsearch.index.codec.vectors.diskbbq.calibrate.DistortionRateTestUtils.clusteredDataset;
import static org.elasticsearch.index.codec.vectors.diskbbq.calibrate.DistortionRateTestUtils.values;
import static org.hamcrest.Matchers.equalTo;

public class ClusteredSampleTests extends ESTestCase {

    private static final int DIM = 16;

    /** Counts the vectors that are read, which is what an off-heap corpus would pay for. */
    private static final class CountingValues extends FloatVectorValues {
        private final FloatVectorValues delegate;
        final AtomicInteger reads;

        CountingValues(FloatVectorValues delegate, AtomicInteger reads) {
            this.delegate = delegate;
            this.reads = reads;
        }

        @Override
        public float[] vectorValue(int ord) throws IOException {
            reads.incrementAndGet();
            return delegate.vectorValue(ord);
        }

        @Override
        public FloatVectorValues copy() {
            return new CountingValues(delegate, reads);
        }

        @Override
        public int dimension() {
            return delegate.dimension();
        }

        @Override
        public int size() {
            return delegate.size();
        }

        @Override
        public DocIndexIterator iterator() {
            return delegate.iterator();
        }
    }

    private static CentroidSupplier centroids(Dataset data) {
        return CentroidSupplier.fromArray(data.centroids(), CentroidIndex.NO_INDEX, DIM);
    }

    /** A random subset of the corpus in random order, so that sample positions and corpus ordinals differ. */
    private static ClusteredSample subsetOf(Dataset data, FloatVectorValues vectors, int size) {
        List<Integer> all = new ArrayList<>();
        for (int i = 0; i < data.docs().length; i++) {
            all.add(i);
        }
        Collections.shuffle(all, random());
        int[] ordinals = new int[size];
        int[] assignment = new int[size];
        for (int i = 0; i < size; i++) {
            ordinals[i] = all.get(i);
            assignment[i] = data.assignment()[ordinals[i]];
        }
        return new ClusteredSample(vectors, ordinals, centroids(data), assignment);
    }

    public void testEverySampledVectorNeedsALandmark() {
        Dataset data = clusteredDataset(5, 3, DIM, 1, randomLong());
        IllegalArgumentException e = expectThrows(
            IllegalArgumentException.class,
            () -> new ClusteredSample(values(data.docs()), new int[3], centroids(data), new int[2])
        );
        assertThat(e.getMessage(), equalTo("one assignment per sampled vector: 3 != 2"));
    }

    public void testVectorsAndLandmarksAreReadThroughTheSampleOrdinals() throws IOException {
        Dataset data = clusteredDataset(10, 4, DIM, 1, randomLong());
        int size = randomIntBetween(5, 30);
        ClusteredSample sample = subsetOf(data, values(data.docs()), size);
        assertEquals(size, sample.size());
        assertEquals(DIM, sample.dimension());
        for (int i = 0; i < size; i++) {
            int ordinal = sample.ordinals()[i];
            assertArrayEquals(data.docs()[ordinal], sample.vector(i), 0f);
            assertArrayEquals(data.centroids()[data.assignment()[ordinal]], sample.centroid(i), 0f);
        }
    }

    public void testResidualsAreTheVectorsMinusTheirLandmarks() throws IOException {
        Dataset data = clusteredDataset(10, 4, DIM, 1, randomLong());
        int size = randomIntBetween(5, 30);
        ClusteredSample sample = subsetOf(data, values(data.docs()), size);
        DistortionRateQuantizer.Residuals residuals = sample.residuals(size);
        assertEquals(size, residuals.size());
        for (int i = 0; i < size; i++) {
            float[] residual = residuals.residual(i);
            float[] vector = data.docs()[sample.ordinals()[i]];
            float[] centroid = data.centroids()[sample.assignment()[i]];
            for (int d = 0; d < DIM; d++) {
                assertEquals(vector[d] - centroid[d], residual[d], 0f);
            }
        }
    }

    public void testResidualsAreLimitedToTheSampleSize() {
        Dataset data = clusteredDataset(10, 4, DIM, 1, randomLong());
        ClusteredSample sample = subsetOf(data, values(data.docs()), 20);
        assertEquals(7, sample.residuals(7).size());
        assertEquals(20, sample.residuals(1000).size());
    }

    public void testResidualsDoNotWriteIntoTheCorpus() throws IOException {
        Dataset data = clusteredDataset(10, 4, DIM, 1, randomLong());
        float[][] before = new float[data.docs().length][];
        for (int i = 0; i < before.length; i++) {
            before[i] = data.docs()[i].clone();
        }
        ClusteredSample sample = subsetOf(data, values(data.docs()), 30);
        DistortionRateQuantizer.Residuals residuals = sample.residuals(30);
        for (int i = 0; i < 30; i++) {
            residuals.residual(i);
        }
        for (int i = 0; i < before.length; i++) {
            assertArrayEquals(before[i], data.docs()[i], 0f);
        }
    }

    public void testVectorsAreReadOnlyWhenAskedFor() throws IOException {
        Dataset data = clusteredDataset(10, 4, DIM, 1, randomLong());
        AtomicInteger reads = new AtomicInteger();
        ClusteredSample sample = subsetOf(data, new CountingValues(values(data.docs()), reads), 30);
        // a sample is a view: building it, sizing it and asking for residuals read nothing
        sample.size();
        sample.dimension();
        DistortionRateQuantizer.Residuals residuals = sample.residuals(10);
        assertThat(reads.get(), equalTo(0));
        // and each residual that is pulled costs one vector
        for (int i = 0; i < 10; i++) {
            residuals.residual(i);
            assertThat(reads.get(), equalTo(i + 1));
        }
    }
}
