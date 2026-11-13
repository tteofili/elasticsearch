/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.cluster;

import org.elasticsearch.test.ESTestCase;

import java.io.IOException;
import java.util.Arrays;

import static org.hamcrest.Matchers.greaterThanOrEqualTo;
import static org.hamcrest.Matchers.lessThanOrEqualTo;

public class EMeansKMeansLocalTests extends ESTestCase {

    private static <V> double mse(ClusteringVectorValues<V> vectors, CentroidOps<V> ops, V[] centroids, int[] assignments)
        throws IOException {
        double sum = 0;
        for (int i = 0; i < vectors.size(); i++) {
            sum += ops.squareDistance(vectors.vectorValue(i), centroids[assignments[i]]);
        }
        return sum / vectors.size();
    }

    /**
     * Runs e-means and 25 Lloyd iterations from the same initial centroids and returns the ratio of their MSEs. Both algorithms can end in
     * different local minima on tiny random inputs, so callers assert on several independent datasets rather than on a single ratio.
     */
    private <V> double checkAgainstLloyd(CentroidOps<V> ops, ClusteringVectorValues<V> vectors, int k) throws IOException {
        V[] eInit = KMeansLocal.pickInitialCentroids(vectors, k, ops);
        KMeansResult<V> e = new KMeansResult<>(eInit, new int[vectors.size()]);
        new EMeansKMeansLocal<>(ops, EMeansKMeansLocal.DEFAULT_EPOCHS, 42L).cluster(vectors, e);

        // assignments must be valid, consistent with the (possibly compacted) centroids, and counts must add up
        int total = 0;
        for (int count : e.clusterCounts()) {
            total += count;
        }
        assertEquals(vectors.size(), total);
        for (int i = 0; i < vectors.size(); i++) {
            int a = e.assignments()[i];
            assertThat(a, greaterThanOrEqualTo(0));
            assertThat(a, lessThanOrEqualTo(e.centroids().length - 1));
        }

        V[] lInit = KMeansLocal.pickInitialCentroids(vectors, k, ops);
        KMeansResult<V> l = new KMeansResult<>(lInit, new int[vectors.size()]);
        new LloydKMeansLocalSerial<>(ops, vectors.size(), 25).cluster(vectors, l);

        double eMse = mse(vectors, ops, e.centroids(), e.assignments());
        double lMse = mse(vectors, ops, l.centroids(), l.assignments());
        return eMse / lMse;
    }

    public void testFloatQuality() throws IOException {
        // From the same random initial centroids both algorithms usually converge to the same optimum (ratio ~1), but on a few percent of
        // tiny random datasets one of them merges two true clusters and ends many times worse (in either direction, e-means being the
        // better
        // one more often). The mean is dominated by such outliers, so assert on the median ratio over several independent datasets.
        int trials = 9;
        double[] ratios = new double[trials];
        for (int trial = 0; trial < trials; trial++) {
            int k = randomIntBetween(2, 20);
            int dims = randomIntBetween(3, 16);
            ratios[trial] = checkAgainstLloyd(
                CentroidOps.FLOAT,
                KMeansTestData.generateFloatData(randomIntBetween(k * 200, k * 400), dims, k),
                k
            );
        }
        Arrays.sort(ratios);
        assertThat(Arrays.toString(ratios), ratios[trials / 2], lessThanOrEqualTo(1.1));
    }

    public void testByteQuality() throws IOException {
        // From the same random initial centroids both algorithms usually converge to the same optimum (ratio ~1), but on a few percent of
        // tiny random datasets one of them merges two true clusters and ends many times worse (in either direction, e-means being the
        // better
        // one more often). The mean is dominated by such outliers, so assert on the median ratio over several independent datasets.
        int trials = 9;
        double[] ratios = new double[trials];
        for (int trial = 0; trial < trials; trial++) {
            int k = randomIntBetween(2, 20);
            int dims = randomIntBetween(3, 16);
            ratios[trial] = checkAgainstLloyd(
                CentroidOps.BYTE,
                KMeansTestData.generateByteData(randomIntBetween(k * 200, k * 400), dims, k),
                k
            );
        }
        Arrays.sort(ratios);
        assertThat(Arrays.toString(ratios), ratios[trials / 2], lessThanOrEqualTo(1.1));
    }

    public void testSingleCluster() throws IOException {
        var vectors = KMeansTestData.generateFloatData(100, 4, 1);
        float[][] init = KMeansLocal.pickInitialCentroids(vectors, 1, CentroidOps.FLOAT);
        KMeansResult<float[]> r = new KMeansResult<>(init, new int[100]);
        new EMeansKMeansLocal<>(CentroidOps.FLOAT, 5, 1L).cluster(vectors, r);
        assertEquals(1, r.centroids().length);
    }

    public void testBatchLawsAreSane() {
        for (int i = 0; i < 100; i++) {
            int n = randomIntBetween(2, 5_000_000);
            int k = randomIntBetween(2, 100_000);
            int b = EMeansKMeansLocal.batchSize(n, k, 5);
            assertThat(b, greaterThanOrEqualTo(1));
            assertThat(b, lessThanOrEqualTo(Math.max(1, n / 2)));
            double a = EMeansKMeansLocal.initialStep(n, k, b, 5);
            assertThat(a, greaterThanOrEqualTo(1e-4));
            assertThat(a, lessThanOrEqualTo(1.0));
        }
    }

    public void testRejectsNonPositiveEpochs() {
        expectThrows(IllegalArgumentException.class, () -> new EMeansKMeansLocal<>(CentroidOps.FLOAT, 0, 1L));
    }
}
