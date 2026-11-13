/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.cluster;

import org.elasticsearch.logging.LogManager;
import org.elasticsearch.logging.Logger;

import java.io.IOException;
import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Single threaded implementation of <a href="https://e-means.github.io/">e-means</a>
 * ("e-means: outperforming k-means in one-tenth the epochs"), a mini-batch optimizer for the k-means objective.
 * <p>
 * Instead of full Lloyd iterations, it runs a few epochs of mini-batch updates over shuffled data. Each centroid keeps an
 * exponentially decayed mass {@code N_j} and position {@code c_j}; for each batch the step is {@code alpha(t)} (cosine
 * schedule), {@code N_j <- (1-alpha) N_j + alpha n_j} and {@code c_j} moves towards the batch mean of its points by
 * {@code alpha n_j / N_j}. Every {@value #REVIVAL_PERIOD} batches, centroids with little mass are revived by splitting
 * mass-weighted randomly selected donors. During the last epoch the assignments are recorded and each centroid is
 * replaced by the mean of the points assigned to it, then a final exact assignment pass is made.
 * <p>
 * Batch size and initial step are derived from the scaling laws of the paper as a function of {@code n}, {@code k} and
 * the number of epochs. This clusterer is global: the neighborhoods used by the neighbor-aware variants of
 * {@link KMeansLocal} are not used to restrict assignments.
 *
 * @param <V> the array type for vectors and centroids ({@code float[]} or {@code byte[]})
 */
class EMeansKMeansLocal<V> extends KMeansLocal<V> {

    private static final Logger logger = LogManager.getLogger(EMeansKMeansLocal.class);

    static final double DEFAULT_EPOCHS = 5;
    /** EXPERIMENT ONLY (stage timers). */
    static final AtomicLong T_TRAIN_NS = new AtomicLong();
    static final AtomicLong T_ASSIGN_NS = new AtomicLong();
    /** EXPERIMENT ONLY: lower bound on the number of mini-batch updates, so very small inputs still get enough updates to converge. */
    private static final int MIN_STEPS = Integer.getInteger("bench.emeans.min_steps", 0);
    private static final int REVIVAL_PERIOD = 5;
    private static final float UNUSED_THRESHOLD = 0.1f;
    /** BF16 machine epsilon, the relative offset used to separate a split donor from its child. */
    private static final float SPLIT_EPSILON = 1f / 128f;
    private static final float INITIAL_MASS = 1f / 65536f;
    private static final float MIN_STEP_FRACTION = 0.1f;

    private final double epochs;
    private final long seed;
    /** Number of vectors the centroids are trained on; the final assignment pass always covers all vectors. */
    private final int sampleSize;

    EMeansKMeansLocal(CentroidOps<V> ops, double epochs, long seed) {
        this(ops, epochs, seed, Integer.MAX_VALUE);
    }

    EMeansKMeansLocal(CentroidOps<V> ops, double epochs, long seed, int sampleSize) {
        super(ops);
        if (epochs <= 0) {
            throw new IllegalArgumentException("epochs must be positive, got [" + epochs + "]");
        }
        this.epochs = epochs;
        this.seed = seed;
        this.sampleSize = sampleSize;
    }

    @Override
    protected int numWorkers() {
        return 1;
    }

    @Override
    protected NeighborHood[] computeNeighborhoods(V[] centroids, int clustersPerNeighborhood) throws IOException {
        return NeighborHood.computeNeighborhoods(ops, centroids, clustersPerNeighborhood);
    }

    /** Batch size from the (throughput) scaling law, rounded to a power of two and capped at {@code n / 2}. */
    static int batchSize(int n, int k, double epochs) {
        double b = 0.035134 * Math.pow(k, 0.134749) * Math.pow(n, 0.876473) * Math.pow(epochs, 0.813344);
        long pow2 = Math.round(Math.pow(2, Math.round(Math.log(Math.max(b, 1)) / Math.log(2))));
        return (int) Math.max(1, Math.min(pow2, Math.max(1, n / 2)));
    }

    /** Initial step size from the scaling law, clamped to {@code [1e-4, 1]}. */
    static double initialStep(int n, int k, int batch, double epochs) {
        double a = 0.8244 * Math.pow(k, -0.4617) * Math.pow(batch, 0.9003) * Math.pow(n, -0.4498) * Math.pow(epochs, -0.1057);
        return Math.max(1e-4, Math.min(1, a));
    }

    @Override
    protected void innerCluster(ClusteringVectorValues<V> vectors, KMeansResult<V> kMeansResult, NeighborHood[] neighborhoods)
        throws IOException {
        final V[] centroids = kMeansResult.centroids();
        final int k = centroids.length;
        final int n = vectors.size();
        final int dim = vectors.dimension();
        final long t0 = System.nanoTime();
        // centroids are trained on (a random sample of) the vectors; assignments always cover all of them
        final ClusteringVectorValues<V> train = sampleSize < n
            ? ClusteringVectorValuesSlice.createRandomSlice(vectors, sampleSize, seed)
            : vectors;
        final int nTrain = train.size();
        final int[] assignments = kMeansResult.assignments();
        assert assignments.length == n;
        if (k == 1 || n == 0) {
            Arrays.fill(assignments, 0);
            return;
        }

        // float working copy of the centroids; for float ops this array is shared with the native centroid
        final float[][] c = new float[k][dim];
        final boolean[] shared = new boolean[1];
        for (int j = 0; j < k; j++) {
            ops.addScaled(1f, centroids[j], c[j]);
        }
        final float[] mass = new float[k];
        Arrays.fill(mass, INITIAL_MASS);
        final V[] work = ops.newCentroidArrayShallow(k);
        for (int j = 0; j < k; j++) {
            work[j] = ops.centroidFromFloat(c[j], dim);
            shared[0] |= (Object) work[j] == c[j];
        }
        final boolean sharedWithNative = shared[0];

        final int batch = batchSize(nTrain, k, epochs);
        final int stepsPerEpoch = Math.max(1, nTrain / batch);
        final int totalSteps = Math.max(Math.max(1, MIN_STEPS), (int) Math.round(epochs * stepsPerEpoch));
        final int lastEpochStart = Math.max(0, totalSteps - stepsPerEpoch);
        final double alpha0 = initialStep(nTrain, k, batch, epochs);
        final float unusedThreshold = dim <= 2 ? 0f : UNUSED_THRESHOLD;

        final Random random = new Random(seed);
        final int[] perm = new int[nTrain];
        for (int i = 0; i < nTrain; i++) {
            perm[i] = i;
        }
        shuffle(perm, random);

        final int[] counts = new int[k];
        final float[][] sums = new float[k][dim];
        final int[] touched = new int[k];
        final int[] batchAssign = new int[batch];
        final float[] dists = new float[4];
        // last-epoch accumulation, used to finalize centroids as the mean of their assigned points
        final long[] finalCounts = new long[k];
        final float[][] finalSums = new float[k][dim];

        final long t1 = System.nanoTime();
        int pos = 0;
        for (int t = 0; t < totalSteps; t++) {
            if (pos + batch > nTrain) {
                shuffle(perm, random);
                pos = 0;
            }
            final double alpha = alpha0 * (MIN_STEP_FRACTION + (1 - MIN_STEP_FRACTION) * 0.5 * (1 + Math.cos(Math.PI * t / totalSteps)));
            final boolean lastEpoch = t >= lastEpochStart;

            // assign
            int numTouched = 0;
            for (int i = 0; i < batch; i++) {
                V v = train.vectorValue(perm[pos + i]);
                int best = nearest(v, work, dists);
                batchAssign[i] = best;
                if (counts[best]++ == 0) {
                    touched[numTouched++] = best;
                }
                ops.addScaled(1f, v, sums[best]);
                if (lastEpoch) {
                    finalCounts[best]++;
                    ops.addScaled(1f, v, finalSums[best]);
                }
            }
            pos += batch;

            // update: N <- gamma N + alpha n_j for touched clusters, N <- gamma N otherwise
            final float gamma = (float) (1 - alpha);
            for (int j = 0; j < k; j++) {
                mass[j] *= gamma;
            }
            for (int ti = 0; ti < numTouched; ti++) {
                int j = touched[ti];
                int nj = counts[j];
                mass[j] += (float) alpha * nj;
                float w = (float) alpha * nj / mass[j];
                float inv = 1f / nj;
                float[] cj = c[j];
                float[] sj = sums[j];
                for (int d = 0; d < dim; d++) {
                    cj[d] += w * (sj[d] * inv - cj[d]);
                }
                Arrays.fill(sj, 0f);
                counts[j] = 0;
                if (sharedWithNative == false) {
                    work[j] = ops.centroidFromFloat(cj, dim);
                }
            }

            if (unusedThreshold > 0 && (t + 1) % REVIVAL_PERIOD == 0) {
                reviveUnused(c, mass, work, unusedThreshold, dim, random, sharedWithNative);
            }
        }

        final long t2 = System.nanoTime();
        // finalize: mean of the points assigned during the last epoch
        for (int j = 0; j < k; j++) {
            if (finalCounts[j] > 0) {
                float inv = 1f / finalCounts[j];
                for (int d = 0; d < dim; d++) {
                    c[j][d] = finalSums[j][d] * inv;
                }
                if (sharedWithNative == false) {
                    work[j] = ops.centroidFromFloat(c[j], dim);
                }
            }
        }

        // publish the centroids and make the assignments consistent with them
        for (int j = 0; j < k; j++) {
            centroids[j] = work[j];
        }
        for (int i = 0; i < n; i++) {
            assignments[i] = nearest(vectors.vectorValue(i), centroids, dists);
        }
        T_TRAIN_NS.addAndGet(t2 - t0);
        T_ASSIGN_NS.addAndGet(System.nanoTime() - t2);
        if (logger.isDebugEnabled()) {
            final long t3 = System.nanoTime();
            logger.debug(
                "e-means n [{}] train [{}] k [{}] dim [{}] batch [{}] steps [{}] init [{}ms] train-loop [{}ms] final-assign [{}ms]",
                n,
                nTrain,
                k,
                dim,
                batch,
                totalSteps,
                (t1 - t0) / 1_000_000,
                (t2 - t1) / 1_000_000,
                (t3 - t2) / 1_000_000
            );
        }
    }

    private int nearest(V v, V[] cs, float[] dists) {
        int k = cs.length;
        int best = 0;
        float bestDist = Float.MAX_VALUE;
        int j = 0;
        for (; j + 3 < k; j += 4) {
            ops.squareDistanceBulk(v, cs[j], cs[j + 1], cs[j + 2], cs[j + 3], 0, dists);
            for (int b = 0; b < 4; b++) {
                if (dists[b] < bestDist) {
                    bestDist = dists[b];
                    best = j + b;
                }
            }
        }
        for (; j < k; j++) {
            float dist = ops.squareDistance(v, cs[j]);
            if (dist < bestDist) {
                bestDist = dist;
                best = j;
            }
        }
        return best;
    }

    /**
     * Revives centroids whose mass fell below {@code threshold * mean(mass)} by splitting donors chosen without replacement
     * with probability proportional to their mass (Gumbel top-m trick). Donor and child get half the donor's mass and
     * positions offset by alternating-sign relative perturbations.
     */
    private void reviveUnused(float[][] c, float[] mass, V[] work, float threshold, int dim, Random random, boolean sharedWithNative) {
        final int k = c.length;
        double mean = 0;
        for (float m : mass) {
            mean += m;
        }
        mean /= k;
        final double cutoff = threshold * mean;
        int numUnused = 0;
        for (int j = 0; j < k; j++) {
            if (mass[j] < cutoff) {
                numUnused++;
            }
        }
        if (numUnused == 0) {
            return;
        }
        int[] unused = new int[numUnused];
        int numDonors = 0;
        int u = 0;
        for (int j = 0; j < k; j++) {
            if (mass[j] < cutoff) {
                unused[u++] = j;
            } else if (mass[j] > 0) {
                numDonors++;
            }
        }
        final int splits = Math.min(numUnused, numDonors);
        if (splits == 0) {
            return;
        }
        // Gumbel keys: sort donors by log(N) + Gumbel noise, descending
        int[] donors = new int[numDonors];
        double[] keys = new double[numDonors];
        int di = 0;
        for (int j = 0; j < k; j++) {
            if (mass[j] >= cutoff && mass[j] > 0) {
                donors[di] = j;
                keys[di] = Math.log(mass[j]) - Math.log(-Math.log(1 - random.nextDouble()));
                di++;
            }
        }
        for (int s = 0; s < splits; s++) {
            int best = s;
            for (int i = s + 1; i < numDonors; i++) {
                if (keys[i] > keys[best]) {
                    best = i;
                }
            }
            int tmpD = donors[s];
            donors[s] = donors[best];
            donors[best] = tmpD;
            double tmpK = keys[s];
            keys[s] = keys[best];
            keys[best] = tmpK;

            int donor = donors[s];
            int child = unused[s];
            float[] cd = c[donor];
            float[] cc = c[child];
            for (int d = 0; d < dim; d++) {
                float base = cd[d] != 0 ? cd[d] : 1e-3f;
                float sign = (d & 1) == 0 ? 1f : -1f;
                float offset = sign * SPLIT_EPSILON * Math.abs(base);
                cc[d] = cd[d] + offset;
                cd[d] = cd[d] - offset;
            }
            float half = mass[donor] / 2;
            mass[donor] = half;
            mass[child] = half;
            if (sharedWithNative == false) {
                work[donor] = ops.centroidFromFloat(cd, dim);
                work[child] = ops.centroidFromFloat(cc, dim);
            }
        }
    }

    private static void shuffle(int[] a, Random random) {
        for (int i = a.length - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int tmp = a[i];
            a[i] = a[j];
            a[j] = tmp;
        }
    }
}
