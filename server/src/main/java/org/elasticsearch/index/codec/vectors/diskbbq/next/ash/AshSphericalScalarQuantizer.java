/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq.next.ash;

import java.util.Arrays;

/**
 * Multi-bit spherical scalar quantizer for ASH. Quantizes each dimension of the
 * projected (latent) vector into {@code bitsPerDim} bits using an optimal greedy
 * level selection that maximizes inner product preservation on the unit sphere.
 * <p>
 * This is a port of the Python reference implementation's {@code SphericalScalarQuantizer}.
 */
public final class AshSphericalScalarQuantizer implements AshDimQuantizer {

    private final int bitsPerDim;

    public AshSphericalScalarQuantizer(int bitsPerDim) {
        if (bitsPerDim < 2) {
            throw new IllegalArgumentException("bitsPerDim must be >= 2 for spherical scalar quantizer; use BinaryQuantizer for 1-bit");
        }
        this.bitsPerDim = bitsPerDim;
    }

    @Override
    public int bitsPerDimension() {
        return bitsPerDim;
    }

    @Override
    public QuantizeResult encode(float[][] x) {
        int n = x.length;
        if (n == 0) {
            return new QuantizeResult(new float[0][0], new float[0]);
        }
        int nDims = x[0].length;
        float[][] centeredCodes = new float[n][nDims];
        float[] codeNorms = new float[n];

        for (int i = 0; i < n; i++) {
            float norm = quantizeExact(x[i], centeredCodes[i], nDims);
            codeNorms[i] = norm;
        }
        return new QuantizeResult(centeredCodes, codeNorms);
    }

    /**
     * Greedy optimal quantization for a single vector.
     * Returns the norm of the centered code and writes the code to {@code out}.
     */
    private float quantizeExact(float[] z, float[] out, int d) {
        int numAbsLevels = 1 << (bitsPerDim - 1);

        // Extract signs and absolute values
        float[] signs = new float[d];
        float[] absZ = new float[d];
        for (int j = 0; j < d; j++) {
            signs[j] = z[j] >= 0 ? 1.0f : -1.0f;
            absZ[j] = Math.abs(z[j]);
        }

        // Base level: all at 0.5
        double currentDot = 0;
        for (int j = 0; j < d; j++) {
            currentDot += 0.5 * absZ[j];
        }
        double currentNormSq = 0.25 * d;

        // Find best magnitude for each dimension via greedy event scanning
        int[] bestIdx = new int[d]; // number of level increments beyond base

        if (numAbsLevels > 1) {
            int nSteps = numAbsLevels - 1;
            int k = nSteps * d;

            // Build events: for each (step, dim), critical time = step / absZ[dim]
            // Sort events by critical time and greedily pick the best stopping point
            double[] eventTimes = new double[k];
            int[] eventDims = new int[k];
            int[] eventLevels = new int[k];

            int eventCount = 0;
            for (int step = 1; step <= nSteps; step++) {
                for (int j = 0; j < d; j++) {
                    if (absZ[j] > 0) {
                        eventTimes[eventCount] = (double) step / absZ[j];
                        eventDims[eventCount] = j;
                        eventLevels[eventCount] = step;
                        eventCount++;
                    }
                }
            }

            // Sort events by time
            int[] order = argsort(eventTimes, eventCount);

            // Sweep through events, tracking cumulative dot product and norm
            double cumDot = currentDot;
            double cumNormSq = currentNormSq;
            double bestValue = cumDot / Math.sqrt(cumNormSq);
            int bestStopIdx = -1; // -1 means stop at base

            int[] dimLevelCount = new int[d]; // track how many levels each dim has been incremented

            for (int idx = 0; idx < eventCount; idx++) {
                int oi = order[idx];
                int dim = eventDims[oi];
                int level = eventLevels[oi];

                cumDot += absZ[dim];
                cumNormSq += 2.0 * level;
                dimLevelCount[dim]++;

                // Handle ties: skip if next event has same time
                if (idx + 1 < eventCount) {
                    int nextOi = order[idx + 1];
                    if (eventTimes[oi] == eventTimes[nextOi]) {
                        continue;
                    }
                }

                double value = cumDot / Math.sqrt(cumNormSq);
                if (value > bestValue) {
                    bestValue = value;
                    bestStopIdx = idx;
                }
            }

            // Reconstruct bestIdx from the events up to bestStopIdx
            if (bestStopIdx >= 0) {
                Arrays.fill(dimLevelCount, 0);
                for (int idx = 0; idx <= bestStopIdx; idx++) {
                    int oi = order[idx];
                    dimLevelCount[eventDims[oi]]++;
                }
                System.arraycopy(dimLevelCount, 0, bestIdx, 0, d);
            }
        }

        // Final conversion: centered code = sign * (0.5 + bestIdx)
        double normSq = 0;
        for (int j = 0; j < d; j++) {
            float mag = 0.5f + bestIdx[j];
            out[j] = signs[j] * mag;
            normSq += (double) out[j] * out[j];
        }
        return (float) Math.sqrt(normSq);
    }

    /**
     * Returns indices that sort the first {@code count} elements of {@code values} in ascending order.
     */
    private static int[] argsort(double[] values, int count) {
        Integer[] indices = new Integer[count];
        for (int i = 0; i < count; i++) {
            indices[i] = i;
        }
        Arrays.sort(indices, (a, b) -> Double.compare(values[a], values[b]));
        int[] result = new int[count];
        for (int i = 0; i < count; i++) {
            result[i] = indices[i];
        }
        return result;
    }
}
