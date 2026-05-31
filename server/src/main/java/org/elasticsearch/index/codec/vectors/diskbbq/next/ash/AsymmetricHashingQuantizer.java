/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq.next.ash;

import java.util.Random;

/**
 * Asymmetric Hashing quantizer. Learns a projection matrix W that maps vectors from
 * original space to a low-dimensional latent space optimized for quantization fidelity.
 * <p>
 * The algorithm:
 * <ol>
 *   <li>KMeans clustering partitions vectors (handled externally via HierarchicalKMeans)</li>
 *   <li>Vectors are centered by subtracting their cluster centroid and normalized</li>
 *   <li>A rotation matrix W is learned (PCA init + Procrustes iterations) to maximize
 *       inner product preservation after quantization in the projected space</li>
 *   <li>Vectors are projected via W, then quantized (binary or multi-bit spherical)</li>
 *   <li>Per-vector scale and offset (float16) are stored for dot product reconstruction</li>
 * </ol>
 * <p>
 * At query time, the query is projected via W but NOT quantized (asymmetric scoring),
 * yielding higher recall than symmetric approaches.
 */
public final class AsymmetricHashingQuantizer {

    /** Training method for the projection matrix W. */
    public enum Method {
        /** Learn W via PCA + iterative Procrustes optimization. */
        LEARNED,
        /** Use a random orthogonal matrix (no training iterations). */
        RANDOM
    }

    private final int totalBits;
    private final int bitsPerDim;
    private final Method method;
    private final int nTrainingIterations;
    private final int trainingFactor;
    private final long seed;
    private final AshDimQuantizer quantizer;

    /**
     * @param totalBits total bits budget per vector (header + body)
     * @param bitsPerDim bits per projected dimension in the body
     * @param method training method for W
     * @param nTrainingIterations number of Procrustes iterations (for LEARNED)
     * @param trainingFactor multiplier on dimension for training sample size
     * @param seed random seed
     */
    public AsymmetricHashingQuantizer(
        int totalBits,
        int bitsPerDim,
        Method method,
        int nTrainingIterations,
        int trainingFactor,
        long seed
    ) {
        if (totalBits <= 0) {
            throw new IllegalArgumentException("totalBits must be positive");
        }
        if (bitsPerDim <= 0) {
            throw new IllegalArgumentException("bitsPerDim must be positive");
        }
        this.totalBits = totalBits;
        this.bitsPerDim = bitsPerDim;
        this.method = method;
        this.nTrainingIterations = nTrainingIterations;
        this.trainingFactor = trainingFactor;
        this.seed = seed;

        if (bitsPerDim == 1) {
            this.quantizer = new AshBinaryQuantizer();
        } else {
            this.quantizer = new AshSphericalScalarQuantizer(bitsPerDim);
        }
    }

    /**
     * Computes the number of header bits for a given cluster count.
     * In IVF layout, cluster assignment is implicit (each posting list belongs to a cluster),
     * so header is only 2 × 16 bits (scale + offset as float16).
     */
    public static int headerBits(int nClusters) {
        return 32; // scale_f16 + offset_f16
    }

    /**
     * Number of projected dimensions given a cluster count.
     * If the configured totalBits is insufficient for any body bits (header exceeds totalBits),
     * we auto-expand to guarantee at least {@link #MIN_PROJECTED_DIMS} projected dimensions.
     */
    public int nDims(int nClusters) {
        int header = headerBits(nClusters);
        int body = totalBits - header;
        if (body <= 0) {
            // Auto-expand: totalBits is insufficient for this cluster count
            body = MIN_PROJECTED_DIMS * bitsPerDim;
        }
        return body / bitsPerDim;
    }

    /** Minimum number of projected dimensions when totalBits is insufficient for the header. */
    private static final int MIN_PROJECTED_DIMS = 8;

    /**
     * Trains the projection matrix W on the given vectors and their cluster assignments.
     *
     * @param vectors all vectors in the segment, shape (nVectors, originalDim)
     * @param centroids cluster centroids, shape (nClusters, originalDim)
     * @param assignments cluster assignment per vector, length nVectors
     * @return the learned projection matrix W, shape (originalDim, nDims)
     */
    public float[][] train(float[][] vectors, float[][] centroids, int[] assignments) {
        int nClusters = centroids.length;
        int nDims = nDims(nClusters);
        int originalDim = vectors[0].length;

        // Too few vectors for meaningful PCA training; fall back to random projection
        if (method == Method.LEARNED && vectors.length < nDims * 2) {
            return randomOrthogonal(originalDim, nDims);
        }

        // Center and normalize vectors
        float[][] xNormalized = new float[vectors.length][originalDim];
        for (int i = 0; i < vectors.length; i++) {
            float[] centroid = centroids[assignments[i]];
            double normSq = 0;
            for (int d = 0; d < originalDim; d++) {
                xNormalized[i][d] = vectors[i][d] - centroid[d];
                normSq += (double) xNormalized[i][d] * xNormalized[i][d];
            }
            float invNorm = (float) (1.0 / Math.sqrt(normSq));
            if (Float.isFinite(invNorm)) {
                for (int d = 0; d < originalDim; d++) {
                    xNormalized[i][d] *= invNorm;
                }
            }
        }

        if (method == Method.RANDOM) {
            return randomOrthogonal(originalDim, nDims);
        }

        // LEARNED: PCA init + Procrustes
        int trainingSize = Math.min(originalDim * trainingFactor, vectors.length);
        float[][] xTraining = sampleRows(xNormalized, trainingSize);
        return learnedTraining(xTraining, originalDim, nDims);
    }

    /**
     * Encodes vectors using the trained projection matrix W.
     *
     * @param vectors all vectors, shape (nVectors, originalDim)
     * @param centroids cluster centroids, shape (nClusters, originalDim)
     * @param assignments cluster assignment per vector
     * @param w the trained projection matrix, shape (originalDim, nDims)
     * @return encoding result with codes, scales, and offsets
     */
    public AsymmetricHashingResult encode(float[][] vectors, float[][] centroids, int[] assignments, float[][] w) {
        int nClusters = centroids.length;
        int originalDim = vectors[0].length;
        int nDims = w[0].length;
        int nVectors = vectors.length;

        // Project centered+normalized vectors via W
        float[][] xLatent = new float[nVectors][nDims];
        float[] norms = new float[nVectors];
        float[] scales = new float[nVectors];
        float[] offsets = new float[nVectors];

        for (int i = 0; i < nVectors; i++) {
            float[] centroid = centroids[assignments[i]];
            // Center
            double normSq = 0;
            float[] centered = new float[originalDim];
            for (int d = 0; d < originalDim; d++) {
                centered[d] = vectors[i][d] - centroid[d];
                normSq += (double) centered[d] * centered[d];
            }
            norms[i] = (float) Math.sqrt(normSq);

            // Normalize
            float invNorm = norms[i] > 0 ? 1.0f / norms[i] : 0;
            for (int d = 0; d < originalDim; d++) {
                centered[d] *= invNorm;
            }

            // Project: xLatent[i] = centered @ W
            for (int j = 0; j < nDims; j++) {
                double sum = 0;
                for (int d = 0; d < originalDim; d++) {
                    sum += (double) centered[d] * w[d][j];
                }
                xLatent[i][j] = (float) sum;
            }
        }

        // Quantize in latent space
        AshDimQuantizer.QuantizeResult qr = quantizer.encode(xLatent);
        float[][] xEnc = qr.centeredCodes();
        float[] codeNorms = qr.codeNorms();

        // Compute scale and offset
        for (int i = 0; i < nVectors; i++) {
            // scale = norm / codeNorm (stored as float16)
            scales[i] = codeNorms[i] > 0 ? norms[i] / codeNorms[i] : 0;

            // offset = dot(vector, centroid) - dot(centroid, centroid)
            float[] centroid = centroids[assignments[i]];
            double dotVecCent = 0;
            double dotCentCent = 0;
            for (int d = 0; d < originalDim; d++) {
                dotVecCent += (double) vectors[i][d] * centroid[d];
                dotCentCent += (double) centroid[d] * centroid[d];
            }
            offsets[i] = (float) (dotVecCent - dotCentCent);
        }

        return new AsymmetricHashingResult(w, xEnc, scales, offsets, nClusters);
    }

    private float[][] learnedTraining(float[][] xTraining, int originalDim, int nDims) {
        // PCA initialization: extract top nDims right singular vectors via power iteration
        // This is much faster than full SVD when nDims << originalDim
        float[][] topVectors = SvdUtil.topKRightSingularVectors(xTraining, xTraining.length, originalDim, nDims, seed);
        // P = top nDims right singular vectors transposed: rows of topVectors are the vectors
        // P shape: (originalDim × nDims) where each column is a right singular vector
        float[][] p = new float[originalDim][nDims];
        for (int i = 0; i < originalDim; i++) {
            for (int j = 0; j < nDims; j++) {
                p[i][j] = topVectors[j][i];
            }
        }

        // Project training data: X_ld = xTraining @ P (nTraining × nDims)
        int nTraining = xTraining.length;
        float[][] xLd = new float[nTraining][nDims];
        for (int i = 0; i < nTraining; i++) {
            for (int j = 0; j < nDims; j++) {
                double sum = 0;
                for (int d = 0; d < originalDim; d++) {
                    sum += (double) xTraining[i][d] * p[d][j];
                }
                xLd[i][j] = (float) sum;
            }
        }

        // Initialize random M (nDims × nDims)
        Random rng = new Random(seed);
        float[][] m = new float[nDims][nDims];
        for (int i = 0; i < nDims; i++) {
            for (int j = 0; j < nDims; j++) {
                m[i][j] = (float) rng.nextGaussian();
            }
        }

        // Iterative Procrustes
        float[][] r = null;
        for (int epoch = 0; epoch <= nTrainingIterations; epoch++) {
            // R = procrustes(M)
            r = SvdUtil.procrustes(m, nDims);

            if (epoch < nTrainingIterations) {
                // X_transformed = X_ld @ R
                float[][] xTransformed = matMul(xLd, r, nTraining, nDims, nDims);
                // Quantize
                AshDimQuantizer.QuantizeResult qr = quantizer.encode(xTransformed);
                float[][] xEnc = qr.centeredCodes();
                float[] codeNorms = qr.codeNorms();
                // Normalize encoded: xEnc[i] /= codeNorms[i]
                for (int i = 0; i < nTraining; i++) {
                    if (codeNorms[i] > 0) {
                        float inv = 1.0f / codeNorms[i];
                        for (int j = 0; j < nDims; j++) {
                            xEnc[i][j] *= inv;
                        }
                    }
                }
                // M = X_ld.T @ X_enc (nDims × nDims)
                m = matMulTransposeA(xLd, xEnc, nTraining, nDims, nDims);
            }
        }

        // W = P @ R (originalDim × nDims)
        return matMul(p, r, originalDim, nDims, nDims);
    }

    private float[][] randomOrthogonal(int originalDim, int nDims) {
        Random rng = new Random(seed);
        // Generate random matrix and orthogonalize columns via modified Gram-Schmidt
        float[][] q = new float[originalDim][nDims];
        for (int i = 0; i < originalDim; i++) {
            for (int j = 0; j < nDims; j++) {
                q[i][j] = (float) rng.nextGaussian();
            }
        }
        // Modified Gram-Schmidt: orthogonalize column by column
        for (int j = 0; j < nDims; j++) {
            // Subtract projections of previous columns
            for (int prev = 0; prev < j; prev++) {
                double dot = 0;
                for (int i = 0; i < originalDim; i++) {
                    dot += (double) q[i][j] * q[i][prev];
                }
                for (int i = 0; i < originalDim; i++) {
                    q[i][j] -= (float) dot * q[i][prev];
                }
            }
            // Normalize
            double normSq = 0;
            for (int i = 0; i < originalDim; i++) {
                normSq += (double) q[i][j] * q[i][j];
            }
            float invNorm = (float) (1.0 / Math.sqrt(normSq));
            for (int i = 0; i < originalDim; i++) {
                q[i][j] *= invNorm;
            }
        }
        return q;
    }

    private float[][] sampleRows(float[][] data, int sampleSize) {
        if (sampleSize >= data.length) {
            return data;
        }
        Random rng = new Random(seed);
        // Fisher-Yates partial shuffle
        int[] indices = new int[data.length];
        for (int i = 0; i < data.length; i++) {
            indices[i] = i;
        }
        for (int i = 0; i < sampleSize; i++) {
            int j = i + rng.nextInt(data.length - i);
            int tmp = indices[i];
            indices[i] = indices[j];
            indices[j] = tmp;
        }
        float[][] sample = new float[sampleSize][];
        for (int i = 0; i < sampleSize; i++) {
            sample[i] = data[indices[i]];
        }
        return sample;
    }

    /** C = A @ B where A is (m × k), B is (k × n) */
    private static float[][] matMul(float[][] a, float[][] b, int m, int k, int n) {
        float[][] c = new float[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0;
                for (int l = 0; l < k; l++) {
                    sum += (double) a[i][l] * b[l][j];
                }
                c[i][j] = (float) sum;
            }
        }
        return c;
    }

    /** C = A.T @ B where A is (m × k), B is (m × n), result is (k × n) */
    private static float[][] matMulTransposeA(float[][] a, float[][] b, int m, int k, int n) {
        float[][] c = new float[k][n];
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0;
                for (int l = 0; l < m; l++) {
                    sum += (double) a[l][i] * b[l][j];
                }
                c[i][j] = (float) sum;
            }
        }
        return c;
    }

    /**
     * Truncates a float to float16 precision (round to nearest, ties to even).
     */
    private static float toFloat16(float value) {
        return Float.float16ToFloat(Float.floatToFloat16(value));
    }

    /**
     * Result of running ASH-specific k-means clustering.
     */
    public record AshKMeansResult(float[][] centroids, int[] assignments) {}

    /**
     * Runs a simple k-means clustering on the given vectors to produce ASH centering clusters.
     * This is independent of the IVF clustering used for posting list layout.
     *
     * @param vectors all vectors, shape (nVectors, dim)
     * @param nClusters number of ASH centering clusters (typically 16)
     * @param maxIterations maximum k-means iterations
     * @param seed random seed for initialization
     * @return centroids and per-vector assignments
     */
    public static AshKMeansResult runAshKMeans(float[][] vectors, int nClusters, int maxIterations, long seed) {
        int nVectors = vectors.length;
        int dim = vectors[0].length;
        Random rng = new Random(seed);

        // k-means++ initialization
        float[][] centroids = new float[nClusters][dim];
        int firstIdx = rng.nextInt(nVectors);
        System.arraycopy(vectors[firstIdx], 0, centroids[0], 0, dim);

        double[] minDist = new double[nVectors];
        for (int c = 1; c < nClusters; c++) {
            // Compute distances to nearest existing centroid
            double totalDist = 0;
            for (int i = 0; i < nVectors; i++) {
                double dist = squaredDistance(vectors[i], centroids[c - 1], dim);
                if (c == 1 || dist < minDist[i]) {
                    minDist[i] = dist;
                }
                totalDist += minDist[i];
            }
            // Sample proportional to distance
            double threshold = rng.nextDouble() * totalDist;
            double cumulative = 0;
            int chosen = nVectors - 1;
            for (int i = 0; i < nVectors; i++) {
                cumulative += minDist[i];
                if (cumulative >= threshold) {
                    chosen = i;
                    break;
                }
            }
            System.arraycopy(vectors[chosen], 0, centroids[c], 0, dim);
        }

        // Lloyd's iterations
        int[] assignments = new int[nVectors];
        for (int iter = 0; iter < maxIterations; iter++) {
            // Assign
            boolean changed = false;
            for (int i = 0; i < nVectors; i++) {
                int bestC = 0;
                double bestDist = squaredDistance(vectors[i], centroids[0], dim);
                for (int c = 1; c < nClusters; c++) {
                    double dist = squaredDistance(vectors[i], centroids[c], dim);
                    if (dist < bestDist) {
                        bestDist = dist;
                        bestC = c;
                    }
                }
                if (assignments[i] != bestC) {
                    assignments[i] = bestC;
                    changed = true;
                }
            }
            if (!changed) break;

            // Update centroids
            int[] counts = new int[nClusters];
            for (int c = 0; c < nClusters; c++) {
                java.util.Arrays.fill(centroids[c], 0f);
            }
            for (int i = 0; i < nVectors; i++) {
                int c = assignments[i];
                counts[c]++;
                for (int d = 0; d < dim; d++) {
                    centroids[c][d] += vectors[i][d];
                }
            }
            for (int c = 0; c < nClusters; c++) {
                if (counts[c] > 0) {
                    float inv = 1.0f / counts[c];
                    for (int d = 0; d < dim; d++) {
                        centroids[c][d] *= inv;
                    }
                }
            }
        }
        return new AshKMeansResult(centroids, assignments);
    }

    private static double squaredDistance(float[] a, float[] b, int dim) {
        double sum = 0;
        for (int d = 0; d < dim; d++) {
            double diff = a[d] - b[d];
            sum += diff * diff;
        }
        return sum;
    }
}
