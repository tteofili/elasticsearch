/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq.next.ash;

/**
 * Utility class providing thin SVD decomposition via one-sided Jacobi rotations.
 * <p>
 * This implementation is designed for small matrices (e.g. 28×28 or 768×28) used
 * during ASH training. It computes the thin SVD: A = U * S * Vt, where U has
 * orthonormal columns, S is diagonal, and Vt has orthonormal rows.
 */
public final class SvdUtil {

    private SvdUtil() {}

    /**
     * Result of a thin SVD decomposition.
     *
     * @param u  left singular vectors (m × k)
     * @param s  singular values (k)
     * @param vt right singular vectors transposed (k × n)
     */
    public record SvdResult(float[][] u, float[] s, float[][] vt) {}

    /**
     * Computes the thin SVD of matrix A (m × n) where m >= n.
     * Returns U (m × n), S (n), Vt (n × n).
     * <p>
     * Uses one-sided Jacobi SVD which is simple and numerically stable for small matrices.
     */
    public static SvdResult thinSvd(float[][] a, int m, int n) {
        if (m < n) {
            // For wide matrices, compute SVD of transpose and swap U/V
            float[][] at = transpose(a, m, n);
            SvdResult result = thinSvd(at, n, m);
            // A = U * S * Vt => At = V * S * Ut
            return new SvdResult(result.vt() != null ? transposeSquare(result.vt(), m) : null, result.s(), transposeToVt(result.u(), n, m));
        }

        // Copy A into working matrix (m × n)
        float[][] work = new float[m][n];
        for (int i = 0; i < m; i++) {
            System.arraycopy(a[i], 0, work[i], 0, n);
        }

        // V starts as identity (n × n)
        float[][] v = new float[n][n];
        for (int i = 0; i < n; i++) {
            v[i][i] = 1.0f;
        }

        // One-sided Jacobi: apply rotations to columns of work until convergence
        int maxIterations = 100 * n;
        for (int iter = 0; iter < maxIterations; iter++) {
            boolean converged = true;
            for (int p = 0; p < n - 1; p++) {
                for (int q = p + 1; q < n; q++) {
                    // Compute 2x2 Gram matrix elements for columns p, q
                    double app = 0, aqq = 0, apq = 0;
                    for (int i = 0; i < m; i++) {
                        app += (double) work[i][p] * work[i][p];
                        aqq += (double) work[i][q] * work[i][q];
                        apq += (double) work[i][p] * work[i][q];
                    }

                    if (Math.abs(apq) < 1e-10 * Math.sqrt(app * aqq)) {
                        continue; // columns already orthogonal
                    }
                    converged = false;

                    // Compute Jacobi rotation angle
                    double tau = (aqq - app) / (2.0 * apq);
                    double t;
                    if (tau >= 0) {
                        t = 1.0 / (tau + Math.sqrt(1.0 + tau * tau));
                    } else {
                        t = -1.0 / (-tau + Math.sqrt(1.0 + tau * tau));
                    }
                    double cos = 1.0 / Math.sqrt(1.0 + t * t);
                    double sin = t * cos;

                    // Apply rotation to columns of work
                    for (int i = 0; i < m; i++) {
                        double wp = work[i][p];
                        double wq = work[i][q];
                        work[i][p] = (float) (cos * wp - sin * wq);
                        work[i][q] = (float) (sin * wp + cos * wq);
                    }

                    // Apply rotation to columns of V
                    for (int i = 0; i < n; i++) {
                        double vp = v[i][p];
                        double vq = v[i][q];
                        v[i][p] = (float) (cos * vp - sin * vq);
                        v[i][q] = (float) (sin * vp + cos * vq);
                    }
                }
            }
            if (converged) {
                break;
            }
        }

        // Extract singular values and normalize columns of work to get U
        float[] s = new float[n];
        float[][] u = new float[m][n];
        for (int j = 0; j < n; j++) {
            double norm = 0;
            for (int i = 0; i < m; i++) {
                norm += (double) work[i][j] * work[i][j];
            }
            s[j] = (float) Math.sqrt(norm);
            if (s[j] > 1e-10f) {
                float invNorm = 1.0f / s[j];
                for (int i = 0; i < m; i++) {
                    u[i][j] = work[i][j] * invNorm;
                }
            }
        }

        // Sort by descending singular value
        sortDescending(u, s, v, m, n);

        // Vt = transpose of V
        float[][] vt = new float[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                vt[i][j] = v[j][i];
            }
        }

        return new SvdResult(u, s, vt);
    }

    /**
     * Computes U * Vt from the SVD of matrix M (k × k). This is the orthogonal Procrustes solution.
     * Returns the nearest orthogonal matrix to M.
     */
    public static float[][] procrustes(float[][] m, int k) {
        SvdResult svd = thinSvd(m, k, k);
        // R = U * Vt
        float[][] r = new float[k][k];
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < k; j++) {
                double sum = 0;
                for (int l = 0; l < k; l++) {
                    sum += (double) svd.u()[i][l] * svd.vt()[l][j];
                }
                r[i][j] = (float) sum;
            }
        }
        return r;
    }

    private static void sortDescending(float[][] u, float[] s, float[][] v, int m, int n) {
        // Simple insertion sort (n is small)
        for (int i = 0; i < n - 1; i++) {
            int maxIdx = i;
            for (int j = i + 1; j < n; j++) {
                if (s[j] > s[maxIdx]) {
                    maxIdx = j;
                }
            }
            if (maxIdx != i) {
                // Swap singular values
                float tmp = s[i];
                s[i] = s[maxIdx];
                s[maxIdx] = tmp;
                // Swap columns of U
                for (int r = 0; r < m; r++) {
                    tmp = u[r][i];
                    u[r][i] = u[r][maxIdx];
                    u[r][maxIdx] = tmp;
                }
                // Swap columns of V
                for (int r = 0; r < n; r++) {
                    tmp = v[r][i];
                    v[r][i] = v[r][maxIdx];
                    v[r][maxIdx] = tmp;
                }
            }
        }
    }

    private static float[][] transpose(float[][] a, int m, int n) {
        float[][] at = new float[n][m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                at[j][i] = a[i][j];
            }
        }
        return at;
    }

    private static float[][] transposeSquare(float[][] a, int n) {
        float[][] at = new float[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                at[i][j] = a[j][i];
            }
        }
        return at;
    }

    private static float[][] transposeToVt(float[][] u, int k, int m) {
        // u is (m × k), we want (k × m)
        float[][] vt = new float[k][m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < k; j++) {
                vt[j][i] = u[i][j];
            }
        }
        return vt;
    }

    /**
     * Computes the top-k right singular vectors of matrix A (m × n) using power iteration
     * on the Gram matrix A^T A with deflation. Much faster than full SVD when k is small.
     *
     * @param a matrix of shape (m × n)
     * @param m number of rows
     * @param n number of columns
     * @param k number of top singular vectors to extract
     * @param seed random seed for initialization
     * @return top-k right singular vectors as rows (k × n)
     */
    public static float[][] topKRightSingularVectors(float[][] a, int m, int n, int k, long seed) {
        // Compute C = A^T A (n × n) — this is symmetric positive semi-definite
        // For m >> n this is cheaper than full SVD
        // For m < n, we use A A^T (m × m) and transform back
        if (m >= n) {
            return topKEigenvectorsGram(a, m, n, k, seed);
        } else {
            // Compute A A^T (m × m), find eigenvectors, transform back to right singular vectors
            return topKEigenvectorsGramTranspose(a, m, n, k, seed);
        }
    }

    private static float[][] topKEigenvectorsGram(float[][] a, int m, int n, int k, long seed) {
        // Power iteration with deflation on A^T A
        // We don't form A^T A explicitly — instead compute (A^T A) v = A^T (A v) for each iteration
        float[][] result = new float[k][n];
        java.util.Random rng = new java.util.Random(seed);

        // Deflation vectors already found
        float[][] deflated = new float[k][];
        int found = 0;

        for (int vec = 0; vec < k; vec++) {
            // Random initial vector
            float[] v = new float[n];
            for (int i = 0; i < n; i++) {
                v[i] = (float) rng.nextGaussian();
            }
            normalize(v);

            // Power iteration: v <- A^T (A v) / ||...||
            for (int iter = 0; iter < 100; iter++) {
                // w = A v (m-dimensional)
                float[] w = new float[m];
                for (int i = 0; i < m; i++) {
                    double sum = 0;
                    for (int j = 0; j < n; j++) {
                        sum += (double) a[i][j] * v[j];
                    }
                    w[i] = (float) sum;
                }
                // v_new = A^T w (n-dimensional)
                float[] vNew = new float[n];
                for (int j = 0; j < n; j++) {
                    double sum = 0;
                    for (int i = 0; i < m; i++) {
                        sum += (double) a[i][j] * w[i];
                    }
                    vNew[j] = (float) sum;
                }
                // Deflate: remove components along previously found vectors
                for (int d = 0; d < found; d++) {
                    double dot = 0;
                    for (int j = 0; j < n; j++) {
                        dot += (double) vNew[j] * deflated[d][j];
                    }
                    for (int j = 0; j < n; j++) {
                        vNew[j] -= (float) (dot * deflated[d][j]);
                    }
                }
                normalize(vNew);
                v = vNew;
            }
            deflated[found] = v;
            result[vec] = v;
            found++;
        }
        return result;
    }

    private static float[][] topKEigenvectorsGramTranspose(float[][] a, int m, int n, int k, long seed) {
        // A is (m × n) with m < n. Find top-k eigenvectors of A A^T (m × m), then transform.
        // u_i = eigenvector of A A^T => v_i = A^T u_i / sigma_i (right singular vector)
        float[][] result = new float[k][n];
        java.util.Random rng = new java.util.Random(seed);
        float[][] deflated = new float[k][];
        int found = 0;

        for (int vec = 0; vec < k; vec++) {
            // Random initial vector (m-dimensional)
            float[] u = new float[m];
            for (int i = 0; i < m; i++) {
                u[i] = (float) rng.nextGaussian();
            }
            normalize(u);

            // Power iteration on A A^T: u <- A (A^T u) / ||...||
            for (int iter = 0; iter < 100; iter++) {
                // w = A^T u (n-dimensional)
                float[] w = new float[n];
                for (int j = 0; j < n; j++) {
                    double sum = 0;
                    for (int i = 0; i < m; i++) {
                        sum += (double) a[i][j] * u[i];
                    }
                    w[j] = (float) sum;
                }
                // u_new = A w (m-dimensional)
                float[] uNew = new float[m];
                for (int i = 0; i < m; i++) {
                    double sum = 0;
                    for (int j = 0; j < n; j++) {
                        sum += (double) a[i][j] * w[j];
                    }
                    uNew[i] = (float) sum;
                }
                // Deflate
                for (int d = 0; d < found; d++) {
                    double dot = 0;
                    for (int i = 0; i < m; i++) {
                        dot += (double) uNew[i] * deflated[d][i];
                    }
                    for (int i = 0; i < m; i++) {
                        uNew[i] -= (float) (dot * deflated[d][i]);
                    }
                }
                normalize(uNew);
                u = uNew;
            }
            deflated[found] = u;
            found++;

            // Recover right singular vector: v = A^T u, then normalize
            float[] v = new float[n];
            for (int j = 0; j < n; j++) {
                double sum = 0;
                for (int i = 0; i < m; i++) {
                    sum += (double) a[i][j] * u[i];
                }
                v[j] = (float) sum;
            }
            normalize(v);
            result[vec] = v;
        }
        return result;
    }

    private static void normalize(float[] v) {
        double norm = 0;
        for (float f : v) {
            norm += (double) f * f;
        }
        norm = Math.sqrt(norm);
        if (norm > 0) {
            float invNorm = (float) (1.0 / norm);
            for (int i = 0; i < v.length; i++) {
                v[i] *= invNorm;
            }
        }
    }
}
