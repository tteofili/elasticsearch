/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.elasticsearch.index.codec.vectors.OptimizedScalarQuantizer;
import org.elasticsearch.test.ESTestCase;

import java.io.IOException;
import java.util.Random;

/**
 * EXPERIMENT (not a regression test): compares the angular error of OSQ with the codebook constant of the paper's
 * distortion-rate law, {@code E[eps] = 2 (1 - rho) + rho * c_b(d)}. With {@code d = D} we have {@code rho = 1}, so the
 * law predicts {@code E[eps] = c_b(D)} on uniformly random directions, which is what a Haar rotation produces.
 * <p>
 * {@code eps_dir = 2 - 2 cos(r, r_hat)} is the paper's shape distortion, with the stored-norm gain applied (the
 * reconstruction is rescaled to the residual's norm). {@code eps_asis = |r - r_hat|^2 / |r|^2} is the error of OSQ's
 * reconstruction as scored, i.e. without that rescaling. ASH is the paper's cosine-optimal grid codebook; its measured
 * constant is the reference.
 */
public class OsqVsDistortionLawExperimentTests extends ESTestCase {

    private static final int N = 1000;

    private record Eps(double dir, double asis) {}

    public void testIsotropicConstants() throws IOException {
        for (int dim : new int[] { 64, 128, 768 }) {
            float[][] vecs = gaussians(new Random(1), N, dim, null);
            float[] zero = new float[dim];
            for (int bits : new int[] { 1, 2, 4 }) {
                Eps osq = osq(dim, bits, vecs, zero);
                double ash = ash(dim, bits, vecs, zero);
                logger.info(
                    "EXP isotropic d={} b={} c_ash(law)={} osq_dir={} osq_asis={} ratio_osq/ash={}",
                    dim,
                    bits,
                    fmt(ash),
                    fmt(osq.dir),
                    fmt(osq.asis),
                    fmt(osq.dir / ash)
                );
            }
        }
    }

    public void testAnisotropicRotation() throws IOException {
        for (int dim : new int[] { 128, 768 }) {
            // axis aligned, power-law spectrum, a few outlier coordinates: typical of un-rotated embeddings
            double[] std = new double[dim];
            for (int j = 0; j < dim; j++) {
                std[j] = 4.0 / Math.sqrt(j + 1) * (j % 37 == 0 ? 6 : 1);
            }
            float[][] raw = gaussians(new Random(2), N, dim, std);
            float[][] haar = rotate(raw, randomOrthogonal(new Random(3), dim));
            Preconditioner pre = Preconditioner.createPreconditioner(dim, 32);
            float[][] block = new float[N][dim];
            for (int i = 0; i < N; i++) {
                pre.applyTransform(raw[i], block[i]);
            }
            for (int bits : new int[] { 1, 2, 4 }) {
                float[][] iso = gaussians(new Random(1), N, dim, null);
                double law = ash(dim, bits, iso, new float[dim]);
                logger.info(
                    "EXP aniso d={} b={} law_c={} osq_raw={} osq_haar={} osq_block32={} ash_randomW_raw={}",
                    dim,
                    bits,
                    fmt(law),
                    fmt(osq(dim, bits, raw, mean(raw)).dir),
                    fmt(osq(dim, bits, haar, mean(haar)).dir),
                    fmt(osq(dim, bits, block, mean(block)).dir),
                    fmt(ash(dim, bits, raw, mean(raw)))
                );
            }
        }
    }

    private static String fmt(double v) {
        return String.format(java.util.Locale.ROOT, "%.4f", v);
    }

    private static Eps osq(int dim, int bits, float[][] vecs, float[] centroid) {
        OptimizedScalarQuantizer q = new OptimizedScalarQuantizer(VectorSimilarityFunction.EUCLIDEAN);
        float[] scratch = new float[dim];
        int[] codes = new int[dim];
        double dir = 0, asis = 0;
        float levels = (1 << bits) - 1;
        for (float[] v : vecs) {
            var res = q.scalarQuantize(v.clone(), scratch, codes, (byte) bits, centroid);
            float step = (res.upperInterval() - res.lowerInterval()) / levels;
            double rr = 0, rh = 0, rd = 0, err = 0;
            for (int k = 0; k < dim; k++) {
                double r = v[k] - centroid[k];
                double h = res.lowerInterval() + step * codes[k];
                rr += r * r;
                rh += h * h;
                rd += r * h;
                err += (r - h) * (r - h);
            }
            dir += 2 - 2 * rd / Math.sqrt(rr * rh);
            asis += err / rr;
        }
        return new Eps(dir / vecs.length, asis / vecs.length);
    }

    /** Mean angular error of the ASH grid codebook behind a random rotation, with d = D. */
    private static double ash(int dim, int bits, float[][] vecs, float[] centroid) throws IOException {
        var quantizer = new org.elasticsearch.index.codec.vectors.ash.AsymmetricHashingQuantizerExperimentAccess(dim, bits);
        return quantizer.meanAngularError(vecs, centroid);
    }

    private static float[] mean(float[][] vecs) {
        float[] m = new float[vecs[0].length];
        for (float[] v : vecs) {
            for (int k = 0; k < m.length; k++) {
                m[k] += v[k] / vecs.length;
            }
        }
        return m;
    }

    private static float[][] gaussians(Random rnd, int n, int dim, double[] std) {
        float[][] out = new float[n][dim];
        for (float[] v : out) {
            double norm = 0;
            for (int k = 0; k < dim; k++) {
                v[k] = (float) (rnd.nextGaussian() * (std == null ? 1.0 : std[k]));
                norm += v[k] * v[k];
            }
            if (std == null) { // uniform directions
                float inv = (float) (1 / Math.sqrt(norm));
                for (int k = 0; k < dim; k++) {
                    v[k] *= inv;
                }
            }
        }
        return out;
    }

    private static float[][] randomOrthogonal(Random rnd, int dim) {
        double[][] m = new double[dim][dim];
        for (double[] row : m) {
            for (int k = 0; k < dim; k++) {
                row[k] = rnd.nextGaussian();
            }
        }
        for (int i = 0; i < dim; i++) { // Gram-Schmidt
            for (int j = 0; j < i; j++) {
                double dot = 0;
                for (int k = 0; k < dim; k++) {
                    dot += m[i][k] * m[j][k];
                }
                for (int k = 0; k < dim; k++) {
                    m[i][k] -= dot * m[j][k];
                }
            }
            double norm = 0;
            for (int k = 0; k < dim; k++) {
                norm += m[i][k] * m[i][k];
            }
            norm = Math.sqrt(norm);
            for (int k = 0; k < dim; k++) {
                m[i][k] /= norm;
            }
        }
        float[][] out = new float[dim][dim];
        for (int i = 0; i < dim; i++) {
            for (int k = 0; k < dim; k++) {
                out[i][k] = (float) m[i][k];
            }
        }
        return out;
    }

    private static float[][] rotate(float[][] vecs, float[][] q) {
        int dim = q.length;
        float[][] out = new float[vecs.length][dim];
        for (int n = 0; n < vecs.length; n++) {
            for (int i = 0; i < dim; i++) {
                double s = 0;
                for (int k = 0; k < dim; k++) {
                    s += q[i][k] * vecs[n][k];
                }
                out[n][i] = (float) s;
            }
        }
        return out;
    }
}
