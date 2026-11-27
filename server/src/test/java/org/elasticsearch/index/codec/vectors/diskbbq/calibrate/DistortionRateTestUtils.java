/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq.calibrate;

import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FloatVectorValues;
import org.elasticsearch.index.codec.vectors.diskbbq.CentroidIndex;
import org.elasticsearch.index.codec.vectors.diskbbq.CentroidSupplier;
import org.elasticsearch.simdvec.ESVectorUtil;

import java.util.Arrays;
import java.util.Random;

/** Synthetic data and rotations for testing the distortion-rate classes. */
public final class DistortionRateTestUtils {

    private DistortionRateTestUtils() {}

    /** A clustered corpus with the generating centers as landmarks, and queries close to documents. */
    public record Dataset(float[][] docs, float[][] centroids, int[] assignment, float[][] queries) {
        /** The corpus as the read-only view the production classes take. */
        public ClusteredSample sample() {
            return sampleOf(docs, centroids, assignment);
        }

        public FloatVectorValues queryValues() {
            return values(queries);
        }

        public int[] queryOrdinals() {
            return ordinals(queries.length);
        }
    }

    /** Test data is generated as arrays; the production classes are given views of them, as they are given off-heap vectors. */
    public static FloatVectorValues values(float[][] vectors) {
        return FloatVectorValues.fromFloats(Arrays.asList(vectors), vectors[0].length);
    }

    public static int[] ordinals(int n) {
        int[] ordinals = new int[n];
        Arrays.setAll(ordinals, i -> i);
        return ordinals;
    }

    public static ClusteredSample sampleOf(float[][] docs, float[][] centroids, int[] assignment) {
        int dim = docs[0].length;
        return new ClusteredSample(
            values(docs),
            ordinals(docs.length),
            CentroidSupplier.fromArray(centroids, CentroidIndex.NO_INDEX, dim),
            assignment
        );
    }

    /**
     * The dataset with every value scaled into the signed byte range and rounded, so that its floats are exactly what the
     * same vectors hold as bytes.
     */
    public static Dataset roundedToBytes(Dataset data) {
        float max = 0;
        for (float[][] set : new float[][][] { data.docs(), data.centroids(), data.queries() }) {
            for (float[] vector : set) {
                for (float v : vector) {
                    max = Math.max(max, Math.abs(v));
                }
            }
        }
        float scale = 100f / max;
        return new Dataset(
            roundedToBytes(data.docs(), scale),
            roundedToBytes(data.centroids(), scale),
            data.assignment(),
            roundedToBytes(data.queries(), scale)
        );
    }

    private static float[][] roundedToBytes(float[][] vectors, float scale) {
        float[][] rounded = new float[vectors.length][];
        for (int i = 0; i < vectors.length; i++) {
            rounded[i] = new float[vectors[i].length];
            for (int d = 0; d < rounded[i].length; d++) {
                rounded[i][d] = Math.round(vectors[i][d] * scale);
            }
        }
        return rounded;
    }

    private static byte[][] toBytes(float[][] byteValued) {
        byte[][] bytes = new byte[byteValued.length][];
        for (int i = 0; i < bytes.length; i++) {
            bytes[i] = new byte[byteValued[i].length];
            for (int d = 0; d < bytes[i].length; d++) {
                bytes[i][d] = (byte) byteValued[i][d];
            }
        }
        return bytes;
    }

    /** Byte-backed vectors, read as the floats they hold, of a dataset made of {@link #roundedToBytes byte values}. */
    public static FloatVectorValues byteBackedValues(float[][] byteValued) {
        return new WidenedByteVectorValues(ByteVectorValues.fromBytes(Arrays.asList(toBytes(byteValued)), byteValued[0].length));
    }

    /** A sample whose vectors and centroids are held as bytes, of a dataset made of {@link #roundedToBytes byte values}. */
    public static ClusteredSample byteBackedSampleOf(Dataset byteValued) {
        int dim = byteValued.docs()[0].length;
        return ClusteredSample.ofBytes(
            ByteVectorValues.fromBytes(Arrays.asList(toBytes(byteValued.docs())), dim),
            ordinals(byteValued.docs().length),
            CentroidSupplier.fromByteArray(toBytes(byteValued.centroids()), CentroidIndex.NO_INDEX, dim),
            byteValued.assignment()
        );
    }

    public static DistortionRateQuantizer.Residuals residualsOf(float[][] residuals) {
        return new DistortionRateQuantizer.Residuals() {
            @Override
            public int size() {
                return residuals.length;
            }

            @Override
            public float[] residual(int i) {
                return residuals[i];
            }
        };
    }

    public static Dataset clusteredDataset(int docsPerCluster, int clusters, int dim, int queries, long seed) {
        return clusteredDataset(docsPerCluster, clusters, dim, queries, seed, 0.6f);
    }

    /** {@code noise} is the spread of the documents around their landmark: the larger, the more clusters overlap. */
    public static Dataset clusteredDataset(int docsPerCluster, int clusters, int dim, int queries, long seed, float noise) {
        Random random = new Random(seed);
        float[][] centroids = new float[clusters][dim];
        for (float[] c : centroids) {
            for (int d = 0; d < dim; d++) {
                c[d] = (float) random.nextGaussian();
            }
        }
        float[][] docs = new float[docsPerCluster * clusters][dim];
        int[] assignment = new int[docs.length];
        for (int i = 0; i < docs.length; i++) {
            assignment[i] = i / docsPerCluster;
            for (int d = 0; d < dim; d++) {
                docs[i][d] = centroids[assignment[i]][d] + noise * (float) random.nextGaussian();
            }
        }
        float[][] qs = new float[queries][dim];
        for (int q = 0; q < queries; q++) {
            float[] base = docs[random.nextInt(docs.length)];
            for (int d = 0; d < dim; d++) {
                qs[q][d] = base[d] + 0.5f * noise * (float) random.nextGaussian();
            }
        }
        return new Dataset(docs, centroids, assignment, qs);
    }

    /** Residuals whose energy is concentrated in the first coordinates, with a smoothly decaying spectrum. */
    public static float[][] anisotropicResiduals(int n, int dim, long seed) {
        Random random = new Random(seed);
        double[] std = new double[dim];
        for (int j = 0; j < dim; j++) {
            std[j] = 1.0 / (1.0 + j / 24.0);
        }
        float[][] out = new float[n][dim];
        for (float[] v : out) {
            for (int j = 0; j < dim; j++) {
                v[j] = (float) (random.nextGaussian() * std[j]);
            }
        }
        return out;
    }

    /** A dense random rotation. */
    public static OsqDistortionRateQuantizer.Rotation haar(int dim, long seed) {
        Random random = new Random(seed);
        double[][] m = new double[dim][dim];
        for (double[] row : m) {
            for (int j = 0; j < dim; j++) {
                row[j] = random.nextGaussian();
            }
        }
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < i; j++) {
                double dot = 0;
                for (int d = 0; d < dim; d++) {
                    dot += m[i][d] * m[j][d];
                }
                for (int d = 0; d < dim; d++) {
                    m[i][d] -= dot * m[j][d];
                }
            }
            double norm = 0;
            for (int d = 0; d < dim; d++) {
                norm += m[i][d] * m[i][d];
            }
            norm = Math.sqrt(norm);
            for (int d = 0; d < dim; d++) {
                m[i][d] /= norm;
            }
        }
        float[][] r = new float[dim][dim];
        for (int i = 0; i < dim; i++) {
            for (int d = 0; d < dim; d++) {
                r[i][d] = (float) m[i][d];
            }
        }
        return (in, out) -> {
            for (int i = 0; i < dim; i++) {
                out[i] = ESVectorUtil.dotProduct(r[i], in);
            }
        };
    }

    public static OsqDistortionRateQuantizer osq(
        int dim,
        int bits,
        float lambda,
        RotationKind kind,
        OsqDistortionRateQuantizer.Rotation rotation
    ) {
        return new OsqDistortionRateQuantizer(dim, bits, 4, lambda, kind, rotation);
    }

    public static OsqDistortionRateQuantizer.Rotation identity() {
        return (in, out) -> System.arraycopy(in, 0, out, 0, in.length);
    }

    /** The residuals of a corpus against the landmarks. */
    public static float[][] residuals(Dataset data, int limit) {
        int n = Math.min(limit, data.docs().length);
        int step = Math.max(1, data.docs().length / n);
        float[][] out = new float[n][];
        for (int j = 0; j < n; j++) {
            int i = Math.min(data.docs().length - 1, j * step);
            float[] r = data.docs()[i].clone();
            float[] c = data.centroids()[data.assignment()[i]];
            for (int d = 0; d < r.length; d++) {
                r[d] -= c[d];
            }
            out[j] = r;
        }
        return out;
    }

    /** The share of the {@code k} nearest neighbors found among the top {@code eta * k} compressed scores. */
    public static double recall(float[][] queries, float[][] docs, float[][] compressedScoresByQuery, int k, int eta) {
        long hits = 0;
        for (int q = 0; q < queries.length; q++) {
            int[] truth = top(exactScores(queries[q], docs), k);
            int[] kept = top(compressedScoresByQuery[q], eta * k);
            for (int t : truth) {
                for (int c : kept) {
                    if (t == c) {
                        hits++;
                    }
                }
            }
        }
        return hits / (double) ((long) queries.length * k);
    }

    private static float[] exactScores(float[] query, float[][] docs) {
        float[] s = new float[docs.length];
        for (int i = 0; i < docs.length; i++) {
            s[i] = ESVectorUtil.dotProduct(query, docs[i]);
        }
        return s;
    }

    private static int[] top(float[] scores, int n) {
        Integer[] order = new Integer[scores.length];
        for (int i = 0; i < order.length; i++) {
            order[i] = i;
        }
        Arrays.sort(order, (x, y) -> Float.compare(scores[y], scores[x]));
        int[] out = new int[n];
        for (int i = 0; i < n; i++) {
            out[i] = order[i];
        }
        return out;
    }
}
