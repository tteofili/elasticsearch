/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.benchmark.vector.scorer;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.elasticsearch.benchmark.internal.BenchmarkLogging;
import org.elasticsearch.benchmark.store.DirectoryType;
import org.elasticsearch.benchmark.vector.VectorizationInfo;
import org.elasticsearch.benchmark.vector.store.DirectoryFactory;
import org.elasticsearch.core.IOUtils;
import org.elasticsearch.core.SuppressForbidden;
import org.elasticsearch.index.codec.vectors.OptimizedScalarQuantizer;
import org.elasticsearch.index.codec.vectors.diskbbq.E8ResidualQuantizer;
import org.elasticsearch.index.codec.vectors.diskbbq.QuantEncoding;
import org.elasticsearch.simdvec.E8Lattice;
import org.elasticsearch.simdvec.E8VectorsScorer;
import org.elasticsearch.simdvec.ES940OSQVectorsScorer;
import org.elasticsearch.simdvec.ESVectorizationProvider;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.TearDown;
import org.openjdk.jmh.annotations.Warmup;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;
import java.util.concurrent.TimeUnit;

/**
 * Scan cost of the experimental E8 block code against the default DiskBBQ code (OSQ, 1-bit documents and 4-bit
 * queries) at the same storage (one byte per 8 dimensions), on the same kind of data and through the same kind of
 * input: {@code scoreBulk} over bulks of {@link ES940OSQVectorsScorer#BULK_SIZE} documents, integer kernel plus score
 * corrections, as a posting-list visit does it.
 * <p>
 * Result: average time per call, which scores {@code numVectors} documents; divide by {@code numVectors} for ns/doc.
 * Compare rows with the same {@code dims}, {@code implementation} and {@code numVectors}. {@code numVectors = 320} keeps
 * everything in L1/L2 (kernel cost); {@code 32768} streams the codes from memory (25–50 MB per run at these dims).
 * <pre>
 * VEC_NATIVE_BUILD=host ./gradlew :libs:simdvec:benchmark --args 'VectorScorerE8Benchmark'
 * </pre>
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(Scope.Benchmark)
@Warmup(iterations = 4, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(value = 1, jvmArgsPrepend = { "--add-modules=jdk.incubator.vector" })
public class VectorScorerE8Benchmark {

    static {
        BenchmarkLogging.configure();
        VectorizationInfo.printOnce();
    }

    public enum VectorImplementation {
        SCALAR,
        PANAMA,
        NATIVE
    }

    public enum Code {
        /** DiskBBQ default: OSQ, 1-bit documents, 4-bit striped query (ES940 scorer, D1Q4). */
        OSQ_D1Q4,
        /** E8 roots, one byte per 8 dimensions, int8 query table (E8VectorsScorer). */
        E8
    }

    @Param({ "384", "768", "1024", "1536" })
    public int dims;

    @Param
    public Code code;

    @Param
    public VectorImplementation implementation;

    @Param({ "320", "32768" })
    public int numVectors;

    @Param({ "MMAP" })
    public DirectoryType directoryType;

    static final int BULK_SIZE = ES940OSQVectorsScorer.BULK_SIZE;
    static final int NUM_QUERIES = 8;
    static final VectorSimilarityFunction SIM = VectorSimilarityFunction.DOT_PRODUCT;

    Path tempDir;
    Directory directory;
    IndexInput input;
    float[] scores = new float[BULK_SIZE];

    // OSQ
    ES940OSQVectorsScorer osq;
    byte[][] osqQueries;
    OptimizedScalarQuantizer.QuantizationResult[] osqQueryCorrections;
    // E8
    E8VectorsScorer e8;
    byte[][] e8Tables;
    float[] e8Scales;

    @Setup
    public void setup() throws IOException {
        Random random = new Random(42);
        tempDir = createTempDirectory("VectorScorerE8Benchmark");
        directory = DirectoryFactory.newDirectory(directoryType, tempDir);
        float[] centroid = new float[dims];
        for (int i = 0; i < dims; i++) {
            centroid[i] = random.nextFloat() * 0.1f;
        }
        float[][] vectors = new float[numVectors][dims];
        for (float[] v : vectors) {
            for (int i = 0; i < dims; i++) {
                v[i] = centroid[i] + (float) random.nextGaussian() * 0.05f;
            }
        }
        float[][] queries = new float[NUM_QUERIES][dims];
        for (float[] q : queries) {
            for (int i = 0; i < dims; i++) {
                q[i] = centroid[i] + (float) random.nextGaussian() * 0.05f;
            }
        }
        QuantEncoding encoding = code == Code.E8 ? QuantEncoding.E8_ROOTS : QuantEncoding.ONE_BIT_4BIT_QUERY;
        int docBytes = encoding.getDocPackedLength(dims);
        int[] quantized = new int[encoding.discretizedDimensions(dims)];
        float[] residual = new float[dims];
        byte[][] packed = new byte[numVectors][docBytes];
        OptimizedScalarQuantizer.QuantizationResult[] corrections = new OptimizedScalarQuantizer.QuantizationResult[numVectors];
        OptimizedScalarQuantizer osqQuantizer = new OptimizedScalarQuantizer(SIM);
        E8ResidualQuantizer e8Quantizer = new E8ResidualQuantizer(SIM);
        for (int i = 0; i < numVectors; i++) {
            corrections[i] = E8ResidualQuantizer.quantize(encoding, osqQuantizer, e8Quantizer, vectors[i], residual, quantized, centroid);
            encoding.pack(quantized, packed[i]);
        }
        try (IndexOutput out = directory.createOutput("vectors", IOContext.DEFAULT)) {
            for (int start = 0; start < numVectors; start += BULK_SIZE) {
                if (code == Code.E8) {
                    for (int b = 0; b < docBytes; b++) {
                        for (int j = 0; j < BULK_SIZE; j++) {
                            out.writeByte(packed[start + j][b]);
                        }
                    }
                } else {
                    for (int j = 0; j < BULK_SIZE; j++) {
                        out.writeBytes(packed[start + j], docBytes);
                    }
                }
                for (int j = 0; j < BULK_SIZE; j++) {
                    out.writeInt(Float.floatToIntBits(corrections[start + j].lowerInterval()));
                }
                for (int j = 0; j < BULK_SIZE; j++) {
                    out.writeInt(Float.floatToIntBits(corrections[start + j].upperInterval()));
                }
                for (int j = 0; j < BULK_SIZE; j++) {
                    out.writeInt(corrections[start + j].quantizedComponentSum());
                }
                for (int j = 0; j < BULK_SIZE; j++) {
                    out.writeInt(Float.floatToIntBits(corrections[start + j].additionalCorrection()));
                }
            }
        }
        input = directory.openInput("vectors", IOContext.DEFAULT);
        if (code == Code.E8) {
            E8VectorsScorer.Kernel kernel = switch (implementation) {
                case SCALAR -> E8VectorsScorer.Kernel.SCALAR;
                case PANAMA -> E8VectorsScorer.Kernel.PANAMA;
                case NATIVE -> E8VectorsScorer.Kernel.NATIVE;
            };
            e8 = E8VectorsScorer.create(input, docBytes, BULK_SIZE, kernel);
            if (e8.kernel() != kernel) {
                throw new IllegalStateException("E8 kernel " + kernel + " is not available here; got " + e8.kernel());
            }
            e8Tables = new byte[NUM_QUERIES][encoding.getQueryPackedLength(dims)];
            e8Scales = new float[NUM_QUERIES];
            float[] r = new float[dims];
            for (int q = 0; q < NUM_QUERIES; q++) {
                for (int i = 0; i < dims; i++) {
                    r[i] = queries[q][i] - centroid[i];
                }
                e8Scales[q] = E8Lattice.buildTable(r, dims, e8Tables[q]);
            }
        } else {
            var provider = switch (implementation) {
                case SCALAR -> ESVectorizationProvider.lookup(false, false);
                case PANAMA -> ESVectorizationProvider.lookup(true, false);
                case NATIVE -> ESVectorizationProvider.lookup(true, true);
            };
            osq = provider.getVectorScorerFactory()
                .newES940OSQVectorsScorer(
                    input,
                    encoding.queryBits(),
                    encoding.bits(),
                    dims,
                    docBytes,
                    BULK_SIZE,
                    ES940OSQVectorsScorer.BitEncoding.STRIPED
                );
            OptimizedScalarQuantizer queryQuantizer = new OptimizedScalarQuantizer(SIM, OptimizedScalarQuantizer.DEFAULT_LAMBDA, 1);
            osqQueries = new byte[NUM_QUERIES][encoding.getQueryPackedLength(dims)];
            osqQueryCorrections = new OptimizedScalarQuantizer.QuantizationResult[NUM_QUERIES];
            int[] qq = new int[encoding.discretizedDimensions(dims)];
            for (int q = 0; q < NUM_QUERIES; q++) {
                osqQueryCorrections[q] = queryQuantizer.scalarQuantize(queries[q], residual, qq, encoding.queryBits(), centroid);
                encoding.packQuery(qq, osqQueries[q]);
            }
        }
    }

    @SuppressForbidden(reason = "scratch directory for the Lucene Directory")
    static Path createTempDirectory(String name) throws IOException {
        return Files.createTempDirectory(name);
    }

    @TearDown
    public void teardown() throws IOException {
        IOUtils.close(input, directory);
        IOUtils.rm(tempDir);
    }

    private int query = 0;

    /** Scores every document once for one query (rotating over {@link #NUM_QUERIES} queries); returns the max score. */
    @Benchmark
    public float scoreBulk() throws IOException {
        query = (query + 1) % NUM_QUERIES;
        input.seek(0);
        float max = Float.NEGATIVE_INFINITY;
        if (code == Code.E8) {
            for (int i = 0; i < numVectors; i += BULK_SIZE) {
                max = Math.max(max, e8.scoreBulk(e8Tables[query], e8Scales[query], 0.1f, SIM, -1, scores, BULK_SIZE));
            }
        } else {
            OptimizedScalarQuantizer.QuantizationResult c = osqQueryCorrections[query];
            for (int i = 0; i < numVectors; i += BULK_SIZE) {
                max = Math.max(
                    max,
                    osq.scoreBulk(osqQueries[query], c.lowerInterval(), c.upperInterval(), c.quantizedComponentSum(), 0.1f, SIM, 0f, scores)
                );
            }
        }
        return max;
    }
}
