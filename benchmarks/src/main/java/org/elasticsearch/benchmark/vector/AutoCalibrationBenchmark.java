/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.benchmark.vector;

import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.elasticsearch.benchmark.Utils;
import org.elasticsearch.index.codec.vectors.cluster.KMeansFloatVectorValues;
import org.elasticsearch.index.codec.vectors.cluster.KMeansResult;
import org.elasticsearch.index.codec.vectors.diskbbq.CentroidAssignments;
import org.elasticsearch.index.codec.vectors.diskbbq.CentroidSupplier;
import org.elasticsearch.index.codec.vectors.diskbbq.next.ManifoldErrorCalibrationSelector;
import org.elasticsearch.index.codec.vectors.diskbbq.next.ESNextDiskBBQVectorsFormat;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.TimeUnit;

/**
 * JMH benchmark for {@link ManifoldErrorCalibrationSelector} calibration on synthetic vectors.
 * Run with:
 * {@code ./gradlew -p benchmarks run --args 'AutoQuantizationCalibrationBenchmark'}
 */
@Fork(value = 1, jvmArgsPrepend = { "--add-modules=jdk.incubator.vector" })
@Warmup(iterations = 2, time = 3)
@Measurement(iterations = 3, time = 5)
@BenchmarkMode({ Mode.AverageTime })
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Benchmark)
public class AutoCalibrationBenchmark {

    static {
        Utils.configureBenchmarkLogging();
    }

    private ManifoldErrorCalibrationSelector selector;
    private KMeansFloatVectorValues fvv;
    private CentroidSupplier supplier;
    private CentroidAssignments assignments;
    private FieldInfo fieldInfo;

    @Setup
    public void setup() throws IOException {
        int dimension = 128;
        int numVectors = 8192;
        Random rng = new Random(42);
        float[][] vectors = new float[numVectors][dimension];
        float[] globalCentroid = new float[dimension];
        for (int i = 0; i < numVectors; i++) {
            for (int d = 0; d < dimension; d++) {
                vectors[i][d] = rng.nextFloat() - 0.5f;
                globalCentroid[d] += vectors[i][d];
            }
        }
        for (int d = 0; d < dimension; d++) {
            globalCentroid[d] /= numVectors;
        }
        assignments = new CentroidAssignments(dimension, new float[][] { globalCentroid }, new int[numVectors], new int[0]);
        supplier = CentroidSupplier.fromArray(
            assignments.centroids(),
            KMeansResult.singleCluster(assignments.globalCentroid(), assignments.numCentroids()),
            dimension
        );
        List<float[]> vectorsList = Arrays.asList(vectors);
        fvv = KMeansFloatVectorValues.build(vectorsList, null, dimension);

        fieldInfo = fieldInfoFromTinyIndex(dimension, VectorSimilarityFunction.EUCLIDEAN);

        selector = new ManifoldErrorCalibrationSelector(ESNextDiskBBQVectorsFormat.DEFAULT_VECTORS_PER_CLUSTER);
    }

    private static FieldInfo fieldInfoFromTinyIndex(int dimension, VectorSimilarityFunction similarity) throws IOException {
        try (Directory dir = new ByteBuffersDirectory()) {
            try (IndexWriter w = new IndexWriter(dir, new IndexWriterConfig())) {
                Document doc = new Document();
                doc.add(new KnnFloatVectorField("f", new float[dimension], similarity));
                w.addDocument(doc);
                w.commit();
            }
            try (var reader = DirectoryReader.open(dir)) {
                FieldInfo fi = reader.leaves().get(0).reader().getFieldInfos().fieldInfo("f");
                if (fi == null) {
                    throw new IllegalStateException("missing field info");
                }
                return fi;
            }
        }
    }

    @Benchmark
    public void calibrateSelect(Blackhole blackhole) throws IOException {
        var result = selector.select(fieldInfo, fvv, supplier, assignments.assignments(), assignments.overspillAssignments(), null);
        blackhole.consume(result);
    }
}
