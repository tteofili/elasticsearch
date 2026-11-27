/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq.next;

import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.SerialMergeScheduler;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.util.TestUtil;
import org.elasticsearch.index.codec.vectors.diskbbq.AshConfigResolver;
import org.elasticsearch.index.codec.vectors.diskbbq.IvfFlushConfigSource;
import org.elasticsearch.index.codec.vectors.diskbbq.IvfMergeConfigResolver;
import org.elasticsearch.index.codec.vectors.diskbbq.IvfSegmentConfig;
import org.elasticsearch.index.codec.vectors.diskbbq.QuantEncoding;
import org.elasticsearch.index.mapper.vectors.DenseVectorFieldMapper.ElementType;
import org.elasticsearch.test.ESTestCase;

import java.util.Arrays;
import java.util.Locale;
import java.util.Random;
import java.util.function.Supplier;

/**
 * EXPERIMENT (not a regression test): times only the force merge of identical, fixed-seed segments in an in-memory
 * directory, which is what a whole-index benchmark cannot resolve on a noisy machine with a nearly full disk. Meant to be
 * run in a pristine checkout and in a refactored one, alternating, one JVM each time, and compared by
 * {@code .scratch_perf/merge_micro.sh}. Runs only with {@code -Dexp.mergetiming=true}.
 * <p>
 * Per variant a JVM builds the segments {@code warmup + measured} times and times the merge of each; the first ones warm up
 * the JIT and are reported but not to be used. The ASH variant is a control: the merge code it runs was not changed.
 */
public class DiskBBQMergeTimingExperimentTests extends ESTestCase {

    private static final String FIELD = "f";
    private static final int VECTORS_PER_CLUSTER = 384;
    private static final int CENTROIDS_PER_PARENT = 8;
    private static final float[][] CENTERS = centers(Integer.getInteger("exp.dim", 384));

    private static float[][] centers(int dims) {
        Random r = new Random(777);
        float[][] c = new float[40][dims];
        for (float[] v : c) {
            for (int k = 0; k < dims; k++) {
                v[k] = (float) r.nextGaussian();
            }
        }
        return c;
    }

    private record Variant(String name, Supplier<KnnVectorsFormat> format) {}

    private static ESNextDiskBBQVectorsFormat osq(QuantEncoding encoding, boolean onDiskMerge) {
        return new ESNextDiskBBQVectorsFormat(
            encoding,
            VECTORS_PER_CLUSTER,
            CENTROIDS_PER_PARENT,
            ElementType.FLOAT,
            false,
            null,
            1,
            false,
            ESNextDiskBBQVectorsFormat.DEFAULT_PRECONDITIONING_BLOCK_DIMENSION,
            ESNextDiskBBQVectorsFormat.defaultFlatThreshold(VECTORS_PER_CLUSTER),
            null,
            IvfFlushConfigSource.empty(),
            IvfMergeConfigResolver.useCodecDefault(),
            onDiskMerge
        );
    }

    private static ESNextDiskASHVectorsFormat ash(int bits) {
        return new ESNextDiskASHVectorsFormat(
            AshConfigResolver.dimsDerived(bits, IvfSegmentConfig.AshConfig.DEFAULT_QUERY_BITS_PER_DIM),
            VECTORS_PER_CLUSTER,
            CENTROIDS_PER_PARENT,
            ElementType.FLOAT,
            false,
            null,
            1,
            null,
            -1,
            null,
            IvfFlushConfigSource.empty(),
            IvfMergeConfigResolver.useCodecDefault(),
            false
        );
    }

    private static float[] vector(Random rnd, VectorSimilarityFunction similarity) {
        float[] center = CENTERS[rnd.nextInt(CENTERS.length)];
        float[] v = new float[center.length];
        double norm = 0;
        for (int k = 0; k < v.length; k++) {
            v[k] = center[k] + 0.6f * (float) rnd.nextGaussian();
            norm += v[k] * v[k];
        }
        if (similarity == VectorSimilarityFunction.DOT_PRODUCT) {
            float inverse = (float) (1 / Math.sqrt(norm));
            for (int k = 0; k < v.length; k++) {
                v[k] *= inverse;
            }
        }
        return v;
    }

    /** The nanoseconds that force-merging the segments of the variant took. */
    private static long mergeNanos(Variant variant, VectorSimilarityFunction similarity, int segments, int perSegment) throws Exception {
        Codec codec = TestUtil.alwaysKnnVectorsFormat(variant.format().get());
        IndexWriterConfig iwc = new IndexWriterConfig();
        iwc.setCodec(codec);
        iwc.setUseCompoundFile(false);
        iwc.setMergeScheduler(new SerialMergeScheduler());
        iwc.setRAMBufferSizeMB(2048); // only the explicit commits flush
        Random rnd = new Random(4242);
        try (Directory dir = new ByteBuffersDirectory(); IndexWriter w = new IndexWriter(dir, iwc)) {
            for (int s = 0; s < segments; s++) {
                for (int i = 0; i < perSegment; i++) {
                    Document doc = new Document();
                    doc.add(new KnnFloatVectorField(FIELD, vector(rnd, similarity), similarity));
                    w.addDocument(doc);
                }
                w.commit();
            }
            long start = System.nanoTime();
            w.forceMerge(1);
            w.commit();
            return System.nanoTime() - start;
        }
    }

    public void testMergeTimings() throws Exception {
        assumeTrue("exp.mergetiming is not set", Boolean.getBoolean("exp.mergetiming"));
        int segments = Integer.getInteger("exp.segments", 4);
        int perSegment = Integer.getInteger("exp.perSegment", 6000);
        int warmup = Integer.getInteger("exp.warmup", 2);
        int measured = Integer.getInteger("exp.measured", 5);
        String tag = System.getProperty("exp.tag", "tree");
        Variant[] variants = {
            new Variant("osq1", () -> osq(QuantEncoding.ONE_BIT_4BIT_QUERY, false)),
            new Variant("osq1-ondisk", () -> osq(QuantEncoding.ONE_BIT_4BIT_QUERY, true)),
            new Variant("osq4", () -> osq(QuantEncoding.FOUR_BIT_SYMMETRIC, false)),
            new Variant("ash2-control", () -> ash(2)) };
        for (Variant variant : variants) {
            long[] nanos = new long[warmup + measured];
            for (int i = 0; i < nanos.length; i++) {
                nanos[i] = mergeNanos(variant, VectorSimilarityFunction.DOT_PRODUCT, segments, perSegment);
            }
            long[] kept = Arrays.copyOfRange(nanos, warmup, nanos.length);
            Arrays.sort(kept);
            StringBuilder all = new StringBuilder();
            for (long n : nanos) {
                all.append(String.format(Locale.ROOT, " %.1f", n / 1e6));
            }
            logger.info(
                "EXP8 tag={} variant={} median_ms={} min_ms={} all_ms:{}",
                tag,
                variant.name(),
                String.format(Locale.ROOT, "%.1f", kept[kept.length / 2] / 1e6),
                String.format(Locale.ROOT, "%.1f", kept[0] / 1e6),
                all
            );
        }
    }
}
