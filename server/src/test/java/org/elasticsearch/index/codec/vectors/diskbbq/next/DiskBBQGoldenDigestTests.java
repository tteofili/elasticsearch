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
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnByteVectorField;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.SortedDocValuesField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.SerialMergeScheduler;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.tests.util.TestUtil;
import org.apache.lucene.util.BytesRef;
import org.elasticsearch.index.codec.vectors.diskbbq.AshConfigResolver;
import org.elasticsearch.index.codec.vectors.diskbbq.IvfFlushConfigSource;
import org.elasticsearch.index.codec.vectors.diskbbq.IvfMergeConfigResolver;
import org.elasticsearch.index.codec.vectors.diskbbq.IvfQueryConfigResolver;
import org.elasticsearch.index.codec.vectors.diskbbq.IvfSegmentConfig;
import org.elasticsearch.index.codec.vectors.diskbbq.QuantEncoding;
import org.elasticsearch.index.mapper.vectors.DenseVectorFieldMapper.ElementType;
import org.elasticsearch.search.vectors.IVFKnnByteVectorQuery;
import org.elasticsearch.search.vectors.IVFKnnFloatVectorQuery;
import org.elasticsearch.test.ESTestCase;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HexFormat;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.TreeSet;
import java.util.function.Supplier;

/**
 * Format stability check for the quantizer-agnostic DiskBBQ refactor (not a conventional test). It pins what the
 * {@code ESNext} IVF formats write: it builds fixed-seed segments through each quantizer, once as flushed segments and
 * once after a force merge, and logs a {@code GOLDEN} line with a digest of
 * <ul>
 *   <li>every IVF file body ({@code .cenivf}, {@code .clivf}, {@code .mivf}; the Lucene header, which carries a random
 *       segment id, and the footer are skipped, and ASH's projection matrix, which carries a random id and the result
 *       of a training that is not bit-reproducible across JVM runs, is zeroed), and</li>
 *   <li>the exact top-k doc ids and score bits of a fixed set of queries, for an exhaustive and a partial visit.</li>
 * </ul>
 * A refactoring step must reproduce the {@code GOLDEN} lines of the baseline exactly. When the system property
 * {@code diskbbq.golden.dump} names a directory, the masked file bodies and the results are also written there, so
 * that a mismatch can be classified (a few floats off by a tiny margin, or a structural change) by
 * {@code .scratch_golden/compare_dumps.py}.
 * <p>
 * Scenarios cover the quantizers and bit widths, the similarity functions, float, byte and bfloat16 vectors, on-disk
 * merge, merges with deletions, segments of uneven and tiny sizes, and sliced segments (whose files are compared, but
 * not searched, as that needs sliced queries). The only assertion is that building the same scenario repeatedly
 * converges to the same digest, i.e. that the digest is stable enough to gate on.
 */
public class DiskBBQGoldenDigestTests extends ESTestCase {

    private static final String FIELD = "f";
    private static final String SLICE_FIELD = "_slice";
    private static final int DIMS = 128;
    // the last batch is below the flat threshold, the others leave ragged tail blocks
    private static final int[] BATCH_SIZES = { 700, 190, 513, 40 };
    private static final int VECTORS_PER_CLUSTER = ESNextDiskBBQVectorsFormat.MIN_VECTORS_PER_CLUSTER;
    private static final int CENTROIDS_PER_PARENT = 8;
    private static final int QUERIES = 20;
    private static final VectorSimilarityFunction[] FLOAT_SIMILARITIES = {
        VectorSimilarityFunction.DOT_PRODUCT,
        VectorSimilarityFunction.EUCLIDEAN,
        VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT };
    private static final VectorSimilarityFunction[] BYTE_SIMILARITIES = {
        VectorSimilarityFunction.DOT_PRODUCT,
        VectorSimilarityFunction.EUCLIDEAN };
    private static final VectorSimilarityFunction[] ONE_SIMILARITY = { VectorSimilarityFunction.EUCLIDEAN };

    /** Cluster centers, so the data has the structure IVF partitions on. */
    private static final float[][] CENTERS = centers();

    private static float[][] centers() {
        Random r = new Random(777);
        float[][] c = new float[30][DIMS];
        for (float[] v : c) {
            for (int k = 0; k < DIMS; k++) {
                v[k] = (float) r.nextGaussian();
            }
        }
        return c;
    }

    private static float[] floatVector(Random rnd, VectorSimilarityFunction sim) {
        float[] center = CENTERS[rnd.nextInt(CENTERS.length)];
        float[] v = new float[DIMS];
        double norm = 0;
        for (int k = 0; k < DIMS; k++) {
            v[k] = center[k] + 0.6f * (float) rnd.nextGaussian();
            norm += v[k] * v[k];
        }
        if (sim == VectorSimilarityFunction.DOT_PRODUCT) {
            float inv = (float) (1 / Math.sqrt(norm));
            for (int k = 0; k < DIMS; k++) {
                v[k] *= inv;
            }
        }
        return v;
    }

    private static byte[] byteVector(Random rnd) {
        float[] center = CENTERS[rnd.nextInt(CENTERS.length)];
        byte[] v = new byte[DIMS];
        for (int k = 0; k < DIMS; k++) {
            v[k] = (byte) Math.max(-127, Math.min(127, Math.round(20 * (center[k] + 0.6f * (float) rnd.nextGaussian()))));
        }
        return v;
    }

    private enum Kind {
        FLOAT,
        BYTE
    }

    /**
     * @param search whether the segments are searched; sliced segments need sliced queries, so only their files are compared
     */
    private record Scenario(
        String name,
        Supplier<KnnVectorsFormat> format,
        int quantBits,
        Kind kind,
        VectorSimilarityFunction[] sims,
        boolean deletes,
        boolean sliced,
        boolean search
    ) {
        Scenario(String name, Supplier<KnnVectorsFormat> format, int quantBits, VectorSimilarityFunction[] sims) {
            this(name, format, quantBits, Kind.FLOAT, sims, false, false, true);
        }
    }

    private static ESNextDiskBBQVectorsFormat osqFormat(
        QuantEncoding enc,
        ElementType elementType,
        boolean precondition,
        boolean onDiskMerge,
        String sliceField
    ) {
        return new ESNextDiskBBQVectorsFormat(
            enc,
            VECTORS_PER_CLUSTER,
            CENTROIDS_PER_PARENT,
            elementType,
            false,
            null,
            1,
            precondition,
            ESNextDiskBBQVectorsFormat.DEFAULT_PRECONDITIONING_BLOCK_DIMENSION,
            ESNextDiskBBQVectorsFormat.defaultFlatThreshold(VECTORS_PER_CLUSTER),
            sliceField,
            IvfFlushConfigSource.empty(),
            IvfMergeConfigResolver.useCodecDefault(),
            onDiskMerge
        );
    }

    private static ESNextDiskASHVectorsFormat ashFormat(int bits, boolean onDiskMerge, String sliceField) {
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
            sliceField,
            IvfFlushConfigSource.empty(),
            IvfMergeConfigResolver.useCodecDefault(),
            onDiskMerge
        );
    }

    private static List<Scenario> scenarios() {
        List<Scenario> out = new ArrayList<>();
        // quantizers and bit widths, all similarities
        for (QuantEncoding enc : new QuantEncoding[] {
            QuantEncoding.ONE_BIT_4BIT_QUERY,
            QuantEncoding.TWO_BIT_4BIT_QUERY,
            QuantEncoding.FOUR_BIT_SYMMETRIC,
            QuantEncoding.SEVEN_BIT_SYMMETRIC }) {
            out.add(
                new Scenario(
                    "osq-" + enc.name(),
                    () -> osqFormat(enc, ElementType.FLOAT, false, false, null),
                    enc.bits(),
                    FLOAT_SIMILARITIES
                )
            );
        }
        out.add(
            new Scenario(
                "osq-ONE_BIT_4BIT_QUERY-precondition",
                () -> osqFormat(QuantEncoding.ONE_BIT_4BIT_QUERY, ElementType.FLOAT, true, false, null),
                1,
                FLOAT_SIMILARITIES
            )
        );
        for (int bits : new int[] { 1, 2, 4 }) {
            out.add(new Scenario("ash-b" + bits, () -> ashFormat(bits, false, null), bits, FLOAT_SIMILARITIES));
        }
        // on-disk merge
        out.add(
            new Scenario(
                "osq-FOUR_BIT_SYMMETRIC-onDiskMerge",
                () -> osqFormat(QuantEncoding.FOUR_BIT_SYMMETRIC, ElementType.FLOAT, false, true, null),
                4,
                ONE_SIMILARITY
            )
        );
        out.add(new Scenario("ash-b2-onDiskMerge", () -> ashFormat(2, true, null), 2, ONE_SIMILARITY));
        // byte and bfloat16 vectors
        for (QuantEncoding enc : new QuantEncoding[] { QuantEncoding.ONE_BIT_4BIT_QUERY, QuantEncoding.FOUR_BIT_SYMMETRIC }) {
            out.add(
                new Scenario(
                    "osq-byte-" + enc.name(),
                    () -> osqFormat(enc, ElementType.BYTE, false, false, null),
                    enc.bits(),
                    Kind.BYTE,
                    BYTE_SIMILARITIES,
                    false,
                    false,
                    true
                )
            );
            out.add(
                new Scenario(
                    "osq-bfloat16-" + enc.name(),
                    () -> osqFormat(enc, ElementType.BFLOAT16, false, false, null),
                    enc.bits(),
                    ONE_SIMILARITY
                )
            );
        }
        // merges with deletions
        out.add(
            new Scenario(
                "osq-ONE_BIT_4BIT_QUERY-deletes",
                () -> osqFormat(QuantEncoding.ONE_BIT_4BIT_QUERY, ElementType.FLOAT, false, false, null),
                1,
                Kind.FLOAT,
                ONE_SIMILARITY,
                true,
                false,
                true
            )
        );
        out.add(new Scenario("ash-b2-deletes", () -> ashFormat(2, false, null), 2, Kind.FLOAT, ONE_SIMILARITY, true, false, true));
        // sliced segments: flushed ones list vectors in ordinal order without doc ids, merged ones carry slice tables
        out.add(
            new Scenario(
                "osq-ONE_BIT_4BIT_QUERY-sliced",
                () -> osqFormat(QuantEncoding.ONE_BIT_4BIT_QUERY, ElementType.FLOAT, false, false, SLICE_FIELD),
                1,
                Kind.FLOAT,
                ONE_SIMILARITY,
                false,
                true,
                false
            )
        );
        out.add(new Scenario("ash-b2-sliced", () -> ashFormat(2, false, SLICE_FIELD), 2, Kind.FLOAT, ONE_SIMILARITY, false, true, false));
        return out;
    }

    public void testGoldenDigests() throws Exception {
        String dump = System.getProperty("diskbbq.golden.dump");
        for (Scenario scenario : scenarios()) {
            for (VectorSimilarityFunction sim : scenario.sims()) {
                // Cold (interpreted or C1) and C2-compiled code can sum floats in a different order, so builds made while
                // the code is still warming up may differ. Build until two consecutive builds agree.
                String previous = digest(scenario, sim, null);
                String current = null;
                for (int i = 0; i < 10; i++) {
                    current = digest(scenario, sim, null);
                    if (current.equals(previous)) {
                        break;
                    }
                    previous = current;
                }
                assertEquals("digest did not converge for " + scenario.name() + " " + sim, previous, current);
                if (dump != null) {
                    // The dump has to be of the build whose digest is logged. Code that is still warming up can make a
                    // build differ even after two consecutive ones agreed, so repeat until the dumped build agrees too.
                    Path dir = Path.of(dump).resolve(scenario.name() + "__" + sim);
                    String dumped = null;
                    for (int i = 0; i < 10; i++) {
                        Files.createDirectories(dir);
                        dumped = digest(scenario, sim, dir);
                        if (dumped.equals(current)) {
                            break;
                        }
                        current = dumped;
                    }
                    assertEquals("dumped build differs from the digest for " + scenario.name() + " " + sim, current, dumped);
                }
                logger.info("GOLDEN {} {} {}", scenario.name(), sim, current);
            }
        }
    }

    private String digest(Scenario scenario, VectorSimilarityFunction sim, Path dumpDir) throws Exception {
        StringBuilder out = new StringBuilder();
        Random rnd = new Random(12345);
        Object[] queries = new Object[QUERIES];
        for (int i = 0; i < QUERIES; i++) {
            queries[i] = scenario.kind() == Kind.BYTE ? byteVector(rnd) : floatVector(rnd, sim);
        }
        Codec codec = TestUtil.alwaysKnnVectorsFormat(scenario.format().get());
        try (Directory dir = new ByteBuffersDirectory()) {
            IndexWriterConfig iwc = new IndexWriterConfig();
            iwc.setCodec(codec);
            iwc.setUseCompoundFile(false);
            iwc.setMergeScheduler(new SerialMergeScheduler());
            iwc.setRAMBufferSizeMB(1024); // only the explicit commits flush
            if (scenario.sliced()) {
                iwc.setIndexSort(new Sort(new SortField(SLICE_FIELD, SortField.Type.STRING, false, SortField.STRING_LAST)));
            }
            try (IndexWriter w = new IndexWriter(dir, iwc)) {
                int id = 0;
                for (int size : BATCH_SIZES) {
                    for (int i = 0; i < size; i++, id++) {
                        Document doc = new Document();
                        doc.add(new StringField("id", Integer.toString(id), Field.Store.NO));
                        if (scenario.sliced()) {
                            doc.add(SortedDocValuesField.indexedField(SLICE_FIELD, new BytesRef("s" + (id % 3))));
                        }
                        if (scenario.kind() == Kind.BYTE) {
                            doc.add(new KnnByteVectorField(FIELD, byteVector(rnd), sim));
                        } else {
                            doc.add(new KnnFloatVectorField(FIELD, floatVector(rnd, sim), sim));
                        }
                        w.addDocument(doc);
                    }
                    w.commit();
                }
                out.append("flushed[").append(describe(dir, w, queries, scenario, "flushed", dumpDir)).append("] ");
                if (scenario.deletes()) {
                    for (int i = 3; i < id; i += 10) {
                        w.deleteDocuments(new Term("id", Integer.toString(i)));
                    }
                    w.commit();
                }
                w.forceMerge(1);
                w.commit();
                out.append("merged[").append(describe(dir, w, queries, scenario, "merged", dumpDir)).append("]");
            }
        }
        return out.toString();
    }

    private String describe(Directory dir, IndexWriter w, Object[] queries, Scenario scenario, String phase, Path dumpDir)
        throws Exception {
        boolean ash = scenario.name().startsWith("ash");
        MessageDigest sha = MessageDigest.getInstance("SHA-256");
        StringBuilder desc = new StringBuilder();
        TreeSet<String> files = new TreeSet<>();
        for (String name : dir.listAll()) {
            if (name.endsWith(".cenivf") || name.endsWith(".clivf") || name.endsWith(".mivf")) {
                files.add(name);
            }
        }
        for (String name : files) {
            byte[] body = body(dir, name);
            if (ash && name.endsWith(".cenivf")) {
                maskAshMatrix(body);
            }
            String ext = name.substring(name.lastIndexOf('.'));
            sha.update(ext.getBytes(StandardCharsets.UTF_8));
            sha.update(body);
            desc.append(ext)
                .append(':')
                .append(body.length)
                .append('#')
                .append(HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(body), 0, 4))
                .append(' ');
            if (dumpDir != null) {
                // dropping the codec name from the file name keeps dumps of different runs comparable
                Files.write(dumpDir.resolve(phase + "." + name.replaceAll("^_(\\d+)_.*(\\.\\w+)$", "seg$1$2") + ".bin"), body);
            }
        }
        desc.append("files=").append(HexFormat.of().formatHex(sha.digest(), 0, 8)).append(' ');

        if (scenario.search()) {
            MessageDigest results = MessageDigest.getInstance("SHA-256");
            StringBuilder dumped = new StringBuilder();
            try (DirectoryReader reader = DirectoryReader.open(w)) {
                IndexSearcher searcher = new IndexSearcher(reader);
                var resolver = IvfQueryConfigResolver.from(false, false, scenario.quantBits(), 1f, null);
                for (float visitRatio : new float[] { 1f, 0.2f }) {
                    for (Object q : queries) {
                        Query query = scenario.kind() == Kind.BYTE
                            ? new IVFKnnByteVectorQuery(FIELD, (byte[]) q, 10, 100, null, visitRatio, resolver)
                            : new IVFKnnFloatVectorQuery(FIELD, (float[]) q, 10, 100, null, visitRatio, resolver);
                        for (ScoreDoc sd : searcher.search(query, 10).scoreDocs) {
                            results.update(Integer.toString(sd.doc).getBytes(StandardCharsets.UTF_8));
                            results.update(Integer.toString(Float.floatToIntBits(sd.score)).getBytes(StandardCharsets.UTF_8));
                            dumped.append(sd.doc).append(':').append(sd.score).append(' ');
                        }
                        results.update((byte) '|');
                        dumped.append('\n');
                    }
                }
            }
            if (dumpDir != null) {
                Files.writeString(dumpDir.resolve(phase + ".results.txt"), dumped.toString());
            }
            desc.append("results=").append(HexFormat.of().formatHex(results.digest(), 0, 8));
        } else {
            desc.append("results=none");
        }
        return desc.toString().toLowerCase(Locale.ROOT);
    }

    /**
     * ASH serializes its projection matrix as {@code [int originalDim][int nDims][byte learned][16 bytes id]
     * [int trainingSampleSize][nDims * originalDim floats]}. The random id is different in every build and the floats
     * are the result of a training that is not bit-reproducible across JVM runs, so the whole matrix is zeroed in the
     * digest. Its effect is still covered: the codes in the posting lists and the search results are hashed.
     */
    private static void maskAshMatrix(byte[] body) {
        for (int i = 0; i + 29 <= body.length; i++) {
            if (body[i + 8] != 0 && body[i + 8] != 1) {
                continue;
            }
            int dim = readInt(body, i, true);
            int nDims = readInt(body, i + 4, true);
            if (dim != DIMS || nDims < 1 || nDims > DIMS) {
                dim = readInt(body, i, false);
                nDims = readInt(body, i + 4, false);
                if (dim != DIMS || nDims < 1 || nDims > DIMS) {
                    continue;
                }
            }
            int end = Math.min(body.length, i + 9 + 16 + 4 + nDims * DIMS * Float.BYTES);
            Arrays.fill(body, i + 9, end, (byte) 0);
            return;
        }
        // a segment that is too small to have an IVF structure has no matrix
    }

    private static int readInt(byte[] b, int off, boolean bigEndian) {
        return bigEndian
            ? ((b[off] & 0xFF) << 24) | ((b[off + 1] & 0xFF) << 16) | ((b[off + 2] & 0xFF) << 8) | (b[off + 3] & 0xFF)
            : ((b[off + 3] & 0xFF) << 24) | ((b[off + 2] & 0xFF) << 16) | ((b[off + 1] & 0xFF) << 8) | (b[off] & 0xFF);
    }

    /** The file without its Lucene index header (which holds a random segment id) and footer (a checksum of it). */
    private static byte[] body(Directory dir, String name) throws IOException {
        try (IndexInput in = dir.openInput(name, IOContext.READONCE)) {
            long length = in.length();
            in.readInt(); // magic
            in.readString(); // codec name
            in.readInt(); // version
            in.skipBytes(16); // segment id
            int suffix = in.readByte() & 0xFF;
            in.skipBytes(suffix);
            long start = in.getFilePointer();
            int bodyLength = Math.toIntExact(length - CodecUtil.footerLength() - start);
            byte[] body = new byte[bodyLength];
            in.readBytes(body, 0, bodyLength);
            return body;
        }
    }
}
