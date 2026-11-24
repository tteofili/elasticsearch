/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq;

import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.Bits;
import org.elasticsearch.search.vectors.BulkKnnCollector;
import org.elasticsearch.simdvec.BulkMask;

import java.io.IOException;

/**
 * The block-level steps of visiting a posting list that do not depend on how vectors are quantized: decoding the doc
 * ids of a block, deciding which of them to score, and handing scored blocks to the collector. Every quantizer's
 * posting visitor needs exactly these, so they live here once instead of in each visitor.
 * <p>
 * A block holds at most {@link Integer#SIZE} vectors, as a block's accepted documents are tracked in an {@code int}
 * bit mask ({@link BulkMask}). Doc ids that must not be scored are marked with {@code -1} in the doc id array.
 * All methods are stateless; callers own the scratch arrays and the running ordinal base.
 */
public final class PostingBlocks {

    private PostingBlocks() {}

    /**
     * Reads the delta encoded doc ids of the next block and turns them into absolute doc ids.
     *
     * @param docIds destination, also receives the decoded deltas before they are made absolute
     * @param count number of doc ids in the block
     * @param docBase the last absolute doc id of the previous block (0 for the first block of a posting list)
     * @return the last absolute doc id of this block, to be passed as {@code docBase} for the next one
     */
    public static int readDocIds(IndexInput in, DocIdsWriter idsWriter, byte docEncoding, int[] docIds, int count, int docBase)
        throws IOException {
        idsWriter.readInts(in, count, docEncoding, docIds);
        // reconstitute from the deltas
        for (int j = 0; j < count; j++) {
            docBase += docIds[j];
            docIds[j] = docBase;
        }
        return docBase;
    }

    /**
     * Resolves the doc ids of the next block for a flushed, sliced posting list, which stores no doc ids and lists its
     * vectors in ordinal order. Vectors whose doc is outside {@code [startDocId, endDocId)} are marked {@code -1}.
     *
     * @param docBase ordinal of the first vector of the block
     * @return the ordinal of the first vector of the next block
     */
    public static int readSlicedDocIds(KnnVectorValues values, int startDocId, int endDocId, int[] docIds, int count, int docBase) {
        for (int j = 0; j < count; j++) {
            int docId = values.ordToDoc(docBase++);
            if (docId >= startDocId && docId < endDocId) {
                docIds[j] = docId;
            } else {
                docIds[j] = -1;
            }
        }
        return docBase;
    }

    /**
     * Computes which docs of a block are accepted, marking the others {@code -1} in {@code docIds}.
     *
     * @param acceptDocs live docs and filter, or {@code null} to accept every doc
     * @return a {@link BulkMask} of the accepted positions
     */
    public static int acceptedMask(int[] docIds, Bits acceptDocs, int bulkSize) {
        assertFitsMask(bulkSize);
        if (acceptDocs == null) {
            return BulkMask.all(bulkSize);
        }
        int mask = 0;
        for (int i = 0; i < bulkSize; i++) {
            if (docIds[i] == -1 || acceptDocs.get(docIds[i]) == false) {
                docIds[i] = -1;
            } else {
                mask = BulkMask.select(mask, i);
            }
        }
        return mask;
    }

    /**
     * Like {@link #acceptedMask}, for blocks whose doc ids were produced by {@link #readSlicedDocIds}: {@code -1}
     * entries are already out of the slice, and the loop must run even when there is no {@code acceptDocs}.
     */
    public static int slicedAcceptedMask(int[] docIds, Bits acceptDocs, int bulkSize) {
        assertFitsMask(bulkSize);
        int mask = 0;
        for (int i = 0; i < bulkSize; i++) {
            if (docIds[i] == -1 || (acceptDocs != null && acceptDocs.get(docIds[i]) == false)) {
                docIds[i] = -1;
            } else {
                mask = BulkMask.select(mask, i);
            }
        }
        return mask;
    }

    /**
     * Hands the scored docs of a block to the collector. For a {@link BulkKnnCollector} the accepted docs and scores
     * are compacted to the front of the arrays first, which is why both are modified.
     *
     * @param mask the accepted positions, as returned by {@link #acceptedMask}
     * @param docsToScore the number of accepted positions, {@code Integer.bitCount(mask)}
     * @param maxScore the highest score of the block
     */
    public static void collectBulk(
        KnnCollector knnCollector,
        int[] docIds,
        float[] scores,
        int bulkSize,
        int mask,
        int docsToScore,
        float maxScore
    ) {
        if (knnCollector instanceof BulkKnnCollector bulkCollector) {
            if (docsToScore == bulkSize) {
                bulkCollector.bulkCollect(docIds, scores, bulkSize, maxScore);
                return;
            }
            int i = 0;
            for (int m = mask; m != 0; m = BulkMask.deselectFirst(m), i++) {
                int offset = BulkMask.first(m);
                docIds[i] = docIds[offset];
                scores[i] = scores[offset];
            }
            bulkCollector.bulkCollect(docIds, scores, docsToScore, maxScore);
            return;
        }
        for (int i = 0; i < bulkSize; i++) {
            final int doc = docIds[i];
            if (doc != -1) {
                knnCollector.collect(doc, scores[i]);
            }
        }
    }

    private static void assertFitsMask(int bulkSize) {
        assert bulkSize <= Integer.SIZE
            : "bulk size "
                + bulkSize
                + " exceeds the mask width; split the block into Integer.SIZE-sized parts and call scoreBulkMasked once per part";
    }
}
