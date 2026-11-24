/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq;

import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopKnnCollector;
import org.apache.lucene.util.FixedBitSet;
import org.elasticsearch.simdvec.BulkMask;
import org.elasticsearch.test.ESTestCase;

import static org.hamcrest.Matchers.equalTo;

public class PostingBlocksTests extends ESTestCase {

    private static final int BLOCK = 32;

    private static int[] ascendingDocs(int count) {
        int[] docs = new int[count];
        int doc = randomIntBetween(0, 5);
        for (int i = 0; i < count; i++) {
            doc += randomIntBetween(1, 4);
            docs[i] = doc;
        }
        return docs;
    }

    public void testAcceptedMaskWithoutAcceptDocsAcceptsEverything() {
        int bulkSize = randomIntBetween(1, BLOCK);
        int[] docs = ascendingDocs(bulkSize);
        int[] before = docs.clone();
        int mask = PostingBlocks.acceptedMask(docs, null, bulkSize);
        assertThat(mask, equalTo(BulkMask.all(bulkSize)));
        assertArrayEquals(before, docs);
    }

    public void testAcceptedMaskMarksRejectedDocsAndAlreadyExcluded() {
        int bulkSize = randomIntBetween(1, BLOCK);
        int[] docs = ascendingDocs(bulkSize);
        FixedBitSet live = new FixedBitSet(docs[bulkSize - 1] + 1);
        int[] expected = docs.clone();
        int expectedMask = 0;
        for (int i = 0; i < bulkSize; i++) {
            if (randomBoolean()) {
                live.set(docs[i]);
            }
            if (randomIntBetween(0, 7) == 0) {
                docs[i] = -1; // already excluded, e.g. outside a slice
            }
            if (docs[i] == -1 || live.get(docs[i]) == false) {
                expected[i] = -1;
            } else {
                expectedMask = BulkMask.select(expectedMask, i);
            }
        }
        assertThat(PostingBlocks.acceptedMask(docs, live, bulkSize), equalTo(expectedMask));
        for (int i = 0; i < bulkSize; i++) {
            assertThat(docs[i] == -1, equalTo(expected[i] == -1));
        }
    }

    public void testSlicedAcceptedMaskHonoursExcludedDocsWithoutAcceptDocs() {
        int bulkSize = randomIntBetween(1, BLOCK);
        int[] docs = ascendingDocs(bulkSize);
        int expectedMask = 0;
        for (int i = 0; i < bulkSize; i++) {
            if (randomBoolean()) {
                docs[i] = -1;
            } else {
                expectedMask = BulkMask.select(expectedMask, i);
            }
        }
        // unlike acceptedMask, a null acceptDocs must still drop the -1 entries
        assertThat(PostingBlocks.slicedAcceptedMask(docs, null, bulkSize), equalTo(expectedMask));
    }

    public void testSlicedAcceptedMaskMatchesAcceptedMaskWhenAcceptDocsPresent() {
        int bulkSize = randomIntBetween(1, BLOCK);
        int[] docs = ascendingDocs(bulkSize);
        FixedBitSet live = new FixedBitSet(docs[bulkSize - 1] + 1);
        live.set(0, live.length());
        for (int i = 0; i < bulkSize; i++) {
            if (randomIntBetween(0, 3) == 0) {
                live.clear(docs[i]);
            }
        }
        int[] other = docs.clone();
        assertThat(PostingBlocks.slicedAcceptedMask(docs, live, bulkSize), equalTo(PostingBlocks.acceptedMask(other, live, bulkSize)));
        assertArrayEquals(other, docs);
    }

    public void testCollectBulkCollectsExactlyTheAcceptedDocs() {
        int bulkSize = randomIntBetween(1, BLOCK);
        int[] docs = ascendingDocs(bulkSize);
        float[] scores = new float[bulkSize];
        FixedBitSet live = new FixedBitSet(docs[bulkSize - 1] + 1);
        for (int i = 0; i < bulkSize; i++) {
            scores[i] = randomFloat();
            if (randomBoolean()) {
                live.set(docs[i]);
            }
        }
        int mask = PostingBlocks.acceptedMask(docs, live, bulkSize);
        float maxScore = Float.NEGATIVE_INFINITY;
        int accepted = 0;
        for (int i = 0; i < bulkSize; i++) {
            if (docs[i] != -1) {
                accepted++;
                maxScore = Math.max(maxScore, scores[i]);
            }
        }
        assertThat(accepted, equalTo(Integer.bitCount(mask)));
        float[] expectedScores = scores.clone();
        int[] expectedDocs = docs.clone();

        // a plain collector takes the doc-by-doc path
        KnnCollector collector = new TopKnnCollector(bulkSize, Integer.MAX_VALUE);
        PostingBlocks.collectBulk(collector, docs, scores, bulkSize, mask, accepted, maxScore);
        TopDocs top = collector.topDocs();
        assertThat(top.scoreDocs.length, equalTo(accepted));
        for (var sd : top.scoreDocs) {
            int pos = -1;
            for (int i = 0; i < bulkSize; i++) {
                if (expectedDocs[i] == sd.doc) {
                    pos = i;
                }
            }
            assertTrue("collected a doc that was not accepted: " + sd.doc, pos >= 0);
            assertThat(sd.score, equalTo(expectedScores[pos]));
        }
    }
}
