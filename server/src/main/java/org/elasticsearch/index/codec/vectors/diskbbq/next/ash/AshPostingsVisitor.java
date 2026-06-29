/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq.next.ash;

import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.Bits;
import org.elasticsearch.index.codec.vectors.diskbbq.DocIdsWriter;
import org.elasticsearch.index.codec.vectors.diskbbq.IVFVectorsReader;
import org.elasticsearch.index.codec.vectors.diskbbq.PostingMetadata;
import org.elasticsearch.search.vectors.BulkKnnCollector;
import org.elasticsearch.simdvec.ESVectorUtil;

import static org.elasticsearch.simdvec.ES940OSQVectorsScorer.BULK_SIZE;

/**
 * PostingVisitor for ASH-encoded posting lists.
 * <p>
 * Reads bit-packed codes with float16 scale/offset per vector and scores them
 * asymmetrically using the precomputed query transforms. Each vector stores a
 * 1-byte ASH cluster ID indicating which centering centroid was used during
 * encoding, allowing the scorer to select the correct transformed query vector.
 * <p>
 * The on-disk per-vector format is:
 * {@code [byte ashClusterId][short scale_f16][short offset_f16][byte[packedCodeBytes] packed_codes]}
 */
public class AshPostingsVisitor implements IVFVectorsReader.PostingVisitor {

    private final float[][] w;
    private final float[][] ashCentroids;
    private final float[] query;
    private final IndexInput parentsSlice;
    private final float[] globalCentroid;
    private final FieldInfo fieldInfo;
    private final IndexInput indexInput;
    private final Bits acceptDocs;
    private final int nDims;
    private final int bitsPerDim;
    private final int packedCodeBytes;
    private final DocIdsWriter idsWriter = new DocIdsWriter();
    private final int[] docIdsScratch = new int[BULK_SIZE];
    private final int[] offsetsScratch = new int[BULK_SIZE];
    private final float[] scores = new float[BULK_SIZE];

    // SIMD scoring support: precomputed values and scratch buffers
    private final float[] queryTransformedPadded; // zero-padded to planeBytes * 8
    private final float sumAllQt; // sum of all queryTransformed values
    private final int planeBytes; // bytes per bit-plane = ceil(nDims/8)
    // Bulk read buffers for structure-of-arrays layout
    private final byte[] bulkCodeBuf; // BULK_SIZE * packedCodeBytes
    private final float[] bulkScales; // decoded scales for one block (Java fallback path)
    private final float[] bulkOffsets; // decoded offsets for one block (Java fallback path)
    // Native scorer buffers (raw fp16 — no Java-side decode needed)
    private final short[] bulkScalesF16; // raw fp16 scales for native path
    private final short[] bulkOffsetsF16; // raw fp16 offsets for native path
    private final short[] bulkDocSums; // precomputed docSums for D2Q4 correction

    // D2Q4 integer scoring path (optional, query-time flag)
    private final boolean useD2Q4Scoring;
    private final byte[] queryQuantized4Bit; // 4-bit quantized query in striped format
    private final float invQScale; // inverse quantization scale
    private final float qOffset; // quantization offset (min of queryTransformed)
    private final float constantCorrection; // precomputed: 1.5 * (queryUnsignedSum * invQScale + qOffset * nDims)

    // Per-ASH-cluster precomputed query transforms (lazily populated)
    private final float[] queryTransformed;
    private final float[] queryDotCentroidByCluster;
    private boolean clusterTransformsReady;
    private float currentQueryDotCentroid;

    // Per-posting-list state
    private int vectors;
    private byte docEncoding;
    private int docBase;
    private long slicePos;
    private float centroidDistance;
    private final org.apache.lucene.index.VectorSimilarityFunction similarityFunction;
    private final float[] currentCentroid;

    public AshPostingsVisitor(
        float[][] w,
        float[][] wT,
        float[][] ashCentroids,
        float[] query,
        IndexInput parentsSlice,
        float[] globalCentroid,
        FieldInfo fieldInfo,
        IndexInput indexInput,
        Bits acceptDocs,
        int bitsPerDim,
        boolean useD2Q4Scoring
    ) {
        this.w = w;
        this.ashCentroids = ashCentroids;
        this.query = query;
        this.parentsSlice = parentsSlice;
        this.globalCentroid = globalCentroid;
        this.fieldInfo = fieldInfo;
        this.indexInput = indexInput;
        this.acceptDocs = acceptDocs;
        this.nDims = w[0].length;
        this.bitsPerDim = bitsPerDim;
        this.packedCodeBytes = AsymmetricHashingScorer.packedByteLength(nDims, bitsPerDim);
        this.similarityFunction = fieldInfo.getVectorSimilarityFunction();
        this.currentCentroid = new float[fieldInfo.getVectorDimension()];

        // Pre-allocate per-ASH-cluster arrays
        int nAshClusters = ashCentroids != null ? ashCentroids.length : 0;
        this.queryTransformed = new float[nDims];
        this.queryDotCentroidByCluster = new float[nAshClusters];
        this.clusterTransformsReady = false;

        for (int j = 0; j < nDims; j++) {
            queryTransformed[j] = ESVectorUtil.dotProduct(query, wT[j]);
        }

        // SIMD scoring setup: padded query, precomputed sum, plane scratch buffers
        this.planeBytes = (nDims + 7) >>> 3;
        int paddedLen = planeBytes * Byte.SIZE;
        this.queryTransformedPadded = new float[paddedLen];
        System.arraycopy(queryTransformed, 0, queryTransformedPadded, 0, nDims);
        // remaining positions stay 0 (zero-padded for nDims not multiple of 8)

        float sumQt = 0;
        for (int j = 0; j < nDims; j++) {
            sumQt += queryTransformed[j];
        }
        this.sumAllQt = sumQt;

        this.bulkCodeBuf = new byte[BULK_SIZE * packedCodeBytes];
        this.bulkScales = new float[BULK_SIZE];
        this.bulkOffsets = new float[BULK_SIZE];
        this.bulkScalesF16 = new short[BULK_SIZE];
        this.bulkOffsetsF16 = new short[BULK_SIZE];
        this.bulkDocSums = new short[BULK_SIZE];

        // D2Q4 integer scoring setup (optional)
        this.useD2Q4Scoring = useD2Q4Scoring && bitsPerDim == 2;
        if (this.useD2Q4Scoring) {
            // Quantize queryTransformed to 4-bit unsigned [0, 15] in striped bit-plane format
            float qMin = Float.MAX_VALUE, qMax = -Float.MAX_VALUE;
            for (int j = 0; j < nDims; j++) {
                qMin = Math.min(qMin, queryTransformed[j]);
                qMax = Math.max(qMax, queryTransformed[j]);
            }
            float range = qMax - qMin;
            float qScale = range > 0 ? 15.0f / range : 1.0f;
            this.qOffset = qMin;
            this.invQScale = range > 0 ? range / 15.0f : 0f;

            this.queryQuantized4Bit = new byte[4 * planeBytes];
            int unsignedSum = 0;
            for (int j = 0; j < nDims; j++) {
                int level = Math.round((queryTransformed[j] - qMin) * qScale);
                level = Math.max(0, Math.min(15, level));
                unsignedSum += level;
                int byteIdx = j >>> 3;
                int bitIdx = 7 - (j & 7);
                for (int p = 0; p < 4; p++) {
                    if (((level >> p) & 1) != 0) {
                        queryQuantized4Bit[p * planeBytes + byteIdx] |= (byte) (1 << bitIdx);
                    }
                }
            }
            this.constantCorrection = 1.5f * (unsignedSum * this.invQScale + this.qOffset * nDims);
        } else {
            this.queryQuantized4Bit = null;
            this.invQScale = 0;
            this.qOffset = 0;
            this.constantCorrection = 0;
        }
    }

    @Override
    public int resetPostingsScorer(PostingMetadata metadata) throws java.io.IOException {
        float score = metadata.documentCentroidScore();
        indexInput.seek(metadata.offset());
        float centroidToParentSqDist = Float.intBitsToFloat(indexInput.readInt());
        indexInput.readFloats(currentCentroid, 0, currentCentroid.length);
        vectors = indexInput.readVInt();
        docEncoding = indexInput.readByte();
        docBase = 0;
        slicePos = indexInput.getFilePointer();

        centroidDistance = switch (similarityFunction) {
            case EUCLIDEAN -> ((1 / score) - 1) - centroidToParentSqDist;
            case COSINE, DOT_PRODUCT -> 2 * score - 1;
            case MAXIMUM_INNER_PRODUCT -> score - 1;
        };

        // Use precomputed queryDotCentroid from PostingMetadata if available, otherwise compute
        if (Float.isNaN(metadata.queryDotCentroid()) == false) {
            currentQueryDotCentroid = metadata.queryDotCentroid();
        } else {
            currentQueryDotCentroid = ESVectorUtil.dotProduct(query, currentCentroid);
        }

        return vectors;
    }

    @Override
    public int visit(KnnCollector knnCollector) throws java.io.IOException {
        indexInput.seek(slicePos);
        int scoredDocs = 0;

        int limit = vectors - BULK_SIZE + 1;
        int i = 0;
        for (; i < limit; i += BULK_SIZE) {
            readDocIds(BULK_SIZE);
            int docsToBulkScore = docToBulkScore(BULK_SIZE);
            if (docsToBulkScore == 0) {
                // Skip the entire block: codes + scales + offsets
                indexInput.skipBytes((long) BULK_SIZE * packedCodeBytes + (long) BULK_SIZE * Short.BYTES * 3);
                continue;
            }
            // Read structure-of-arrays: all codes, then all scales, then all offsets
            indexInput.readBytes(bulkCodeBuf, 0, BULK_SIZE * packedCodeBytes);
            for (int j = 0; j < BULK_SIZE; j++) {
                bulkScalesF16[j] = indexInput.readShort();
            }
            for (int j = 0; j < BULK_SIZE; j++) {
                bulkOffsetsF16[j] = indexInput.readShort();
            }
            for (int j = 0; j < BULK_SIZE; j++) {
                bulkDocSums[j] = indexInput.readShort();
            }

            float maxScore = Float.NEGATIVE_INFINITY;
            if (bitsPerDim == 2) {
                scoreBulk2BitBlock(BULK_SIZE);
                for (int j = 0; j < BULK_SIZE; j++) {
                    if (docIdsScratch[j] != -1) {
                        scores[j] = convertScore(scores[j]);
                        if (scores[j] > maxScore) {
                            maxScore = scores[j];
                        }
                    }
                }
            } else {
                decodeBulkScalesOffsets(BULK_SIZE);
                for (int j = 0; j < BULK_SIZE; j++) {
                    if (docIdsScratch[j] != -1) {
                        float s = scoreVector(bulkCodeBuf, j * packedCodeBytes, bulkScales[j], bulkOffsets[j]);
                        scores[j] = convertScore(s);
                        if (scores[j] > maxScore) {
                            maxScore = scores[j];
                        }
                    }
                }
            }
            if (knnCollector.minCompetitiveSimilarity() < maxScore) {
                collectBulk(knnCollector, BULK_SIZE, docsToBulkScore, maxScore);
            }
            scoredDocs += docsToBulkScore;
        }
        // Tail
        if (i < vectors) {
            int tailSize = vectors - i;
            readDocIds(tailSize);
            int docsToBulkScore = docToBulkScore(tailSize);
            if (docsToBulkScore > 0) {
                // Read tail block in structure-of-arrays format
                indexInput.readBytes(bulkCodeBuf, 0, tailSize * packedCodeBytes);
                for (int j = 0; j < tailSize; j++) {
                    bulkScalesF16[j] = indexInput.readShort();
                }
                for (int j = 0; j < tailSize; j++) {
                    bulkOffsetsF16[j] = indexInput.readShort();
                }
                for (int j = 0; j < tailSize; j++) {
                    bulkDocSums[j] = indexInput.readShort();
                }

                float maxScore = Float.NEGATIVE_INFINITY;
                if (bitsPerDim == 2) {
                    scoreBulk2BitBlock(tailSize);
                    for (int j = 0; j < tailSize; j++) {
                        if (docIdsScratch[j] != -1) {
                            scores[j] = convertScore(scores[j]);
                            if (scores[j] > maxScore) {
                                maxScore = scores[j];
                            }
                        }
                    }
                } else {
                    decodeBulkScalesOffsets(tailSize);
                    for (int j = 0; j < tailSize; j++) {
                        if (docIdsScratch[j] != -1) {
                            float s = scoreVector(bulkCodeBuf, j * packedCodeBytes, bulkScales[j], bulkOffsets[j]);
                            scores[j] = convertScore(s);
                            if (scores[j] > maxScore) {
                                maxScore = scores[j];
                            }
                        }
                    }
                }
                if (knnCollector.minCompetitiveSimilarity() < maxScore) {
                    collectBulk(knnCollector, tailSize, docsToBulkScore, maxScore);
                }
                scoredDocs += docsToBulkScore;
            } else {
                // Skip the tail block
                indexInput.skipBytes((long) tailSize * packedCodeBytes + (long) tailSize * Short.BYTES * 3);
            }
        }
        if (scoredDocs > 0) {
            knnCollector.incVisitedCount(scoredDocs);
        }
        return scoredDocs;
    }

    private float convertScore(float rawDotProduct) {
        return switch (similarityFunction) {
            case EUCLIDEAN -> 1 / (1 + rawDotProduct);
            case COSINE, DOT_PRODUCT -> (1 + rawDotProduct) / 2;
            case MAXIMUM_INNER_PRODUCT -> rawDotProduct >= 0 ? rawDotProduct + 1 : 1 / (1 - rawDotProduct);
        };
    }

    private float scoreVector(byte[] codeBuf, int codeOffset, float scale, float offset) {
        if (bitsPerDim == 1) {
            // For 1-bit, extract the plane into a temp buffer (1-bit is not the default path)
            byte[] planeBuf = new byte[planeBytes];
            System.arraycopy(codeBuf, codeOffset, planeBuf, 0, planeBytes);
            return AsymmetricHashingScorer.scoreOneVectorBinary(queryTransformed, currentQueryDotCentroid, planeBuf, nDims, scale, offset);
        } else if (bitsPerDim == 2) {
            return AsymmetricHashingScorer.scoreMultiBitSIMD2Planes(
                queryTransformedPadded,
                currentQueryDotCentroid,
                codeBuf,
                codeOffset,
                planeBytes,
                scale,
                offset,
                sumAllQt
            );
        } else {
            // General multi-bit fallback: extract codes into temp buffer
            byte[] tempCodes = new byte[packedCodeBytes];
            System.arraycopy(codeBuf, codeOffset, tempCodes, 0, packedCodeBytes);
            return AsymmetricHashingScorer.scoreOneVectorMultiBit(
                queryTransformed,
                currentQueryDotCentroid,
                tempCodes,
                nDims,
                bitsPerDim,
                scale,
                offset
            );
        }
    }

    /**
     * Scores a bulk block of 2-bit ASH vectors. Dispatches between:
     * - D2Q4 integer path (AND+popcount with 4-bit quantized query, O(1) correction via stored docSum)
     * - Float path (native NEON or Panama ipFloatBit)
     */
    private void scoreBulk2BitBlock(int count) {
        if (useD2Q4Scoring) {
            ESVectorUtil.ashScoreBulk2BitD2Q4(
                queryQuantized4Bit,
                bulkCodeBuf,
                packedCodeBytes,
                planeBytes,
                bulkScalesF16,
                bulkOffsetsF16,
                bulkDocSums,
                count,
                currentQueryDotCentroid,
                invQScale,
                qOffset,
                constantCorrection,
                scores
            );
        } else {
            ESVectorUtil.ashScoreBulk2Bit(
                queryTransformedPadded,
                bulkCodeBuf,
                bulkScalesF16,
                bulkOffsetsF16,
                packedCodeBytes,
                planeBytes,
                count,
                sumAllQt,
                currentQueryDotCentroid,
                scores
            );
        }
    }

    /** Decodes fp16 scales/offsets to float for the Java scoring path. */
    private void decodeBulkScalesOffsets(int count) {
        for (int j = 0; j < count; j++) {
            bulkScales[j] = Float.float16ToFloat(bulkScalesF16[j]);
            bulkOffsets[j] = Float.float16ToFloat(bulkOffsetsF16[j]);
        }
    }

    private void readDocIds(int count) throws java.io.IOException {
        idsWriter.readInts(indexInput, count, docEncoding, docIdsScratch);
        for (int j = 0; j < count; j++) {
            docBase += docIdsScratch[j];
            docIdsScratch[j] = docBase;
        }
    }

    private int docToBulkScore(int bulkSize) {
        if (acceptDocs == null) {
            return bulkSize;
        }
        int docToScore = 0;
        for (int ii = 0; ii < bulkSize; ii++) {
            if (docIdsScratch[ii] == -1 || acceptDocs.get(docIdsScratch[ii]) == false) {
                docIdsScratch[ii] = -1;
            } else {
                offsetsScratch[docToScore] = ii;
                docToScore++;
            }
        }
        return docToScore;
    }

    private void collectBulk(KnnCollector knnCollector, int bulkSize, int docsToBulkScore, float maxScore) {
        if (knnCollector instanceof BulkKnnCollector bulkCollector) {
            if (docsToBulkScore == bulkSize) {
                bulkCollector.bulkCollect(docIdsScratch, scores, bulkSize, maxScore);
                return;
            }
            for (int ii = 0; ii < docsToBulkScore; ii++) {
                int offset = offsetsScratch[ii];
                docIdsScratch[ii] = docIdsScratch[offset];
                scores[ii] = scores[offset];
            }
            bulkCollector.bulkCollect(docIdsScratch, scores, docsToBulkScore, maxScore);
            return;
        }
        for (int ii = 0; ii < bulkSize; ii++) {
            final int doc = docIdsScratch[ii];
            if (doc != -1) {
                knnCollector.collect(doc, scores[ii]);
            }
        }
    }
}
