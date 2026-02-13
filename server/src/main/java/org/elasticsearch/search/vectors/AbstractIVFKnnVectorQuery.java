/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.search.vectors;

import com.carrotsearch.hppc.IntHashSet;

import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.FieldExistsQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.MatchNoDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.QueryVisitor;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.ScorerSupplier;
import org.apache.lucene.search.TaskExecutor;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopDocsCollector;
import org.apache.lucene.search.Weight;
import org.apache.lucene.search.knn.KnnCollectorManager;
import org.apache.lucene.search.knn.KnnSearchStrategy;
import org.apache.lucene.util.Bits;
import org.elasticsearch.common.lucene.Lucene;
import org.elasticsearch.common.lucene.search.Queries;
import org.elasticsearch.index.codec.vectors.diskbbq.IVFVectorsReader;
import org.elasticsearch.index.codec.vectors.diskbbq.PostingMetadata;
import org.elasticsearch.search.profile.query.QueryProfiler;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.Callable;
import java.util.concurrent.atomic.LongAccumulator;

import static org.elasticsearch.search.vectors.AbstractMaxScoreKnnCollector.LEAST_COMPETITIVE;

abstract class AbstractIVFKnnVectorQuery extends Query implements QueryProfilerProvider {

    static final TopDocs NO_RESULTS = TopDocsCollector.EMPTY_TOPDOCS;

    /** Segments with at most this many docs are batched together for fewer tasks. */
    private static final int MIN_DOCS_TINY_SEGMENT = 10_000;
    /** Target minimum docs per batch when batching tiny segments. */
    private static final int MIN_DOCS_PER_BATCH = 10_000;
    /** Max posting lists per chunk when splitting a large segment. */
    private static final int MAX_POSTING_LISTS_PER_CHUNK = 500;
    /** Only split a segment when it has at least this many posting lists. */
    private static final int MIN_POSTING_LISTS_TO_SPLIT = 1000;

    protected final String field;
    protected final float providedVisitRatio;
    protected final int k;
    protected final int numCands;
    protected final Query filter;
    protected int vectorOpsCount;
    protected boolean doPrecondition;

    protected AbstractIVFKnnVectorQuery(String field, float visitRatio, int k, int numCands, Query filter, boolean doPrecondition) {
        if (k < 1) {
            throw new IllegalArgumentException("k must be at least 1, got: " + k);
        }
        if (visitRatio < 0.0f || visitRatio > 1.0f) {
            throw new IllegalArgumentException("visitRatio must be between 0.0 and 1.0 (both inclusive), got: " + visitRatio);
        }
        if (numCands < k) {
            throw new IllegalArgumentException("numCands must be at least k, got: " + numCands);
        }
        this.field = field;
        this.providedVisitRatio = visitRatio;
        this.k = k;
        this.filter = filter;
        this.numCands = numCands;
        this.doPrecondition = doPrecondition;
    }

    @Override
    public void visit(QueryVisitor visitor) {
        if (visitor.acceptField(field)) {
            visitor.visitLeaf(this);
        }
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        AbstractIVFKnnVectorQuery that = (AbstractIVFKnnVectorQuery) o;
        return k == that.k
            && Objects.equals(field, that.field)
            && Objects.equals(filter, that.filter)
            && Objects.equals(providedVisitRatio, that.providedVisitRatio);
    }

    @Override
    public int hashCode() {
        return Objects.hash(field, k, filter, providedVisitRatio);
    }

    @Override
    public Query rewrite(IndexSearcher indexSearcher) throws IOException {
        vectorOpsCount = 0;
        IndexReader reader = indexSearcher.getIndexReader();

        final Weight filterWeight;
        if (filter != null) {
            BooleanQuery booleanQuery = new BooleanQuery.Builder().add(filter, BooleanClause.Occur.FILTER)
                .add(new FieldExistsQuery(field), BooleanClause.Occur.FILTER)
                .build();
            Query rewritten = indexSearcher.rewrite(booleanQuery);
            if (rewritten.getClass() == MatchNoDocsQuery.class) {
                return rewritten;
            }
            filterWeight = indexSearcher.createWeight(rewritten, ScoreMode.COMPLETE_NO_SCORES, 1f);
        } else {
            filterWeight = null;
        }

        // we request numCands as we are using it as an approximation measure
        // we need to ensure we are getting at least 2*k results to ensure we cover overspill duplicates
        // TODO move the logic for automatically adjusting percentages to the query, so we can only pass
        // 2k to the collector.
        IVFCollectorManager knnCollectorManager = getKnnCollectorManager(Math.round(2f * k), indexSearcher);
        TaskExecutor taskExecutor = indexSearcher.getTaskExecutor();
        List<LeafReaderContext> leafReaderContexts = reader.leaves();

        assert this instanceof IVFKnnFloatVectorQuery;
        int totalVectors = 0;
        for (LeafReaderContext leafReaderContext : leafReaderContexts) {
            LeafReader leafReader = leafReaderContext.reader();
            FloatVectorValues floatVectorValues = leafReader.getFloatVectorValues(field);
            if (floatVectorValues != null) {
                totalVectors += floatVectorValues.size();
            }
        }

        final float visitRatio;
        if (providedVisitRatio == 0.0f) {
            // dynamically set the percentage
            float expected = (float) Math.round(
                Math.log10(totalVectors) * Math.log10(totalVectors) * (Math.min(10_000, Math.max(numCands, 5 * k)))
            );
            visitRatio = expected / totalVectors;
        } else {
            visitRatio = providedVisitRatio;
        }

        if (leafReaderContexts.isEmpty()) {
            return Queries.NO_DOCS_INSTANCE;
        }

        // Classify segments: tiny (batch together) vs large (may split by posting lists)
        List<LeafReaderContext> tinySegments = new ArrayList<>();
        List<LeafReaderContext> largeSegments = new ArrayList<>();
        for (LeafReaderContext ctx : leafReaderContexts) {
            if (ctx.reader().numDocs() <= MIN_DOCS_TINY_SEGMENT) {
                tinySegments.add(ctx);
            } else {
                largeSegments.add(ctx);
            }
        }

        // Batch tiny segments so each task has at least MIN_DOCS_PER_BATCH docs
        List<List<LeafReaderContext>> tinyBatches = new ArrayList<>();
        List<LeafReaderContext> currentBatch = new ArrayList<>();
        int currentBatchDocs = 0;
        for (LeafReaderContext ctx : tinySegments) {
            currentBatch.add(ctx);
            currentBatchDocs += ctx.reader().numDocs();
            if (currentBatchDocs >= MIN_DOCS_PER_BATCH) {
                tinyBatches.add(new ArrayList<>(currentBatch));
                currentBatch.clear();
                currentBatchDocs = 0;
            }
        }
        if (currentBatch.isEmpty() == false) {
            tinyBatches.add(currentBatch);
        }

        List<Callable<TopDocs>> tasks = new ArrayList<>();

        // Tasks for batched tiny segments
        for (List<LeafReaderContext> batch : tinyBatches) {
            tasks.add(() -> {
                TopDocs[] batchResults = new TopDocs[batch.size()];
                for (int i = 0; i < batch.size(); i++) {
                    LeafReaderContext ctx = batch.get(i);
                    if (doPrecondition) {
                        preconditionQuery(ctx);
                    }
                    batchResults[i] = searchLeaf(ctx, filterWeight, knnCollectorManager, visitRatio);
                }
                return batchResults.length == 1 ? batchResults[0] : TopDocs.merge(k, batchResults);
            });
        }

        // Large segments: one task per segment, or split by posting-list chunks when supported
        // Collect segments that qualify for chunking (IVF reader + acceptDocs), then run phase 1 in parallel when multiple
        List<LeafReaderContext> phase1Contexts = new ArrayList<>();
        List<IVFVectorsReader> phase1Readers = new ArrayList<>();
        List<AcceptDocs> phase1AcceptDocs = new ArrayList<>();
        for (LeafReaderContext ctx : largeSegments) {
            if (doPrecondition) {
                preconditionQuery(ctx);
            }
            IVFVectorsReader ivfReader = getIVFReader(ctx, field);
            if (ivfReader == null) {
                tasks.add(() -> searchLeaf(ctx, filterWeight, knnCollectorManager, visitRatio));
                continue;
            }
            AcceptDocs acceptDocs = getAcceptDocs(ctx, filterWeight);
            if (acceptDocs == null) {
                tasks.add(() -> searchLeaf(ctx, filterWeight, knnCollectorManager, visitRatio));
                continue;
            }
            phase1Contexts.add(ctx);
            phase1Readers.add(ivfReader);
            phase1AcceptDocs.add(acceptDocs);
        }

        float[] queryVector = getQueryVector();
        if (phase1Contexts.size() == 1) {
            // Single segment: run phase 1 inline
            LeafReaderContext ctx = phase1Contexts.get(0);
            IVFVectorsReader ivfReader = phase1Readers.get(0);
            AcceptDocs acceptDocs = phase1AcceptDocs.get(0);
            List<PostingMetadata> orderedList;
            try {
                orderedList = ivfReader.getOrderedPostingMetadataList(field, queryVector, acceptDocs, visitRatio);
            } catch (IOException e) {
                orderedList = List.of();
            }
            addChunkOrSearchLeafTasks(ctx, orderedList, tasks, filterWeight, knnCollectorManager, visitRatio);
        } else if (phase1Contexts.size() > 1) {
            // Multiple large segments: run phase 1 in parallel
            List<Callable<List<PostingMetadata>>> phase1Tasks = new ArrayList<>(phase1Contexts.size());
            for (int i = 0; i < phase1Contexts.size(); i++) {
                IVFVectorsReader ivfReader = phase1Readers.get(i);
                AcceptDocs acceptDocs = phase1AcceptDocs.get(i);
                phase1Tasks.add(() -> {
                    try {
                        return ivfReader.getOrderedPostingMetadataList(field, queryVector, acceptDocs, visitRatio);
                    } catch (IOException e) {
                        return List.<PostingMetadata>of();
                    }
                });
            }
            List<List<PostingMetadata>> phase1Results = taskExecutor.invokeAll(phase1Tasks);
            for (int i = 0; i < phase1Contexts.size(); i++) {
                LeafReaderContext ctx = phase1Contexts.get(i);
                List<PostingMetadata> orderedList = phase1Results.get(i);
                addChunkOrSearchLeafTasks(ctx, orderedList, tasks, filterWeight, knnCollectorManager, visitRatio);
            }
        }

        TopDocs[] perLeafResults = taskExecutor.invokeAll(tasks).toArray(TopDocs[]::new);

        // Merge sort the results
        TopDocs topK = TopDocs.merge(k, perLeafResults);
        vectorOpsCount = (int) topK.totalHits.value();
        if (topK.scoreDocs.length == 0) {
            return Queries.NO_DOCS_INSTANCE;
        }
        // Single task cannot have cross-task duplicate doc IDs; skip final dedup
        if (tasks.size() == 1) {
            return new KnnScoreDocQuery(topK.scoreDocs, reader);
        }
        // Final deduplication by doc ID (merge does not remove duplicates; same doc can appear from multiple chunks)
        IntHashSet seenDocIds = new IntHashSet(topK.scoreDocs.length * 4 / 3);
        List<ScoreDoc> deduplicated = new ArrayList<>(topK.scoreDocs.length);
        for (ScoreDoc scoreDoc : topK.scoreDocs) {
            if (seenDocIds.add(scoreDoc.doc)) {
                deduplicated.add(scoreDoc);
            }
        }
        return new KnnScoreDocQuery(deduplicated.toArray(new ScoreDoc[0]), reader);
    }

    private void addChunkOrSearchLeafTasks(
        LeafReaderContext ctx,
        List<PostingMetadata> orderedList,
        List<Callable<TopDocs>> tasks,
        Weight filterWeight,
        IVFCollectorManager knnCollectorManager,
        float visitRatio
    ) {
        if (orderedList == null || orderedList.size() < MIN_POSTING_LISTS_TO_SPLIT) {
            tasks.add(() -> searchLeaf(ctx, filterWeight, knnCollectorManager, visitRatio));
            return;
        }
        for (int start = 0; start < orderedList.size(); start += MAX_POSTING_LISTS_PER_CHUNK) {
            int end = Math.min(start + MAX_POSTING_LISTS_PER_CHUNK, orderedList.size());
            List<PostingMetadata> chunk = new ArrayList<>(orderedList.subList(start, end));
            LeafReaderContext ctxFinal = ctx;
            tasks.add(() -> searchChunk(ctxFinal, filterWeight, knnCollectorManager, visitRatio, chunk));
        }
    }

    private TopDocs searchLeaf(LeafReaderContext ctx, Weight filterWeight, IVFCollectorManager knnCollectorManager, float visitRatio)
        throws IOException {
        TopDocs results = getLeafResults(ctx, filterWeight, knnCollectorManager, visitRatio);
        IntHashSet dedup = new IntHashSet(results.scoreDocs.length * 4 / 3);
        int deduplicateCount = 0;
        for (ScoreDoc scoreDoc : results.scoreDocs) {
            if (dedup.add(scoreDoc.doc)) {
                deduplicateCount++;
            }
        }
        ScoreDoc[] deduplicatedScoreDocs = new ScoreDoc[deduplicateCount];
        dedup.clear();
        int index = 0;
        for (ScoreDoc scoreDoc : results.scoreDocs) {
            if (dedup.add(scoreDoc.doc)) {
                scoreDoc.doc += ctx.docBase;
                deduplicatedScoreDocs[index++] = scoreDoc;
            }
        }
        return new TopDocs(results.totalHits, deduplicatedScoreDocs);
    }

    TopDocs getLeafResults(LeafReaderContext ctx, Weight filterWeight, IVFCollectorManager knnCollectorManager, float visitRatio)
        throws IOException {
        final LeafReader reader = ctx.reader();
        final Bits liveDocs = reader.getLiveDocs();
        final int maxDoc = reader.maxDoc();

        if (filterWeight == null) {
            return approximateSearch(
                ctx,
                liveDocs == null ? ESAcceptDocs.ESAcceptDocsAll.INSTANCE : new ESAcceptDocs.BitsAcceptDocs(liveDocs, maxDoc),
                Integer.MAX_VALUE,
                knnCollectorManager,
                visitRatio
            );
        }

        ScorerSupplier supplier = filterWeight.scorerSupplier(ctx);
        if (supplier == null) {
            return TopDocsCollector.EMPTY_TOPDOCS;
        }

        return approximateSearch(
            ctx,
            new ESAcceptDocs.ScorerSupplierAcceptDocs(supplier, liveDocs, maxDoc),
            Integer.MAX_VALUE,
            knnCollectorManager,
            visitRatio
        );
    }

    /** Returns the query vector for chunked IVF search. */
    protected abstract float[] getQueryVector();

    abstract void preconditionQuery(LeafReaderContext context) throws IOException;

    abstract TopDocs approximateSearch(
        LeafReaderContext context,
        AcceptDocs acceptDocs,
        int visitedLimit,
        IVFCollectorManager knnCollectorManager,
        float visitRatio
    ) throws IOException;

    protected IVFCollectorManager getKnnCollectorManager(int k, IndexSearcher searcher) {
        return new IVFCollectorManager(k, searcher);
    }

    /**
     * Returns the IVF reader for the segment's field, or null if not IVF or not a segment reader.
     */
    private static IVFVectorsReader getIVFReader(LeafReaderContext ctx, String field) {
        LeafReader reader = ctx.reader();
        SegmentReader segmentReader = Lucene.tryUnwrapSegmentReader(reader);
        if (segmentReader == null) {
            return null;
        }
        KnnVectorsReader fieldsReader = segmentReader.getVectorReader();
        if (fieldsReader instanceof PerFieldKnnVectorsFormat.FieldsReader perField) {
            KnnVectorsReader fieldReader = perField.getFieldReader(field);
            if (fieldReader instanceof IVFVectorsReader ivfReader) {
                return ivfReader;
            }
        }
        return null;
    }

    /**
     * Runs IVF search over a single chunk of posting lists (one segment). Used when splitting a
     * large segment across threads. Caller must not share the reader's IndexInputs across threads.
     */
    private TopDocs searchChunk(
        LeafReaderContext ctx,
        Weight filterWeight,
        IVFCollectorManager knnCollectorManager,
        float visitRatio,
        List<PostingMetadata> postingsChunk
    ) throws IOException {
        AcceptDocs acceptDocs = getAcceptDocs(ctx, filterWeight);
        if (acceptDocs == null) {
            return TopDocsCollector.EMPTY_TOPDOCS;
        }
        IVFVectorsReader ivfReader = getIVFReader(ctx, field);
        if (ivfReader == null) {
            return TopDocsCollector.EMPTY_TOPDOCS;
        }
        IVFKnnSearchStrategy strategy = new IVFKnnSearchStrategy(visitRatio, knnCollectorManager.longAccumulator);
        AbstractMaxScoreKnnCollector knnCollector = knnCollectorManager.newCollector(
            Integer.MAX_VALUE,
            strategy,
            ctx
        );
        if (knnCollector == null) {
            return TopDocsCollector.EMPTY_TOPDOCS;
        }
        strategy.setCollector(knnCollector);
        ivfReader.searchPostingListChunk(field, getQueryVector(), acceptDocs, postingsChunk, knnCollector);
        TopDocs results = knnCollector.topDocs();
        if (results == null || results.scoreDocs.length == 0) {
            return results != null ? results : TopDocsCollector.EMPTY_TOPDOCS;
        }
        IntHashSet dedup = new IntHashSet(results.scoreDocs.length * 4 / 3);
        int deduplicateCount = 0;
        for (ScoreDoc scoreDoc : results.scoreDocs) {
            if (dedup.add(scoreDoc.doc)) {
                deduplicateCount++;
            }
        }
        ScoreDoc[] deduplicatedScoreDocs = new ScoreDoc[deduplicateCount];
        dedup.clear();
        int index = 0;
        for (ScoreDoc scoreDoc : results.scoreDocs) {
            if (dedup.add(scoreDoc.doc)) {
                scoreDoc.doc += ctx.docBase;
                deduplicatedScoreDocs[index++] = scoreDoc;
            }
        }
        return new TopDocs(results.totalHits, deduplicatedScoreDocs);
    }

    private static AcceptDocs getAcceptDocs(LeafReaderContext ctx, Weight filterWeight) throws IOException {
        LeafReader reader = ctx.reader();
        Bits liveDocs = reader.getLiveDocs();
        int maxDoc = reader.maxDoc();
        if (filterWeight == null) {
            return liveDocs == null ? ESAcceptDocs.ESAcceptDocsAll.INSTANCE : new ESAcceptDocs.BitsAcceptDocs(liveDocs, maxDoc);
        }
        ScorerSupplier supplier = filterWeight.scorerSupplier(ctx);
        if (supplier == null) {
            return null;
        }
        return new ESAcceptDocs.ScorerSupplierAcceptDocs(supplier, liveDocs, maxDoc);
    }

    @Override
    public final void profile(QueryProfiler queryProfiler) {
        queryProfiler.addVectorOpsCount(vectorOpsCount);
    }

    static class IVFCollectorManager implements KnnCollectorManager {
        private final int k;
        final LongAccumulator longAccumulator;

        IVFCollectorManager(int k, IndexSearcher searcher) {
            this.k = k;
            longAccumulator = searcher.getIndexReader().leaves().size() > 1 ? new LongAccumulator(Long::max, LEAST_COMPETITIVE) : null;
        }

        @Override
        public AbstractMaxScoreKnnCollector newCollector(int visitedLimit, KnnSearchStrategy searchStrategy, LeafReaderContext context)
            throws IOException {
            return new MaxScoreTopKnnCollector(k, visitedLimit, searchStrategy);
        }
    }
}
