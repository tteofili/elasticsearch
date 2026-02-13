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
import org.apache.lucene.search.KnnCollector;
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
import org.elasticsearch.index.codec.vectors.diskbbq.CentroidIterator;
import org.elasticsearch.index.codec.vectors.diskbbq.IVFVectorsReader;
import org.elasticsearch.index.codec.vectors.diskbbq.PostingMetadata;
import org.elasticsearch.search.profile.query.QueryProfiler;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;
import java.util.PriorityQueue;
import java.util.concurrent.Callable;
import java.util.concurrent.atomic.LongAccumulator;

import static org.elasticsearch.search.vectors.AbstractMaxScoreKnnCollector.LEAST_COMPETITIVE;

abstract class AbstractIVFKnnVectorQuery extends Query implements QueryProfilerProvider {

    static final TopDocs NO_RESULTS = TopDocsCollector.EMPTY_TOPDOCS;

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

        // Global best-first: when multiple segments and all support IVF, visit clusters in global score order
        if (leafReaderContexts.size() > 1 && knnCollectorManager.getClass() == IVFCollectorManager.class) {
            if (doPrecondition) {
                for (LeafReaderContext context : leafReaderContexts) {
                    preconditionQuery(context);
                }
            }
            TopDocs globalTopDocs = tryGlobalBestFirstSearch(
                reader,
                leafReaderContexts,
                filterWeight,
                getQueryVector(),
                visitRatio
            );
            if (globalTopDocs != null) {
                vectorOpsCount = (int) globalTopDocs.totalHits.value();
                if (globalTopDocs.scoreDocs.length == 0) {
                    return Queries.NO_DOCS_INSTANCE;
                }
                return new KnnScoreDocQuery(globalTopDocs.scoreDocs, reader);
            }
        }

        List<Callable<TopDocs>> tasks = new ArrayList<>(leafReaderContexts.size());
        for (LeafReaderContext context : leafReaderContexts) {
            if (doPrecondition) {
                preconditionQuery(context);
            }
            tasks.add(() -> searchLeaf(context, filterWeight, knnCollectorManager, visitRatio));
        }
        TopDocs[] perLeafResults = taskExecutor.invokeAll(tasks).toArray(TopDocs[]::new);

        // Merge sort the results
        TopDocs topK = TopDocs.merge(k, perLeafResults);
        vectorOpsCount = (int) topK.totalHits.value();
        if (topK.scoreDocs.length == 0) {
            return Queries.NO_DOCS_INSTANCE;
        }
        return new KnnScoreDocQuery(topK.scoreDocs, reader);
    }

    /**
     * Runs global best-first multi-segment IVF: one max-heap over all segment centroids,
     * visit clusters in descending score order, stop when max unvisited score ≤ min(top-k).
     *
     * @return TopDocs with global doc IDs, or null to fall back to per-segment search
     */
    private TopDocs tryGlobalBestFirstSearch(
        IndexReader reader,
        List<LeafReaderContext> leafReaderContexts,
        Weight filterWeight,
        float[] queryVector,
        float visitRatio
    ) throws IOException {
        assert leafReaderContexts.size() > 1;
        List<SegmentWithIVF> segments = new ArrayList<>(leafReaderContexts.size());
        int totalVectors = 0;
        for (LeafReaderContext ctx : leafReaderContexts) {
            LeafReader leafReader = ctx.reader();
            FloatVectorValues values = leafReader.getFloatVectorValues(field);
            if (values == null || values.size() == 0) {
                continue;
            }
            totalVectors += values.size();
            AcceptDocs acceptDocs = acceptDocsFor(ctx, filterWeight);
            IVFVectorsReader ivfReader = getIVFReader(leafReader);
            if (ivfReader == null) {
                return null;
            }
            CentroidIterator it = ivfReader.getOrderedCentroidIterator(field, queryVector, acceptDocs);
            if (it == null) {
                return null;
            }
            segments.add(new SegmentWithIVF(ctx, ivfReader, acceptDocs, it));
        }
        if (segments.size() <= 1) {
            return null;
        }

        long maxVectorVisited = (long) (2.0 * visitRatio * totalVectors);

        // Max-heap by centroid score (descending)
        PriorityQueue<CentroidHeapEntry> heap = new PriorityQueue<>(
            segments.size(),
            Comparator.comparingDouble((CentroidHeapEntry e) -> e.score).reversed()
        );
        for (int s = 0; s < segments.size(); s++) {
            SegmentWithIVF seg = segments.get(s);
            if (seg.centroidIterator.hasNext()) {
                PostingMetadata meta = seg.centroidIterator.nextPosting();
                heap.add(new CentroidHeapEntry(meta.documentCentroidScore(), s, meta, seg));
            }
        }

        IVFKnnSearchStrategy strategy = new IVFKnnSearchStrategy(visitRatio, null);
        AbstractMaxScoreKnnCollector globalCollector = new MaxScoreTopKnnCollector(
            k,
            Integer.MAX_VALUE,
            strategy
        );
        strategy.setCollector(globalCollector);

        long vectorsVisited = 0;
        while (heap.isEmpty() == false) {
            if (vectorsVisited >= maxVectorVisited) {
                break;
            }
            CentroidHeapEntry entry = heap.poll();
            float score = entry.score;
            if (globalCollector.numCollected() >= k && score <= globalCollector.minCompetitiveSimilarity()) {
                break;
            }
            SegmentWithIVF seg = entry.segment;
            KnnCollector wrappingCollector = new DocBaseOffsetKnnCollector(globalCollector, seg.ctx.docBase);
            vectorsVisited += seg.ivfReader.visitSingleCluster(field, queryVector, entry.meta, wrappingCollector, seg.acceptDocs);
            if (seg.centroidIterator.hasNext()) {
                PostingMetadata nextMeta = seg.centroidIterator.nextPosting();
                heap.add(new CentroidHeapEntry(nextMeta.documentCentroidScore(), entry.segmentIndex, nextMeta, seg));
            }
        }

        TopDocs topDocs = globalCollector.topDocs();
        return topDocs != null ? topDocs : TopDocsCollector.EMPTY_TOPDOCS;
    }

    private AcceptDocs acceptDocsFor(LeafReaderContext ctx, Weight filterWeight) throws IOException {
        LeafReader reader = ctx.reader();
        Bits liveDocs = reader.getLiveDocs();
        int maxDoc = reader.maxDoc();
        if (filterWeight == null) {
            return liveDocs == null ? ESAcceptDocs.ESAcceptDocsAll.INSTANCE : new ESAcceptDocs.BitsAcceptDocs(liveDocs, maxDoc);
        }
        ScorerSupplier supplier = filterWeight.scorerSupplier(ctx);
        if (supplier == null) {
            return new ESAcceptDocs.BitsAcceptDocs(
                new Bits() {
                    @Override
                    public boolean get(int index) {
                        return false;
                    }

                    @Override
                    public int length() {
                        return maxDoc;
                    }
                },
                maxDoc
            );
        }
        return new ESAcceptDocs.ScorerSupplierAcceptDocs(supplier, liveDocs, maxDoc);
    }

    private IVFVectorsReader getIVFReader(LeafReader reader) {
        SegmentReader segmentReader = Lucene.tryUnwrapSegmentReader(reader);
        if (segmentReader == null) {
            return null;
        }
        KnnVectorsReader fieldsReader = segmentReader.getVectorReader();
        if (fieldsReader instanceof PerFieldKnnVectorsFormat.FieldsReader perField) {
            KnnVectorsReader fieldReader = perField.getFieldReader(field);
            return fieldReader instanceof IVFVectorsReader ivf ? ivf : null;
        }
        return null;
    }

    private static class SegmentWithIVF {
        final LeafReaderContext ctx;
        final IVFVectorsReader ivfReader;
        final AcceptDocs acceptDocs;
        final CentroidIterator centroidIterator;

        SegmentWithIVF(LeafReaderContext ctx, IVFVectorsReader ivfReader, AcceptDocs acceptDocs, CentroidIterator centroidIterator) {
            this.ctx = ctx;
            this.ivfReader = ivfReader;
            this.acceptDocs = acceptDocs;
            this.centroidIterator = centroidIterator;
        }
    }

    private static class CentroidHeapEntry {
        final float score;
        final int segmentIndex;
        final PostingMetadata meta;
        final SegmentWithIVF segment;

        CentroidHeapEntry(float score, int segmentIndex, PostingMetadata meta, SegmentWithIVF segment) {
            this.score = score;
            this.segmentIndex = segmentIndex;
            this.meta = meta;
            this.segment = segment;
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

    /**
     * Returns the query vector for global best-first multi-segment search.
     * Only used when all segments are IVF and we run the global path.
     */
    abstract float[] getQueryVector();

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
