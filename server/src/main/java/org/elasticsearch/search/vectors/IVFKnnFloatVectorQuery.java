/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */
package org.elasticsearch.search.vectors;

import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TopDocs;
import org.elasticsearch.common.lucene.Lucene;
import org.elasticsearch.index.codec.vectors.diskbbq.Preconditioner;
import org.elasticsearch.index.codec.vectors.diskbbq.VectorPreconditioner;

import java.io.IOException;
import java.util.Arrays;

/** A {@link IVFKnnFloatVectorQuery} that uses the IVF search strategy. */
public class IVFKnnFloatVectorQuery extends AbstractIVFKnnVectorQuery {

    private boolean isQueryPreconditioned = false;
    private float[] query;
    /**
     * Always the original, caller-supplied (un-preconditioned) query vector. Used for segments that
     * have no stored preconditioner (e.g., flush segments) when a different segment in the same search
     * has been preconditioned. Keeping this reference avoids using a preconditioned query against data
     * that was indexed in the original coordinate space.
     */
    private final float[] originalQuery;
    private final VectorSimilarityFunction vectorSimilarityFunction;

    /**
     * Creates a new {@link IVFKnnFloatVectorQuery} with the given parameters.
     * @param field the field to search
     * @param query the query vector
     * @param k the number of nearest neighbors to return
     * @param numCands the number of nearest neighbors to gather per shard
     * @param filter the filter to apply to the results
     * @param visitRatio the ratio of vectors to score for the IVF search strategy
     */
    public IVFKnnFloatVectorQuery(
        String field,
        float[] query,
        int k,
        int numCands,
        Query filter,
        float visitRatio,
        boolean doPrecondition
    ) {
        super(field, visitRatio, k, numCands, filter, doPrecondition);
        this.query = query;
        this.originalQuery = query;
        this.vectorSimilarityFunction = null;
    }

    public IVFKnnFloatVectorQuery(
        String field,
        float[] query,
        int k,
        int numCands,
        Query filter,
        float visitRatio,
        boolean doPrecondition,
        boolean useCalibrationOversample
    ) {
        super(field, visitRatio, k, numCands, filter, doPrecondition, useCalibrationOversample);
        this.query = query;
        this.originalQuery = query;
        this.vectorSimilarityFunction = null;
    }

    public IVFKnnFloatVectorQuery(
        String field,
        float[] query,
        int k,
        int numCands,
        Query filter,
        float visitRatio,
        boolean doPrecondition,
        boolean useCalibrationOversample,
        VectorSimilarityFunction vectorSimilarityFunction
    ) {
        super(field, visitRatio, k, numCands, filter, doPrecondition, useCalibrationOversample);
        this.query = query;
        this.originalQuery = query;
        this.vectorSimilarityFunction = vectorSimilarityFunction;
    }

    public float[] getQuery() {
        return query;
    }

    @Override
    public String toString(String field) {
        StringBuilder buffer = new StringBuilder();
        buffer.append(getClass().getSimpleName())
            .append(":")
            .append(this.field)
            .append("[")
            .append(query[0])
            .append(",...]")
            .append("[")
            .append(k)
            .append("]");
        if (this.filter != null) {
            buffer.append("[").append(this.filter).append("]");
        }
        return buffer.toString();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (super.equals(o) == false) return false;
        IVFKnnFloatVectorQuery that = (IVFKnnFloatVectorQuery) o;
        return Arrays.equals(query, that.query);
    }

    @Override
    public int hashCode() {
        int result = super.hashCode();
        result = 31 * result + Arrays.hashCode(query);
        return result;
    }

    @Override
    protected void preconditionQuery(LeafReaderContext context) throws IOException {
        if (isQueryPreconditioned) {
            // already preconditioned
            return;
        }
        LeafReader reader = context.reader();
        SegmentReader segmentReader = Lucene.tryUnwrapSegmentReader(reader);
        if (segmentReader == null) {
            // ignore and continue to the next leaf context to see if we can get a segment reader there
            return;
        }
        KnnVectorsReader fieldsReader = segmentReader.getVectorReader();
        if (fieldsReader instanceof PerFieldKnnVectorsFormat.FieldsReader) {
            KnnVectorsReader knnVectorsReader = ((PerFieldKnnVectorsFormat.FieldsReader) fieldsReader).getFieldReader(field);
            if (knnVectorsReader instanceof VectorPreconditioner) {
                FieldInfo fieldInfo = segmentReader.getFieldInfos().fieldInfo(field);
                Preconditioner preconditioner = ((VectorPreconditioner) knnVectorsReader).getPreconditioner(fieldInfo);
                if (preconditioner != null) {
                    final float[] out = new float[originalQuery.length];
                    preconditioner.applyTransform(originalQuery, out);
                    // have to keep the copy to avoid issues with reused arrays by the caller of IVFKnnFloatVectorQuery which expects
                    // a non-preconditioned query vector to still exist
                    query = out;
                    isQueryPreconditioned = true;
                }
            }
        }
    }

    /**
     * Returns the effective query vector for the given segment: the preconditioned vector when
     * the segment has a stored preconditioner (i.e., its data is in preconditioned space), or the
     * original un-preconditioned vector otherwise (e.g., flush segments that were indexed before
     * calibration decided to use preconditioning).
     */
    private float[] resolveEffectiveQuery(LeafReaderContext context) throws IOException {
        if (isQueryPreconditioned == false) {
            return originalQuery;
        }
        SegmentReader segmentReader = Lucene.tryUnwrapSegmentReader(context.reader());
        if (segmentReader != null) {
            KnnVectorsReader fieldsReader = segmentReader.getVectorReader();
            if (fieldsReader instanceof PerFieldKnnVectorsFormat.FieldsReader perFieldReader) {
                KnnVectorsReader knnVectorsReader = perFieldReader.getFieldReader(field);
                if (knnVectorsReader instanceof VectorPreconditioner vp) {
                    FieldInfo fieldInfo = segmentReader.getFieldInfos().fieldInfo(field);
                    if (fieldInfo != null && vp.getPreconditioner(fieldInfo) != null) {
                        // segment data is in preconditioned space → use preconditioned query
                        return query;
                    }
                }
            }
        }
        // segment has no preconditioner → data is in original space
        return originalQuery;
    }

    @Override
    protected TopDocs approximateSearch(
        LeafReaderContext context,
        AcceptDocs acceptDocs,
        int visitedLimit,
        IVFCollectorManager knnCollectorManager,
        float visitRatio
    ) throws IOException {
        LeafReader reader = context.reader();
        IVFKnnSearchStrategy strategy = new IVFKnnSearchStrategy(visitRatio, numCands, k, knnCollectorManager.longAccumulator);
        AbstractMaxScoreKnnCollector knnCollector = knnCollectorManager.newCollector(visitedLimit, strategy, context);
        if (knnCollector == null) {
            return NO_RESULTS;
        }
        strategy.setCollector(knnCollector);
        reader.searchNearestVectors(field, resolveEffectiveQuery(context), knnCollector, acceptDocs);
        TopDocs results = knnCollector.topDocs();
        return results != null ? results : NO_RESULTS;
    }

    @Override
    protected Query getAutoRescoreQuery(IndexSearcher indexSearcher, TopDocs topOversampled, int effectiveK) {
        if (vectorSimilarityFunction == null) {
            return null;
        }
        Query topDocsQuery = new KnnScoreDocQuery(topOversampled.scoreDocs, indexSearcher.getIndexReader());
        return RescoreKnnVectorQuery.fromInnerQuery(field, originalQuery, vectorSimilarityFunction, k, effectiveK, topDocsQuery);
    }
}
