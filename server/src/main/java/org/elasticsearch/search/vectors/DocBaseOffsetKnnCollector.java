/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.search.vectors;

import org.apache.lucene.search.KnnCollector;

/**
 * A {@link KnnCollector} wrapper that adds a doc base offset to every collected doc ID.
 * Used by global best-first multi-segment IVF search so that segment-local doc IDs
 * are converted to global index space before being passed to the delegate collector.
 */
public final class DocBaseOffsetKnnCollector extends KnnCollector.Decorator {

    private final KnnCollector delegate;
    private final int docBase;

    public DocBaseOffsetKnnCollector(KnnCollector delegate, int docBase) {
        super(delegate);
        this.delegate = delegate;
        this.docBase = docBase;
    }

    @Override
    public boolean collect(int docId, float similarity) {
        return delegate.collect(docId + docBase, similarity);
    }
}
