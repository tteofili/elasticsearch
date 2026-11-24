/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq.next;

import org.apache.lucene.codecs.DocValuesConsumer;
import org.apache.lucene.codecs.DocValuesProducer;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.OrdinalMap;
import org.apache.lucene.index.SortedDocValues;

import java.io.IOException;

/**
 * Gives the {@code ESNext} IVF writers access to the protected merge machinery of {@link DocValuesConsumer}, which is
 * what builds a merged view of the sorted doc values of a field (the slice field of sliced segments) across the
 * segments being merged. Nothing is ever written through it.
 */
final class DocValueConsumerHelper extends DocValuesConsumer {

    static final DocValueConsumerHelper INSTANCE = new DocValueConsumerHelper();

    private DocValueConsumerHelper() {}

    public SortedDocValues getMergeSortedField(FieldInfo fieldInfo, final MergeState mergeState) throws IOException {
        // This is the magic to get a merged view from the segments.
        final OrdinalMap map = createOrdinalMapForSortedDV(fieldInfo, mergeState);
        return getMergedSortedSetDocValues(fieldInfo, mergeState, map);
    }

    @Override
    public void addNumericField(FieldInfo field, DocValuesProducer valuesProducer) {
        throw new AssertionError("Method should not be called");
    }

    @Override
    public void addBinaryField(FieldInfo field, DocValuesProducer valuesProducer) {
        throw new AssertionError("Method should not be called");
    }

    @Override
    public void addSortedField(FieldInfo field, DocValuesProducer valuesProducer) {
        throw new AssertionError("Method should not be called");
    }

    @Override
    public void addSortedNumericField(FieldInfo field, DocValuesProducer valuesProducer) {
        throw new AssertionError("Method should not be called");
    }

    @Override
    public void addSortedSetField(FieldInfo field, DocValuesProducer valuesProducer) {
        throw new AssertionError("Method should not be called");
    }

    @Override
    public void close() {
        throw new AssertionError("Method should not be called");
    }
}
