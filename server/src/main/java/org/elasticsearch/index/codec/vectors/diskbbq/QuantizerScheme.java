/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq;

import org.apache.lucene.codecs.KnnVectorsFormat;
import org.elasticsearch.core.Nullable;
import org.elasticsearch.index.mapper.vectors.DenseVectorFieldMapper;

import java.util.concurrent.ExecutorService;

/**
 * One way of quantizing the vectors of an IVF segment (for example OSQ or ASH). A scheme is what a field's mapping
 * selects with {@code quantization_type}; everything that differs between quantizers is reached through it, so that
 * code which only needs "the format for this field" does not branch on the quantizer.
 * <p>
 * Schemes are looked up in {@code QuantizerSchemes}.
 */
public interface QuantizerScheme {

    /** Stable identifier of the scheme, which segments can persist to say how their vectors were quantized. */
    int id();

    /** The name used for the scheme in mappings, as the value of {@code quantization_type}. */
    String name();

    /**
     * Creates the vectors format that writes and reads fields quantized with this scheme.
     *
     * @param options the field's {@code bbq_disk} index options
     * @param elementType the element type of the field
     * @param mergingExecutorService executor for parallel merges, or {@code null}
     * @param maxMergingWorkers how many merge workers may be used
     * @param quantizerExecutorService executor for quantizer training, or {@code null}
     * @param sliceField the field that slices the segment, or {@code null} when the index is not sliced
     */
    KnnVectorsFormat newFormat(
        DenseVectorFieldMapper.BBQIVFIndexOptions options,
        DenseVectorFieldMapper.ElementType elementType,
        @Nullable ExecutorService mergingExecutorService,
        int maxMergingWorkers,
        @Nullable ExecutorService quantizerExecutorService,
        @Nullable String sliceField
    );
}
