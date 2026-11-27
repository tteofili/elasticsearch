/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq.next;

import org.apache.lucene.codecs.KnnVectorsFormat;
import org.elasticsearch.core.Nullable;
import org.elasticsearch.index.codec.vectors.ash.AshDistortionRateQuantizer;
import org.elasticsearch.index.codec.vectors.diskbbq.IvfAutoCalibration;
import org.elasticsearch.index.codec.vectors.diskbbq.IvfFlushConfigSource;
import org.elasticsearch.index.codec.vectors.diskbbq.IvfMergeConfigResolver;
import org.elasticsearch.index.codec.vectors.diskbbq.QuantEncoding;
import org.elasticsearch.index.codec.vectors.diskbbq.QuantizerScheme;
import org.elasticsearch.index.codec.vectors.diskbbq.calibrate.ClusteredSample;
import org.elasticsearch.index.codec.vectors.diskbbq.calibrate.DistortionRateQuantizer;
import org.elasticsearch.index.codec.vectors.diskbbq.calibrate.OsqDistortionRateQuantizer;
import org.elasticsearch.index.mapper.vectors.DenseVectorFieldMapper;
import org.elasticsearch.index.mapper.vectors.DenseVectorFieldMapper.BBQIVFIndexOptions;
import org.elasticsearch.index.mapper.vectors.DenseVectorFieldMapper.ElementType;

import java.io.IOException;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.ExecutorService;

/**
 * The {@link QuantizerScheme}s that the {@code ESNext} IVF formats support, and the only place that decides which
 * format class serves which scheme.
 */
public final class QuantizerSchemes {

    private QuantizerSchemes() {}

    /** Optimized scalar quantization, written by {@link ESNextDiskBBQVectorsFormat}. */
    public static final QuantizerScheme OSQ = new QuantizerScheme() {
        @Override
        public int id() {
            return 1;
        }

        @Override
        public String name() {
            return "osq";
        }

        @Override
        public KnnVectorsFormat newFormat(
            BBQIVFIndexOptions options,
            ElementType elementType,
            @Nullable ExecutorService mergingExecutorService,
            int maxMergingWorkers,
            @Nullable ExecutorService quantizerExecutorService,
            @Nullable String sliceField
        ) {
            IvfMergeConfigResolver mergeConfigResolver = options.autoCalibrate()
                ? IvfAutoCalibration.mergeConfigResolver(options.getClusterSize(), options.autoCalibrationProfile())
                : IvfMergeConfigResolver.useCodecDefault();
            return new ESNextDiskBBQVectorsFormat(
                QuantEncoding.fromBits((byte) options.resolve(null).bits()),
                options.getClusterSize(),
                ESNextDiskBBQVectorsFormat.DEFAULT_CENTROIDS_PER_PARENT_CLUSTER,
                elementType,
                options.isOnDiskRescore(),
                mergingExecutorService,
                maxMergingWorkers,
                options.doPrecondition(),
                ESNextDiskBBQVectorsFormat.DEFAULT_PRECONDITIONING_BLOCK_DIMENSION,
                options.getFlatIndexThreshold(),
                sliceField,
                IvfFlushConfigSource.empty(),
                mergeConfigResolver,
                options.isOnDiskMerge()
            );
        }

        @Override
        public DistortionRateQuantizer distortionRateQuantizer(
            BBQIVFIndexOptions options,
            int dimension,
            @Nullable ClusteredSample sample,
            long seed
        ) {
            return OsqDistortionRateQuantizer.forIndex(
                dimension,
                QuantEncoding.fromBits((byte) options.resolve(dimension).bits()),
                options.doPrecondition()
            );
        }
    };

    /** Asymmetric hashing, written by {@link ESNextDiskASHVectorsFormat}. */
    public static final QuantizerScheme ASH = new QuantizerScheme() {
        @Override
        public int id() {
            return 2;
        }

        @Override
        public String name() {
            return "ash";
        }

        @Override
        public KnnVectorsFormat newFormat(
            BBQIVFIndexOptions options,
            ElementType elementType,
            @Nullable ExecutorService mergingExecutorService,
            int maxMergingWorkers,
            @Nullable ExecutorService quantizerExecutorService,
            @Nullable String sliceField
        ) {
            return new ESNextDiskASHVectorsFormat(
                options.ashConfigResolver(),
                options.getClusterSize(),
                ESNextDiskASHVectorsFormat.DEFAULT_CENTROIDS_PER_PARENT_CLUSTER,
                elementType,
                options.isOnDiskRescore(),
                mergingExecutorService,
                maxMergingWorkers,
                quantizerExecutorService,
                options.getFlatIndexThreshold(),
                sliceField,
                IvfFlushConfigSource.empty(),
                IvfMergeConfigResolver.useCodecDefault(),
                options.isOnDiskMerge()
            );
        }

        @Override
        public DistortionRateQuantizer distortionRateQuantizer(
            BBQIVFIndexOptions options,
            int dimension,
            @Nullable ClusteredSample sample,
            long seed
        ) throws IOException {
            Objects.requireNonNull(sample, "ASH learns its projection from a sample of the corpus");
            return AshDistortionRateQuantizer.train(options.ashConfigResolver().resolve(dimension), sample, seed);
        }
    };

    private static final List<QuantizerScheme> ALL = List.of(OSQ, ASH);

    /** The scheme with the given {@link QuantizerScheme#id()}. */
    public static QuantizerScheme byId(int id) {
        for (QuantizerScheme scheme : ALL) {
            if (scheme.id() == id) {
                return scheme;
            }
        }
        throw new IllegalArgumentException("unknown quantizer scheme id [" + id + "]");
    }

    /** The scheme with the given mapping {@link QuantizerScheme#name()}. */
    public static QuantizerScheme byName(String name) {
        for (QuantizerScheme scheme : ALL) {
            if (scheme.name().equalsIgnoreCase(name)) {
                return scheme;
            }
        }
        throw new IllegalArgumentException("unknown quantizer scheme [" + name + "]");
    }

    /**
     * The scheme the {@code ESNext} format of a field is written with. A mapping that asks for ASH is written with OSQ
     * when ASH is not available to the index, see {@link BBQIVFIndexOptions#usesAshFormat()}.
     */
    public static QuantizerScheme forField(DenseVectorFieldMapper.BBQIVFIndexOptions options) {
        return options.usesAshFormat() ? ASH : OSQ;
    }
}
