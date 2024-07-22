/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.compute;

import org.elasticsearch.common.breaker.CircuitBreakingException;
import org.elasticsearch.common.unit.ByteSizeValue;
import org.elasticsearch.compute.data.Block;
import org.elasticsearch.compute.data.BlockFactory;
import org.elasticsearch.compute.data.DocVector;
import org.elasticsearch.compute.data.DoubleVector;
import org.elasticsearch.compute.data.DoubleVectorVector;
import org.elasticsearch.compute.data.DoubleVectorVectorBlockBuilder;
import org.elasticsearch.compute.data.IntVector;
import org.elasticsearch.compute.data.Page;
import org.elasticsearch.compute.data.TestBlockFactory;
import org.elasticsearch.compute.operator.ComputeTestCase;
import org.elasticsearch.core.Releasables;
import org.elasticsearch.test.BreakerTestUtil;

import java.util.function.Function;

import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.is;

public class DoubleVectorVectorTests extends ComputeTestCase {

    public void testBuildBreaks() {
        var maxBreakLimit = BreakerTestUtil.findBreakerLimit(ByteSizeValue.ofMb(128), limit -> {
            BlockFactory blockFactory = blockFactory(limit);
            buildVectorsBlock(blockFactory).close();
        });
        var limit = ByteSizeValue.ofBytes(randomLongBetween(0, maxBreakLimit.getBytes()));
        BlockFactory blockFactory = blockFactory(limit);
        Exception e = expectThrows(CircuitBreakingException.class, () -> buildVectorsBlock(blockFactory).close());
        assertThat(e.getMessage(), equalTo("over test limit"));
        logger.info("break position", e);
        assertThat(blockFactory.breaker().getUsed(), equalTo(0L));
    }

    private Block buildVectorsBlock(BlockFactory blockFactory) {
        int size = 100;
        try (DoubleVectorVectorBlockBuilder builder = new DoubleVectorVectorBlockBuilder(blockFactory, size)) {
            for (int r = 0; r < size; r++) {
                builder.appendDoubles(randomDouble(), randomDouble(), randomDouble());
            }
            return builder.build();
        }
    }

    public void testCannotDoubleRelease() {
        BlockFactory blockFactory = blockFactory();
        DoubleVector[] vectors = new DoubleVector[2];
        double[] values = new double[3];
        vectors[0] = blockFactory.newDoubleArrayVector(values, 3);
        vectors[1] = blockFactory.newDoubleArrayVector(values, 3);
        var block = new DoubleVectorVector(vectors, blockFactory).asBlock();
        assertThat(block.isReleased(), is(false));
        Page page = new Page(block);

        Releasables.closeExpectNoException(block);
        assertThat(block.isReleased(), is(true));

        Exception e = expectThrows(IllegalStateException.class, () -> block.close());
        assertThat(e.getMessage(), containsString("can't release already released object"));

        e = expectThrows(IllegalStateException.class, () -> page.getBlock(0));
        assertThat(e.getMessage(), containsString("can't read released block"));

        e = expectThrows(IllegalArgumentException.class, () -> new Page(block));
        assertThat(e.getMessage(), containsString("can't build page out of released blocks"));
    }

    public void testRamBytesUsed() {
        BlockFactory blockFactory = blockFactory();
        DoubleVector[] vectors = new DoubleVector[2];
        double[] values = new double[3];
        vectors[0] = blockFactory.newDoubleArrayVector(values, 3);
        vectors[1] = blockFactory.newDoubleArrayVector(values, 3);
        DoubleVectorVector vector = new DoubleVectorVector(vectors, blockFactory);
        long l = vector.ramBytesUsed();
        assertEquals(240, l);
        vector.close();
    }

    public void testFilter() {
        BlockFactory blockFactory = blockFactory();
        DoubleVector[] vectors = new DoubleVector[2];
        double[] values = new double[3];
        vectors[0] = blockFactory.newDoubleArrayVector(values, 3);
        vectors[1] = blockFactory.newDoubleArrayVector(values, 3);

        double[] values1 = {randomDouble(), randomDouble(), randomDouble()};
        double[] values2 = {randomDouble(), randomDouble(), randomDouble()};
        try (
            DoubleVectorVector vector = new DoubleVectorVector(new DoubleVector[]{
                blockFactory.newDoubleArrayVector(values1, 3),
                blockFactory.newDoubleArrayVector(new double[]{randomDouble(), randomDouble(), randomDouble()}, 3),
                blockFactory.newDoubleArrayVector(values2, 3),
            }, blockFactory);
            DoubleVectorVector filtered = vector.filter(0, 2);
            DoubleVectorVector expected = new DoubleVectorVector(new DoubleVector[]{
                blockFactory.newDoubleArrayVector(values1, 3),
                blockFactory.newDoubleArrayVector(values2, 3),
            },blockFactory
            );
        ) {
            assertThat(filtered, equalTo(expected));
        }
    }

    public void testFilterBreaks() {
        Function<BlockFactory, DocVector> buildDocVector = factory -> {
            IntVector shards = null;
            IntVector segments = null;
            IntVector docs = null;
            DocVector result = null;
            try {
                shards = factory.newConstantIntVector(0, 10);
                segments = factory.newConstantIntVector(0, 10);
                docs = factory.newConstantIntVector(0, 10);
                result = new DocVector(shards, segments, docs, false);
                return result;
            } finally {
                if (result == null) {
                    Releasables.close(shards, segments, docs);
                }
            }
        };
        ByteSizeValue buildBreakLimit = BreakerTestUtil.findBreakerLimit(ByteSizeValue.ofMb(128), limit -> {
            BlockFactory factory = blockFactory(limit);
            buildDocVector.apply(factory).close();
        });
        ByteSizeValue filterBreakLimit = BreakerTestUtil.findBreakerLimit(ByteSizeValue.ofMb(128), limit -> {
            BlockFactory factory = blockFactory(limit);
            try (DocVector docs = buildDocVector.apply(factory)) {
                docs.filter(1, 2, 3).close();
            }
        });
        ByteSizeValue limit = ByteSizeValue.ofBytes(randomLongBetween(buildBreakLimit.getBytes() + 1, filterBreakLimit.getBytes()));
        BlockFactory factory = blockFactory(limit);
        try (DocVector docs = buildDocVector.apply(factory)) {
            Exception e = expectThrows(CircuitBreakingException.class, () -> docs.filter(1, 2, 3));
            assertThat(e.getMessage(), equalTo("over test limit"));
        }
    }

    IntVector intRange(int startInclusive, int endExclusive) {
        return IntVector.range(startInclusive, endExclusive, TestBlockFactory.getNonBreakingInstance());
    }
}
