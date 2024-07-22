/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.compute.data;

import org.elasticsearch.common.io.stream.StreamOutput;
import org.elasticsearch.common.unit.ByteSizeValue;
import org.elasticsearch.core.ReleasableIterator;
import org.elasticsearch.core.Releasables;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Wrapper around {@link DocVector} to make a valid {@link Block}.
 */
public class DoubleVectorVectorBlock extends AbstractVectorBlock implements Block {

    private final DoubleVectorVector vector;

    DoubleVectorVectorBlock(DoubleVectorVector vector) {
        this.vector = vector;
    }

    @Override
    public String getWriteableName() {
        throw new UnsupportedOperationException();
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        throw new UnsupportedOperationException();
    }

    @Override
    public DoubleVectorVector asVector() {
        return vector;
    }

    @Override
    public ElementType elementType() {
        return ElementType.DOC;
    }

    @Override
    public Block filter(int... positions) {
        return new DoubleVectorVectorBlock(asVector().filter(positions));
    }

    @Override
    public ReleasableIterator<? extends Block> lookup(IntBlock positions, ByteSizeValue targetBlockSize) {
        throw new UnsupportedOperationException("can't lookup values from DocBlock");
    }

    @Override
    public DoubleVectorVectorBlock expand() {
        incRef();
        return this;
    }

    @Override
    public int hashCode() {
        return vector.hashCode();
    }

    @Override
    public boolean equals(Object obj) {
        if (obj instanceof DoubleVectorVectorBlock == false) {
            return false;
        }
        return this == obj || vector.equals(((DoubleVectorVectorBlock) obj).vector);
    }

    @Override
    public long ramBytesUsed() {
        return vector.ramBytesUsed();
    }

    @Override
    public void closeInternal() {
        assert (vector.isReleased() == false) : "can't release block [" + this + "] containing already released vector";
        Releasables.closeExpectNoException(vector);
    }

    /**
     * A builder the for {@link DoubleVectorVectorBlock}.
     */
    public static Builder newBlockBuilder(BlockFactory blockFactory, int estimatedSize) {
        return new Builder(blockFactory, estimatedSize);
    }

    public static class Builder implements Block.Builder {
        private final List<DoubleVector> vectors;

        private DoubleVectorBuilder vectorBuilder;

        private Builder(BlockFactory blockFactory, int estimatedSize) {
            vectors = new ArrayList<>(estimatedSize);
            vectorBuilder = new DoubleVectorBuilder(estimatedSize, blockFactory);
        }

        public Builder appendDoubles(double... doubles) {
            beginPositionEntry();
            for (double aDouble : doubles) {
                vectorBuilder.appendDouble(aDouble);
            }
            endPositionEntry();
            return this;
        }

        @Override
        public Builder appendNull() {
            throw new UnsupportedOperationException("double vector vector blocks can't contain null");
        }

        @Override
        public Builder beginPositionEntry() {
            return this;
        }

        @Override
        public Builder endPositionEntry() {
            int estimatedSize = vectorBuilder.valuesLength();
            vectors.add(vectorBuilder.build());
            vectorBuilder = new DoubleVectorBuilder(estimatedSize, vectorBuilder.blockFactory);
            return this;
        }

        @Override
        public Builder copyFrom(Block block, int beginInclusive, int endExclusive) {
            DoubleVectorVector doubleVectorVector = ((DoubleVectorVectorBlock) block).asVector();
            for (int i = beginInclusive; i < endExclusive; i++) {
                shards.appendInt(docVector.shards().getInt(i));
                segments.appendInt(docVector.segments().getInt(i));
                docs.appendInt(docVector.docs().getInt(i));
            }
            return this;
        }

        @Override
        public Block.Builder mvOrdering(MvOrdering mvOrdering) {
            /*
             * This is called when copying but otherwise doesn't do
             * anything because there aren't multivalue fields in a
             * block containing doc references. Every position can
             * only reference one doc.
             */
            return this;
        }

        @Override
        public long estimatedBytes() {
            return DocVector.BASE_RAM_BYTES_USED + shards.estimatedBytes() + segments.estimatedBytes() + docs.estimatedBytes();
        }

        @Override
        public DoubleVectorVectorBlock build() {
            // Pass null for singleSegmentNonDecreasing so we calculate it when we first need it.
            IntVector shards = null;
            IntVector segments = null;
            IntVector docs = null;
            DocVector result = null;
            try {
                shards = this.shards.build();
                segments = this.segments.build();
                docs = this.docs.build();
                result = new DocVector(shards, segments, docs, null);
                return result.asBlock();
            } finally {
                if (result == null) {
                    Releasables.closeExpectNoException(shards, segments, docs);
                }
            }
        }

        @Override
        public void close() {
            Releasables.closeExpectNoException(shards, segments, docs);
        }
    }

    @Override
    public void allowPassingToDifferentDriver() {
        vector.allowPassingToDifferentDriver();
    }

    @Override
    public int getPositionCount() {
        return vector.getPositionCount();
    }

    @Override
    public BlockFactory blockFactory() {
        return vector.blockFactory();
    }
}
