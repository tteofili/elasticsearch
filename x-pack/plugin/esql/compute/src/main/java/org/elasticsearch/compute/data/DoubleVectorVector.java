/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.compute.data;

import org.apache.lucene.util.RamUsageEstimator;
import org.elasticsearch.common.unit.ByteSizeValue;
import org.elasticsearch.core.ReleasableIterator;
import org.elasticsearch.core.Releasables;

import java.util.Arrays;

/**
 * {@link Vector} where each entry references a double vector.
 */
public final class DoubleVectorVector extends AbstractVector implements Vector {

    static final long BASE_RAM_BYTES_USED = RamUsageEstimator.shallowSizeOfInstance(DoubleVectorVector.class);

    /**
     * Per position memory cost to build the shard segment doc map required
     * to load fields out of order.
     */
    //public static final int SHARD_SEGMENT_DOC_MAP_PER_ROW_OVERHEAD = Integer.BYTES * 2;

    private final DoubleVector[] vectors;

    public DoubleVectorVector(DoubleVector[] vectors, BlockFactory blockFactory) {
        super(vectors.length > 0 ? vectors[0].getPositionCount() : 0, blockFactory);
        this.vectors = vectors;
        //blockFactory().adjustBreaker(BASE_RAM_BYTES_USED);
    }

    public DoubleVector[] vectors() {
        return vectors;
    }

    @Override
    public DoubleVectorVectorBlock asBlock() {
        return new DoubleVectorVectorBlock(this);
    }

    @Override
    public DoubleVectorVector filter(int... positions) {
        DoubleVectorVector result = null;
        DoubleVector[] newVectors = new DoubleVector[this.vectors.length];
        // TODO : wrong, fix it!
        try {
            int i = 0;
            for (DoubleVector vector : this.vectors) {
                vector = vector.filter(positions);
                newVectors[i] = vector;
                i++;
            }
            result = new DoubleVectorVector(newVectors, blockFactory());
            return result;
        } finally {
            if (result == null) {
                Releasables.closeExpectNoException(newVectors);
            }
        }
    }

    @Override
    public ReleasableIterator<? extends Block> lookup(IntBlock positions, ByteSizeValue targetBlockSize) {
        throw new UnsupportedOperationException("can't lookup values from DoubleVectorVector");
    }

    @Override
    public ElementType elementType() {
        return ElementType.DENSE_VECTOR;
    }

    @Override
    public boolean isConstant() {
        return Arrays.stream(vectors).allMatch(DoubleVector::isConstant);
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(vectors);
    }

    @Override
    public boolean equals(Object obj) {
        if (obj instanceof DoubleVectorVector == false) {
            return false;
        }
        DoubleVectorVector other = (DoubleVectorVector) obj;
        return Arrays.equals(vectors, other.vectors);
    }

    @Override
    public long ramBytesUsed() {
        return Arrays.stream(vectors).map(DoubleVector::ramBytesUsed).reduce(Long::sum).orElseThrow();
    }

    @Override
    public void allowPassingToDifferentDriver() {
        super.allowPassingToDifferentDriver();
        Arrays.stream(vectors).forEach(Vector::allowPassingToDifferentDriver);
    }

    @Override
    public void closeInternal() {
        Releasables.closeExpectNoException(vectors);
    }
}
