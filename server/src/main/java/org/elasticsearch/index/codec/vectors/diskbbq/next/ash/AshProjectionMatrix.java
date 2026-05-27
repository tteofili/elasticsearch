/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq.next.ash;

import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Serialization for the ASH projection matrix W. Stored in the preconditioner slot
 * of the {@code .cenivf} file.
 * <p>
 * Format:
 * <pre>
 *   [int] originalDim (number of rows in W)
 *   [int] nDims (number of columns in W, i.e. projected dimensions)
 *   [float[originalDim * nDims]] W matrix in row-major order (little-endian)
 * </pre>
 */
public final class AshProjectionMatrix {

    private final float[][] w;
    private final int originalDim;
    private final int nDims;

    public AshProjectionMatrix(float[][] w) {
        this.w = w;
        this.originalDim = w.length;
        this.nDims = w.length > 0 ? w[0].length : 0;
    }

    public float[][] w() {
        return w;
    }

    public int originalDim() {
        return originalDim;
    }

    public int nDims() {
        return nDims;
    }

    /**
     * Writes the projection matrix to the given output.
     */
    public void write(IndexOutput out) throws IOException {
        out.writeInt(originalDim);
        out.writeInt(nDims);
        ByteBuffer buffer = ByteBuffer.allocate(nDims * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < originalDim; i++) {
            buffer.clear();
            buffer.asFloatBuffer().put(w[i]);
            out.writeBytes(buffer.array(), buffer.array().length);
        }
    }

    /**
     * Reads a projection matrix from the given input.
     */
    public static AshProjectionMatrix read(IndexInput in) throws IOException {
        int originalDim = in.readInt();
        int nDims = in.readInt();
        float[][] w = new float[originalDim][nDims];
        byte[] rowBytes = new byte[nDims * Float.BYTES];
        ByteBuffer buffer = ByteBuffer.wrap(rowBytes).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < originalDim; i++) {
            in.readBytes(rowBytes, 0, rowBytes.length);
            buffer.clear();
            buffer.asFloatBuffer().get(w[i]);
        }
        return new AshProjectionMatrix(w);
    }

    /**
     * Returns the byte size of the serialized matrix.
     */
    public long byteSize() {
        return Integer.BYTES * 2 + (long) originalDim * nDims * Float.BYTES;
    }
}
