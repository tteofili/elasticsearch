/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq;

import org.apache.lucene.store.ByteBuffersDataOutput;
import org.apache.lucene.store.ByteBuffersIndexInput;
import org.apache.lucene.store.ByteBuffersIndexOutput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.elasticsearch.common.logging.LogConfigurator;

import java.io.IOException;

public class PreconditionerTests extends LuceneTestCase {
    static {
        LogConfigurator.loadLog4jPlugins();
        LogConfigurator.configureESLogging();
    }

    public void testRandomProviderConfigurations() throws IOException {
        int dim = random().nextInt(128, 1024);

        int corpusLen = random().nextInt(100, 200);
        float[][] corpus = new float[corpusLen][];
        for (int i = 0; i < corpusLen; i++) {
            corpus[i] = new float[dim];
            for (int j = 0; j < dim; j++) {
                if (j > 320) {
                    corpus[i][j] = 0f;
                } else {
                    corpus[i][j] = random().nextFloat();
                }
            }
        }

        float[] query = new float[dim];
        for (int i = 0; i < dim; i++) {
            query[i] = random().nextFloat();
        }

        int blockDim = random().nextInt(8, dim);

        Preconditioner preconditioner = Preconditioner.createPreconditioner(dim, blockDim);

        float[] out = new float[dim];
        preconditioner.applyTransform(query, out);

        assertEquals(blockDim, preconditioner.blockDim);
        assertEquals(dim / blockDim + (dim % blockDim == 0 ? 0 : 1), preconditioner.permutationMatrix.length);
        assertEquals(Math.min(blockDim, dim), preconditioner.permutationMatrix[0].length);
        if (dim % blockDim == 0) {
            assertEquals(blockDim, preconditioner.permutationMatrix[preconditioner.permutationMatrix.length - 1].length);
        } else {
            assertEquals(
                dim - (long) (dim / blockDim) * blockDim,
                preconditioner.permutationMatrix[preconditioner.permutationMatrix.length - 1].length
            );
        }
        assertEquals(dim / blockDim + (dim % blockDim == 0 ? 0 : 1), preconditioner.blocks.length);
        assertEquals(Math.min(blockDim, dim), preconditioner.blocks[0].length);
        assertEquals(Math.min(blockDim, dim), preconditioner.blocks[0][0].length);

        // verify can be written and read back
        ByteBuffersDataOutput byteBuffersDataOutput = new ByteBuffersDataOutput();
        IndexOutput output = new ByteBuffersIndexOutput(byteBuffersDataOutput, "test", "test");
        preconditioner.write(output);
        Preconditioner.read(new ByteBuffersIndexInput(byteBuffersDataOutput.toDataInput(), "test"));
    }

    public void testWriteReadPreservesTransformExactly() throws IOException {
        int dim = random().nextInt(32, 256);
        int blockDim = random().nextInt(4, dim);
        Preconditioner preconditioner = Preconditioner.createPreconditioner(dim, blockDim);

        float[] vector = new float[dim];
        for (int i = 0; i < dim; i++) {
            vector[i] = random().nextFloat();
        }
        float[] before = new float[dim];
        preconditioner.applyTransform(vector, before);

        ByteBuffersDataOutput data = new ByteBuffersDataOutput();
        try (IndexOutput out = new ByteBuffersIndexOutput(data, "test", "test")) {
            preconditioner.write(out);
        }
        Preconditioner restored = Preconditioner.read(new ByteBuffersIndexInput(data.toDataInput(), "test"));
        float[] after = new float[dim];
        restored.applyTransform(vector, after);
        assertArrayEquals(before, after, 0.0f);
    }

    public void testFactoryDeterministicForSameInputs() {
        int dim = random().nextInt(32, 256);
        int blockDim = random().nextInt(4, dim);
        Preconditioner first = Preconditioner.createPreconditioner(dim, blockDim);
        Preconditioner second = Preconditioner.createPreconditioner(dim, blockDim);

        float[] vector = new float[dim];
        for (int i = 0; i < dim; i++) {
            vector[i] = random().nextFloat();
        }
        float[] out1 = new float[dim];
        float[] out2 = new float[dim];
        first.applyTransform(vector, out1);
        second.applyTransform(vector, out2);
        assertArrayEquals(out1, out2, 0.0f);
    }

    public void testReadRejectsNegativeLengths() throws IOException {
        ByteBuffersDataOutput data = new ByteBuffersDataOutput();
        try (IndexOutput out = new ByteBuffersIndexOutput(data, "test", "test")) {
            out.writeInt(1); // blocksLen
            out.writeInt(8); // blockDim
            out.writeInt(-1); // rem (invalid)
            out.writeInt(1); // permutation matrix len
            // stop here intentionally; read should fail when allocating/reading
        }
        expectThrows(RuntimeException.class, () -> Preconditioner.read(new ByteBuffersIndexInput(data.toDataInput(), "test")));
    }
}
