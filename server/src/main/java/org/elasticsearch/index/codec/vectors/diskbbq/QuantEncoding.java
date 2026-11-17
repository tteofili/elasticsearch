/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.elasticsearch.index.codec.vectors.OptimizedScalarQuantizer;
import org.elasticsearch.simdvec.E8Roots;
import org.elasticsearch.simdvec.ESVectorUtil;

public enum QuantEncoding {
    ONE_BIT_4BIT_QUERY(0, (byte) 1, (byte) 4) {
        @Override
        public void pack(int[] quantized, byte[] destination) {
            ESVectorUtil.pack1BitValues(quantized, destination);
        }

        @Override
        public void packQuery(int[] quantized, byte[] destination) {
            ESVectorUtil.stride4BitValues(quantized, destination);
        }
    },
    TWO_BIT_4BIT_QUERY(1, (byte) 2, (byte) 4) {
        @Override
        public void pack(int[] quantized, byte[] destination) {
            ESVectorUtil.pack2BitValues(quantized, destination);
        }

        @Override
        public void packQuery(int[] quantized, byte[] destination) {
            stripe4Values(quantized, destination);
        }

        @Override
        public int getDocPackedLength(int dimensions) {
            int discretized = discretizedDimensions(dimensions);
            return discretized / 4;
        }

        @Override
        public int getQueryPackedLength(int dimensions) {
            return discretizedDimensions(dimensions);
        }
    },
    FOUR_BIT_SYMMETRIC(2, (byte) 4, (byte) 4) {
        @Override
        public void packQuery(int[] quantized, byte[] destination) {
            ESVectorUtil.packAsBytes(quantized, destination, quantized.length);
        }

        @Override
        public void pack(int[] quantized, byte[] destination) {
            pack4BitValues(quantized, destination);
        }

        @Override
        public int getDocPackedLength(int dimensions) {
            int discretized = discretizedDimensions(dimensions);
            return discretized / 2;
        }

        @Override
        public int getQueryPackedLength(int dimensions) {
            return discretizedDimensions(dimensions);
        }

        @Override
        public int discretizedDimensions(int dimensions) {
            int totalBits = dimensions * 4;
            return (totalBits + 7) / 8 * 8 / 4;
        }
    },
    SEVEN_BIT_SYMMETRIC(3, (byte) 7, (byte) 7) {
        @Override
        public void pack(int[] quantized, byte[] destination) {
            ESVectorUtil.packAsBytes(quantized, destination, quantized.length);
        }

        @Override
        public void packQuery(int[] quantized, byte[] destination) {
            ESVectorUtil.packAsBytes(quantized, destination, quantized.length);
        }

        @Override
        public int discretizedDimensions(int dimensions) {
            return dimensions;
        }

        @Override
        public int getDocPackedLength(int dimensions) {
            return discretizedDimensions(dimensions);
        }

        @Override
        public int getQueryPackedLength(int dimensions) {
            return discretizedDimensions(dimensions);
        }
    },
    ONE_BIT_1BIT_QUERY(5, (byte) 1, (byte) 1) {
        @Override
        public void pack(int[] quantized, byte[] destination) {
            ESVectorUtil.pack1BitValues(quantized, destination);
        }

        @Override
        public void packQuery(int[] quantized, byte[] destination) {
            ESVectorUtil.pack1BitValues(quantized, destination);
        }
    },
    /**
     * Each block of 8 dimensions is coded as the nearest of the 240 roots of the E8 lattice, in one byte (about one bit
     * per dimension); see {@link E8Roots}. Codes are written transposed within each bulk, and queries are turned into an
     * int8 lookup table of {@link E8Roots#TABLE_SIZE} entries per block.
     */
    E8_ROOTS(6, (byte) 1, (byte) 8) {
        @Override
        public void pack(int[] quantized, byte[] destination) {
            for (int i = 0; i < destination.length; i++) {
                destination[i] = (byte) quantized[i];
            }
        }

        @Override
        public void packQuery(int[] quantized, byte[] destination) {
            pack(quantized, destination);
        }

        @Override
        public int discretizedDimensions(int dimensions) {
            return E8Roots.blocks(dimensions) * E8Roots.BLOCK_DIMS;
        }

        @Override
        public int getDocPackedLength(int dimensions) {
            return E8Roots.blocks(dimensions);
        }

        @Override
        public int getQueryPackedLength(int dimensions) {
            return E8Roots.blocks(dimensions) * E8Roots.TABLE_SIZE;
        }

        @Override
        public OptimizedScalarQuantizer newQuantizer(VectorSimilarityFunction similarityFunction) {
            return new E8RootQuantizer(similarityFunction);
        }

        @Override
        public OptimizedScalarQuantizer.QuantizationResult quantizeQuery(
            OptimizedScalarQuantizer quantizer,
            float[] query,
            float[] residualScratch,
            int[] quantizedScratch,
            float[] centroid,
            byte[] destination
        ) {
            return E8RootQuantizer.quantizeQuery(query, centroid, residualScratch, destination);
        }

        @Override
        public boolean transposed() {
            return true;
        }
    };

    private static void pack4BitValues(int[] quantized, byte[] destination) {
        assert quantized.length == destination.length * 2;
        int packedLength = destination.length;
        for (int i = 0; i < packedLength; i++) {
            destination[i] = (byte) ((quantized[i] << 4) | (quantized[packedLength + i] & 0x0F));
        }
    }

    /**
     * Moves 4 successive byte values into striped format - all the first bytes first, then all the second bytes, then third, then fourth
     */
    private static void stripe4Values(int[] quantized, byte[] destination) {
        assert quantized.length == destination.length;
        assert destination.length % 4 == 0;
        int packedLength = destination.length / 4;
        for (int i = 0; i < packedLength; i++) {
            destination[i] = (byte) quantized[4 * i];
            destination[i + packedLength] = (byte) quantized[4 * i + 1];
            destination[i + 2 * packedLength] = (byte) quantized[4 * i + 2];
            destination[i + 3 * packedLength] = (byte) quantized[4 * i + 3];
        }
    }

    private final int id;
    private final byte bits, queryBits;

    QuantEncoding(int id, byte bits, byte queryBits) {
        this.id = id;
        this.bits = bits;
        this.queryBits = queryBits;
    }

    public abstract void pack(int[] quantized, byte[] destination);

    public abstract void packQuery(int[] quantized, byte[] destination);

    public int id() {
        return id;
    }

    public byte bits() {
        return bits;
    }

    public byte queryBits() {
        return queryBits;
    }

    public int discretizedDimensions(int dimensions) {
        if (queryBits == bits) {
            int totalBits = dimensions * bits;
            return (totalBits + 7) / 8 * 8 / bits;
        }
        int queryDiscretized = (dimensions * queryBits + 7) / 8 * 8 / queryBits;
        int docDiscretized = (dimensions * bits + 7) / 8 * 8 / bits;
        int maxDiscretized = Math.max(queryDiscretized, docDiscretized);
        assert maxDiscretized % (8.0 / queryBits) == 0 : "bad discretized=" + maxDiscretized + " for dim=" + dimensions;
        assert maxDiscretized % (8.0 / bits) == 0 : "bad discretized=" + maxDiscretized + " for dim=" + dimensions;
        return maxDiscretized;
    }

    /** Return the number of bytes required to store a packed vector of the given dimensions. */
    public int getDocPackedLength(int dimensions) {
        int discretized = discretizedDimensions(dimensions);
        // how many bytes do we need to store the quantized vector?
        int totalBits = discretized * bits;
        return (totalBits + 7) / 8;
    }

    public int getQueryPackedLength(int dimensions) {
        int discretized = discretizedDimensions(dimensions);
        // how many bytes do we need to store the quantized vector?
        int totalBits = discretized * queryBits;
        return (totalBits + 7) / 8;
    }

    /** The quantizer that encodes documents for this encoding. */
    public OptimizedScalarQuantizer newQuantizer(VectorSimilarityFunction similarityFunction) {
        return new OptimizedScalarQuantizer(similarityFunction);
    }

    /** Quantizes a query, centred on {@code centroid}, into {@code destination} ({@link #getQueryPackedLength} bytes). */
    public OptimizedScalarQuantizer.QuantizationResult quantizeQuery(
        OptimizedScalarQuantizer quantizer,
        float[] query,
        float[] residualScratch,
        int[] quantizedScratch,
        float[] centroid,
        byte[] destination
    ) {
        OptimizedScalarQuantizer.QuantizationResult result = quantizer.scalarQuantize(
            query,
            residualScratch,
            quantizedScratch,
            queryBits,
            centroid
        );
        packQuery(quantizedScratch, destination);
        return result;
    }

    /**
     * Whether document codes are stored transposed within each bulk (block-major), in which case vectors can only be
     * scored a bulk at a time.
     */
    public boolean transposed() {
        return false;
    }

    public static QuantEncoding fromId(int id) {
        for (QuantEncoding encoding : values()) {
            if (encoding.id == id) {
                return encoding;
            }
        }
        throw new IllegalArgumentException("Unknown QuantEncoding id: " + id);
    }

    /**
     * Returns {@code true} if the given bit width is a supported OSQ document quantization value.
     */
    public static boolean isValidBits(byte bits) {
        return switch (bits) {
            case 1, 2, 4, 7 -> true;
            default -> false;
        };
    }

    public static QuantEncoding fromBits(byte bits) {
        return switch (bits) {
            case 1 -> ONE_BIT_4BIT_QUERY;
            case 2 -> TWO_BIT_4BIT_QUERY;
            case 4 -> FOUR_BIT_SYMMETRIC;
            case 7 -> SEVEN_BIT_SYMMETRIC;
            default -> throw new IllegalArgumentException("Unsupported bits: " + bits);
        };
    }

    /**
     * Resolves the quantization encoding from document and query bit widths.
     *
     * @throws IllegalArgumentException if the combination is unsupported
     */
    public static QuantEncoding fromDocAndQueryBits(byte docBits, byte queryBits) {
        return switch (docBits) {
            case 1 -> {
                if (queryBits == 1) {
                    yield ONE_BIT_1BIT_QUERY;
                }
                if (queryBits == 4) {
                    yield ONE_BIT_4BIT_QUERY;
                }
                throw new IllegalArgumentException("1-bit document quantization supports query bits 1 or 4, but got: " + queryBits);
            }
            case 2 -> {
                if (queryBits != 4) {
                    throw new IllegalArgumentException(
                        "2-bit document quantization requires 4-bit query quantization, but got: " + queryBits
                    );
                }
                yield TWO_BIT_4BIT_QUERY;
            }
            case 4 -> {
                if (queryBits != 4) {
                    throw new IllegalArgumentException("4-bit symmetric quantization requires query bits 4, but got: " + queryBits);
                }
                yield FOUR_BIT_SYMMETRIC;
            }
            case 7 -> {
                if (queryBits != 7) {
                    throw new IllegalArgumentException("7-bit symmetric quantization requires query bits 7, but got: " + queryBits);
                }
                yield SEVEN_BIT_SYMMETRIC;
            }
            default -> throw new IllegalArgumentException("Unsupported document bits: " + docBits);
        };
    }
}
