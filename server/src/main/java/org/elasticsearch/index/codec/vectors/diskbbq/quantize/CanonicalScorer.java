/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq.quantize;

import java.io.IOException;

/**
 * The score of a vector in a posting list, which every scheme computes as one formula with different bookkeeping:
 * <pre>
 *   score = s * (alpha * r + beta * sigma + gamma) + l * kappa + o + omega
 * </pre>
 * where {@code r} is the raw dot product of the code of the vector and the query's, {@code (s, sigma, l, o)} are the
 * {@link ScoreLanes} stored for the vector and {@code (alpha, beta, gamma, kappa, omega)} the {@link QueryScalars} of the
 * list. This is the scalar reference of what the scorers compute in bulk, and what they are tested against.
 */
public final class CanonicalScorer {

    private CanonicalScorer() {}

    public static float score(ScoreLanes lanes, QueryScalars query, float rawDot) {
        return lanes.s * (query.alpha() * rawDot + query.beta() * lanes.sigma + query.gamma()) + lanes.l * query.kappa() + lanes.o + query
            .omega();
    }

    /** The score of the {@code i}-th vector of the block that {@code encoder} holds, for the list that {@code query} is on. */
    public static float score(DocEncoder encoder, int i, PreparedQuery query) throws IOException {
        int[] levels = new int[encoder.codeLength()];
        DocLanes decodeLanes = new DocLanes();
        encoder.read(i, levels, decodeLanes);
        ScoreLanes lanes = new ScoreLanes();
        encoder.scoreLanes(i, lanes);
        return score(lanes, query.scalars(), query.rawDot(levels));
    }
}
