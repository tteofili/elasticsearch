/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq.calibrate;

import org.elasticsearch.index.codec.vectors.diskbbq.IvfSegmentConfig;

import java.io.IOException;
import java.util.List;

/**
 * The quantizer-specific part of IVF auto-calibration.
 *
 * <p>Auto-calibration picks the cheapest quantization configuration whose predicted recall meets a target. Most of
 * that machinery does not depend on how vectors are quantized: the manifold fit, the expected-recall model and the
 * extrapolation of the measured error to the real corpus size only consume a quantization <em>error standard
 * deviation</em>. A {@code CalibrationQuantizer} supplies the remaining pieces, which are inherently tied to one
 * quantization scheme:
 * <ul>
 *   <li>the ordered set of {@link Candidate}s worth trying, with the cost of each,</li>
 *   <li>how to measure, for a candidate, the error between estimated and exact similarities on a shared
 *       clustering of a corpus sample ({@link #measureError}), and</li>
 *   <li>the {@link IvfSegmentConfig.QuantConfig} to emit for the winning candidate.</li>
 * </ul>
 * The clustering itself (doc centroids, assignments, query centroids) is computed once by {@link ErrorModel} and
 * handed to the quantizer as a {@link ResidualSample}, so it is shared across candidates and quantizers.
 */
public interface CalibrationQuantizer {

    /**
     * One quantization configuration the sweep can select, e.g. an OSQ encoding.
     */
    interface Candidate {
        /**
         * Identifies the candidate within its quantizer. Candidates with the same id are assumed to have the same
         * error behavior, so measured error models are cached by id.
         */
        String id();

        /** The configuration written to the segment if this candidate is selected. */
        IvfSegmentConfig.QuantConfig quantConfig();

        /**
         * Permanent per-segment storage and memory cost of this candidate, on the same scale as the rerank-depth
         * cost it is summed with when ordering the sweep. Candidates are tried in ascending total cost.
         */
        double storageCost();

        /** Breaks ties between candidates of equal total cost: cheaper query-side work is tried first. */
        double queryCost();
    }

    /**
     * Opaque reusable buffers for {@link #measureError}, created once per sweep by {@link #newScratch} and threaded
     * through every measurement so large arrays are not reallocated per candidate.
     */
    interface Scratch {}

    /**
     * Clustering of a corpus prefix that every candidate's error is measured against.
     *
     * @param source the calibration inputs (vectors, query and corpus ordinals, preconditioner, similarity)
     * @param usePreconditioned whether vectors are measured in the preconditioned (rotated) space; the centroids
     *                          below are already in that space
     * @param nDocs number of corpus vectors in the sample
     * @param docAssignments for each of the {@code nDocs} sample vectors, the index of its doc centroid
     * @param docCentroids centroids of the doc clusters
     * @param queryCentroids centroids of the (coarser) query clusters
     * @param docCentroidToQueryCentroid for each doc centroid, the index of the query centroid its docs are scored with
     */
    record ResidualSample(
        CalibrationSource source,
        boolean usePreconditioned,
        int nDocs,
        int[] docAssignments,
        float[][] docCentroids,
        float[][] queryCentroids,
        int[] docCentroidToQueryCentroid
    ) {}

    /**
     * Candidates to sweep, in no particular order, restricted to those within the document-bit ceiling.
     *
     * @param maxDocBits ceiling on per-dimension document bits; candidates that need more are excluded
     */
    List<Candidate> candidates(int maxDocBits);

    /** Candidate reported when the sweep has no candidate at all (e.g. the ceiling excludes everything). */
    Candidate defaultCandidate();

    /** Candidate whose error is used to fit how error scales with corpus size in the multi-sample (full) path. */
    Candidate scalingCandidate();

    /** Allocates buffers sized for measuring up to {@code maxNDocs} corpus vectors of {@code source}. */
    Scratch newScratch(CalibrationSource source, int maxNDocs);

    /**
     * Quantizes the sample's documents and the calibration queries with {@code candidate} and returns the standard
     * deviation of (exact similarity - estimated similarity) over the top-ranked estimated results.
     */
    double measureError(Candidate candidate, ResidualSample sample, Scratch scratch) throws IOException;
}
