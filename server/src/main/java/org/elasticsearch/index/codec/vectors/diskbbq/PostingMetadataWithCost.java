/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq;

/**
 * Posting list metadata plus estimated doc count for load-balanced partitioning.
 *
 * @param metadata         The posting list metadata (offset, length, centroid info).
 * @param estimatedDocCount Estimated number of documents in the posting list (from resetPostingsScorer).
 */
public record PostingMetadataWithCost(PostingMetadata metadata, int estimatedDocCount) {}
