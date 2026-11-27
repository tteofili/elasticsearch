/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq.quantize;

/**
 * What a posting list needs of the query, in the form that scoring takes it, see {@link CanonicalScorer}. They depend on
 * the query and on the centroid of the list and never on a vector, and the factor that the Euclidean distance puts in front
 * of the inner product is folded in, so that what is stored for a vector does not depend on the query.
 */
public record QueryScalars(float alpha, float beta, float gamma, float kappa, float omega) {}
