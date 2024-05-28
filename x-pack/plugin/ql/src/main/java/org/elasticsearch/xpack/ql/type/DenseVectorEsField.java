/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.ql.type;

import java.util.Map;

import static org.elasticsearch.xpack.ql.type.DataTypes.DENSE_VECTOR;

public class DenseVectorEsField extends EsField {

    public DenseVectorEsField(String name, Map<String, EsField> properties, boolean aggregatable) {
        super(name, DENSE_VECTOR, properties, aggregatable);
    }
}
