/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.esql.expression.function.scalar;

import org.elasticsearch.xpack.esql.core.expression.Expression;
import org.elasticsearch.xpack.esql.core.tree.Source;
import org.elasticsearch.xpack.esql.expression.function.AbstractScalarFunctionTestCase;
import org.elasticsearch.xpack.esql.session.Configuration;

import java.util.List;

import static org.elasticsearch.xpack.esql.ConfigurationTestUtils.randomConfiguration;
import static org.elasticsearch.xpack.esql.ConfigurationTestUtils.randomTables;
import static org.elasticsearch.xpack.esql.SerializationTestUtils.assertSerialization;

public abstract class AbstractConfigurationFunctionTestCase extends AbstractScalarFunctionTestCase {
    protected abstract Expression buildWithConfiguration(Source source, List<Expression> args, Configuration configuration);

    @Override
    protected Expression build(Source source, List<Expression> args) {
        return buildWithConfiguration(source, args, testCase.getConfiguration());
    }

    public void testSerializationWithConfiguration() {
        assumeTrue("can't serialize function", canSerialize());

        Configuration config = randomConfiguration(testCase.getSource().text(), randomTables());
        Expression expr = buildWithConfiguration(testCase.getSource(), testCase.getDataAsFields(), config);

        assertSerialization(expr, config);

        Configuration differentConfig = randomValueOtherThan(
            config,
            // The source must match the original (static) one, as function source serialization depends on it
            () -> randomConfiguration(testCase.getSource().text(), randomTables())
        );

        Expression differentExpr = buildWithConfiguration(testCase.getSource(), testCase.getDataAsFields(), differentConfig);
        assertNotEquals(expr, differentExpr);
    }
}
