/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.search.profile.query;

import org.elasticsearch.common.io.stream.Writeable.Reader;
import org.elasticsearch.search.SearchResponseUtils;
import org.elasticsearch.search.profile.ProfileResult;
import org.elasticsearch.search.profile.ProfileResultTests;
import org.elasticsearch.search.vectors.IVFProfile;
import org.elasticsearch.search.vectors.IVFSegmentProfile;
import org.elasticsearch.test.AbstractXContentSerializingTestCase;
import org.elasticsearch.xcontent.XContentParser;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Predicate;

import static org.elasticsearch.common.xcontent.XContentParserUtils.ensureExpectedToken;
import static org.hamcrest.Matchers.hasKey;

public class QueryProfileShardResultTests extends AbstractXContentSerializingTestCase<QueryProfileShardResult> {
    public static QueryProfileShardResult createTestItem() {
        int size = randomIntBetween(0, 5);
        List<ProfileResult> queryProfileResults = new ArrayList<>(size);
        for (int i = 0; i < size; i++) {
            queryProfileResults.add(ProfileResultTests.createTestItem(1));
        }
        CollectorResult profileCollector = CollectorResultTests.createTestItem(2);
        long rewriteTime = randomNonNegativeLong();
        if (randomBoolean()) {
            rewriteTime = rewriteTime % 1000; // make sure to often test this with small values too
        }

        Long vectorOperationsCount = randomBoolean() ? null : randomNonNegativeLong();
        return new QueryProfileShardResult(queryProfileResults, rewriteTime, profileCollector, vectorOperationsCount);
    }

    private static IVFProfile createRandomIvfProfile() {
        IVFSegmentProfile seg = new IVFSegmentProfile(
            100,
            16,
            3,
            50,
            0.5f,
            0.9f,
            2.0,
            3L,
            1.5,
            10,
            20,
            50L,
            3L,
            900L
        );
        return new IVFProfile(
            50L,
            3L,
            List.of(seg),
            0.5f,
            0.9f,
            0.7f,
            0.2f,
            10,
            20,
            16.6f,
            5f
        );
    }

    public void testIvfProfileIncludedInXContentWhenPresent() throws IOException {
        QueryProfileShardResult result = new QueryProfileShardResult(
            List.of(),
            0L,
            CollectorResultTests.createTestItem(1),
            null,
            createRandomIvfProfile()
        );
        assertNotNull(result.getIvfProfile());
        String json = org.elasticsearch.common.Strings.toString(result);
        assertThat(org.elasticsearch.common.xcontent.XContentHelper.convertToMap(
            org.elasticsearch.xcontent.json.JsonXContent.jsonXContent,
            json,
            false
        ), hasKey(QueryProfileShardResult.IVF));
    }

    @Override
    protected QueryProfileShardResult createTestInstance() {
        return createTestItem();
    }

    @Override
    protected QueryProfileShardResult mutateInstance(QueryProfileShardResult instance) {
        return null;// TODO implement https://github.com/elastic/elasticsearch/issues/25929
    }

    @Override
    protected QueryProfileShardResult doParseInstance(XContentParser parser) throws IOException {
        ensureExpectedToken(XContentParser.Token.START_OBJECT, parser.nextToken(), parser);
        QueryProfileShardResult result = SearchResponseUtils.parseQueryProfileShardResult(parser);
        ensureExpectedToken(null, parser.nextToken(), parser);
        return result;
    }

    @Override
    protected Reader<QueryProfileShardResult> instanceReader() {
        return QueryProfileShardResult::new;
    }

    @Override
    protected Predicate<String> getRandomFieldsExcludeFilter() {
        return ProfileResultTests.RANDOM_FIELDS_EXCLUDE_FILTER;
    }
}
