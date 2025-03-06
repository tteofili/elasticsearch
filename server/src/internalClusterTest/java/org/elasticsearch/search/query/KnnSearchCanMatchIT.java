/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.search.query;

import org.elasticsearch.action.index.IndexRequestBuilder;
import org.elasticsearch.action.search.SearchRequest;
import org.elasticsearch.cluster.metadata.IndexMetadata;
import org.elasticsearch.cluster.routing.allocation.DataTier;
import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.index.mapper.vectors.DenseVectorFieldMapper;
import org.elasticsearch.index.query.TermQueryBuilder;
import org.elasticsearch.search.builder.SearchSourceBuilder;
import org.elasticsearch.search.vectors.KnnSearchBuilder;
import org.elasticsearch.test.ESIntegTestCase;
import org.elasticsearch.xcontent.XContentBuilder;
import org.elasticsearch.xcontent.XContentFactory;
import org.junit.Before;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.stream.Collectors;

import static org.elasticsearch.test.hamcrest.ElasticsearchAssertions.assertResponse;
import static org.hamcrest.Matchers.equalTo;

@ESIntegTestCase.ClusterScope(scope = ESIntegTestCase.Scope.TEST, numDataNodes = 4, numClientNodes = 1)
public class KnnSearchCanMatchIT extends ESIntegTestCase {

    public static final String INDEX_NAME = "test";
    public static final String VECTOR_FIELD = "vector";

    @Before
    public void setUpMasterNode() {
        internalCluster().startMasterOnlyNode();
    }

    @Before
    public void setup() throws IOException {
        String type = randomFrom(
            Arrays.stream(DenseVectorFieldMapper.VectorIndexType.values())
                .filter(DenseVectorFieldMapper.VectorIndexType::isQuantized)
                .map(t -> t.name().toLowerCase(Locale.ROOT))
                .collect(Collectors.toList())
        );
        XContentBuilder mapping = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(VECTOR_FIELD)
            .field("type", "dense_vector")
            .field("similarity", "l2_norm")
            .startObject("index_options")
            .field("type", type)
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        Settings settings = Settings.builder()
            .put(IndexMetadata.SETTING_NUMBER_OF_REPLICAS, 0)
            .put(IndexMetadata.SETTING_NUMBER_OF_SHARDS, randomIntBetween(5, 10))
            .put(DataTier.TIER_PREFERENCE, DataTier.DATA_HOT)
            .build();
        prepareCreate(INDEX_NAME).setMapping(mapping).setSettings(settings).get();
        ensureGreen(INDEX_NAME);
    }

    public void testFilteredKnnCanMatch() throws Exception {
        int numDocs = randomIntBetween(5, 10);
        IndexRequestBuilder[] docs = new IndexRequestBuilder[numDocs];

        int numDims = 64;
        for (int i = 0; i < numDocs; i++) {
            docs[i] = prepareIndex(INDEX_NAME).setId("" + i).setSource(VECTOR_FIELD, randomVector(numDims),
                "cat", random().nextBoolean() ? "text" : "image");
        }
        indexRandom(true, docs);
        refresh(INDEX_NAME);

        KnnSearchBuilder knnSearchBuilder = new KnnSearchBuilder(VECTOR_FIELD, randomVector(numDims), 1, 1, null, null).addFilterQuery(
            new TermQueryBuilder("cat", "image")
        );

        final SearchRequest searchRequest = new SearchRequest();
        searchRequest.indices(INDEX_NAME);

        // we set the pre filter shard size to 1 automatically to make sure the can_match phase runs
        searchRequest.setPreFilterShardSize(1);
        searchRequest.source(SearchSourceBuilder.searchSource().query(knnSearchBuilder.toQueryBuilder()));

        assertResponse(client().search(searchRequest), searchResponse -> {
            // we're only querying the hot tier which is available so we shouldn't get any failures
            assertThat(searchResponse.getFailedShards(), equalTo(0));
            // we should be receiving the 2 docs from the index that's in the data_hot tier
            assertNotNull(searchResponse.getHits().getTotalHits());
            assertThat(searchResponse.getHits().getTotalHits().value(), equalTo(1L));
        });
    }

    private static float[] randomVector(int numDimensions) {
        float[] vector = new float[numDimensions];
        for (int j = 0; j < numDimensions; j++) {
            vector[j] = randomFloatBetween(0, 1, true);
        }
        return vector;
    }

}
