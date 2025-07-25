/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.inference.services.ibmwatsonx;

import org.apache.http.HttpHeaders;
import org.apache.http.client.methods.HttpPost;
import org.elasticsearch.ElasticsearchException;
import org.elasticsearch.ElasticsearchStatusException;
import org.elasticsearch.action.ActionListener;
import org.elasticsearch.action.support.PlainActionFuture;
import org.elasticsearch.common.ValidationException;
import org.elasticsearch.common.bytes.BytesArray;
import org.elasticsearch.common.bytes.BytesReference;
import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.common.xcontent.XContentHelper;
import org.elasticsearch.core.TimeValue;
import org.elasticsearch.inference.ChunkInferenceInput;
import org.elasticsearch.inference.ChunkedInference;
import org.elasticsearch.inference.ChunkingSettings;
import org.elasticsearch.inference.EmptyTaskSettings;
import org.elasticsearch.inference.InferenceServiceConfiguration;
import org.elasticsearch.inference.InferenceServiceResults;
import org.elasticsearch.inference.InputType;
import org.elasticsearch.inference.Model;
import org.elasticsearch.inference.ModelConfigurations;
import org.elasticsearch.inference.SimilarityMeasure;
import org.elasticsearch.inference.TaskType;
import org.elasticsearch.test.ESTestCase;
import org.elasticsearch.test.http.MockResponse;
import org.elasticsearch.test.http.MockWebServer;
import org.elasticsearch.threadpool.ThreadPool;
import org.elasticsearch.xcontent.ToXContent;
import org.elasticsearch.xcontent.XContentType;
import org.elasticsearch.xpack.core.inference.action.InferenceAction;
import org.elasticsearch.xpack.core.inference.results.ChunkedInferenceEmbedding;
import org.elasticsearch.xpack.core.inference.results.TextEmbeddingFloatResults;
import org.elasticsearch.xpack.inference.common.Truncator;
import org.elasticsearch.xpack.inference.external.http.HttpClientManager;
import org.elasticsearch.xpack.inference.external.http.sender.HttpRequestSender;
import org.elasticsearch.xpack.inference.external.http.sender.HttpRequestSenderTests;
import org.elasticsearch.xpack.inference.external.http.sender.Sender;
import org.elasticsearch.xpack.inference.external.request.Request;
import org.elasticsearch.xpack.inference.logging.ThrottlerManager;
import org.elasticsearch.xpack.inference.services.ServiceComponents;
import org.elasticsearch.xpack.inference.services.ServiceFields;
import org.elasticsearch.xpack.inference.services.ibmwatsonx.action.IbmWatsonxActionCreator;
import org.elasticsearch.xpack.inference.services.ibmwatsonx.embeddings.IbmWatsonxEmbeddingsModel;
import org.elasticsearch.xpack.inference.services.ibmwatsonx.embeddings.IbmWatsonxEmbeddingsModelTests;
import org.elasticsearch.xpack.inference.services.ibmwatsonx.request.IbmWatsonxEmbeddingsRequest;
import org.elasticsearch.xpack.inference.services.ibmwatsonx.rerank.IbmWatsonxRerankModel;
import org.elasticsearch.xpack.inference.services.openai.completion.OpenAiChatCompletionModelTests;
import org.hamcrest.MatcherAssert;
import org.hamcrest.Matchers;
import org.junit.After;
import org.junit.Before;

import java.io.IOException;
import java.net.URI;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

import static org.elasticsearch.common.xcontent.XContentHelper.toXContent;
import static org.elasticsearch.test.hamcrest.ElasticsearchAssertions.assertToXContentEquivalent;
import static org.elasticsearch.xpack.core.inference.results.TextEmbeddingFloatResultsTests.buildExpectationFloat;
import static org.elasticsearch.xpack.inference.Utils.getInvalidModel;
import static org.elasticsearch.xpack.inference.Utils.getPersistedConfigMap;
import static org.elasticsearch.xpack.inference.Utils.inferenceUtilityPool;
import static org.elasticsearch.xpack.inference.Utils.mockClusterServiceEmpty;
import static org.elasticsearch.xpack.inference.chunking.ChunkingSettingsTests.createRandomChunkingSettings;
import static org.elasticsearch.xpack.inference.chunking.ChunkingSettingsTests.createRandomChunkingSettingsMap;
import static org.elasticsearch.xpack.inference.external.http.Utils.entityAsMap;
import static org.elasticsearch.xpack.inference.external.http.Utils.getUrl;
import static org.elasticsearch.xpack.inference.services.ServiceComponentsTests.createWithEmptySettings;
import static org.elasticsearch.xpack.inference.services.cohere.embeddings.CohereEmbeddingsTaskSettingsTests.getTaskSettingsMapEmpty;
import static org.elasticsearch.xpack.inference.services.settings.DefaultSecretSettingsTests.getSecretSettingsMap;
import static org.hamcrest.CoreMatchers.instanceOf;
import static org.hamcrest.CoreMatchers.is;
import static org.hamcrest.Matchers.aMapWithSize;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.hasSize;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoMoreInteractions;
import static org.mockito.Mockito.when;

public class IbmWatsonxServiceTests extends ESTestCase {
    private static final TimeValue TIMEOUT = new TimeValue(30, TimeUnit.SECONDS);
    private final MockWebServer webServer = new MockWebServer();
    private ThreadPool threadPool;

    private HttpClientManager clientManager;

    private static final String apiKey = "apiKey";
    private static final String modelId = "model";
    private static final String projectId = "project_id";
    private static final String url = "https://abc.com";
    private static final String apiVersion = "2023-04-03";

    @Before
    public void init() throws Exception {
        webServer.start();
        threadPool = createThreadPool(inferenceUtilityPool());
        clientManager = HttpClientManager.create(Settings.EMPTY, threadPool, mockClusterServiceEmpty(), mock(ThrottlerManager.class));
    }

    @After
    public void shutdown() throws IOException {
        clientManager.close();
        terminate(threadPool);
        webServer.close();
    }

    public void testParseRequestConfig_CreatesAIbmWatsonxEmbeddingsModel() throws IOException {
        try (var service = createIbmWatsonxService()) {
            ActionListener<Model> modelListener = ActionListener.wrap(model -> {
                assertThat(model, instanceOf(IbmWatsonxEmbeddingsModel.class));

                var embeddingsModel = (IbmWatsonxEmbeddingsModel) model;
                assertThat(embeddingsModel.getServiceSettings().modelId(), is(modelId));
                assertThat(embeddingsModel.getServiceSettings().projectId(), is(projectId));
                assertThat(embeddingsModel.getServiceSettings().url(), is(URI.create(url)));
                assertThat(embeddingsModel.getServiceSettings().apiVersion(), is(apiVersion));
                assertThat(embeddingsModel.getSecretSettings().apiKey().toString(), is(apiKey));
                assertThat(embeddingsModel.getConfigurations().getChunkingSettings(), instanceOf(ChunkingSettings.class));
            }, e -> fail("Model parsing should have succeeded, but failed: " + e.getMessage()));

            service.parseRequestConfig(
                "id",
                TaskType.TEXT_EMBEDDING,
                getRequestConfigMap(
                    new HashMap<>(
                        Map.of(
                            ServiceFields.MODEL_ID,
                            modelId,
                            IbmWatsonxServiceFields.PROJECT_ID,
                            projectId,
                            ServiceFields.URL,
                            url,
                            IbmWatsonxServiceFields.API_VERSION,
                            apiVersion
                        )
                    ),
                    new HashMap<>(Map.of()),
                    getSecretSettingsMap(apiKey)
                ),
                modelListener
            );
        }
    }

    public void testParseRequestConfig_CreatesAIbmWatsonxRerankModel() throws IOException {
        try (var service = createIbmWatsonxService()) {
            ActionListener<Model> modelListener = ActionListener.wrap(model -> {
                assertThat(model, instanceOf(IbmWatsonxRerankModel.class));

                var rerankModel = (IbmWatsonxRerankModel) model;
                assertThat(rerankModel.getServiceSettings().modelId(), is(modelId));
                assertThat(rerankModel.getServiceSettings().projectId(), is(projectId));
                assertThat(rerankModel.getServiceSettings().apiVersion(), is(apiVersion));
                assertThat(rerankModel.getSecretSettings().apiKey().toString(), is(apiKey));
            }, e -> fail("Model parsing should have succeeded, but failed: " + e.getMessage()));

            service.parseRequestConfig(
                "id",
                TaskType.RERANK,
                getRequestConfigMap(
                    new HashMap<>(
                        Map.of(
                            ServiceFields.MODEL_ID,
                            modelId,
                            IbmWatsonxServiceFields.PROJECT_ID,
                            projectId,
                            ServiceFields.URL,
                            url,
                            IbmWatsonxServiceFields.API_VERSION,
                            apiVersion
                        )
                    ),
                    new HashMap<>(Map.of()),
                    getSecretSettingsMap(apiKey)
                ),
                modelListener
            );
        }
    }

    public void testParseRequestConfig_CreatesAIbmWatsonxEmbeddingsModelWhenChunkingSettingsProvided() throws IOException {
        try (var service = createIbmWatsonxService()) {
            ActionListener<Model> modelListener = ActionListener.wrap(model -> {
                assertThat(model, instanceOf(IbmWatsonxEmbeddingsModel.class));

                var embeddingsModel = (IbmWatsonxEmbeddingsModel) model;
                assertThat(embeddingsModel.getServiceSettings().modelId(), is(modelId));
                assertThat(embeddingsModel.getServiceSettings().projectId(), is(projectId));
                assertThat(embeddingsModel.getServiceSettings().url(), is(URI.create(url)));
                assertThat(embeddingsModel.getServiceSettings().apiVersion(), is(apiVersion));
                assertThat(embeddingsModel.getSecretSettings().apiKey().toString(), is(apiKey));
                assertThat(embeddingsModel.getConfigurations().getChunkingSettings(), instanceOf(ChunkingSettings.class));
            }, e -> fail("Model parsing should have succeeded, but failed: " + e.getMessage()));

            service.parseRequestConfig(
                "id",
                TaskType.TEXT_EMBEDDING,
                getRequestConfigMap(
                    new HashMap<>(
                        Map.of(
                            ServiceFields.MODEL_ID,
                            modelId,
                            IbmWatsonxServiceFields.PROJECT_ID,
                            projectId,
                            ServiceFields.URL,
                            url,
                            IbmWatsonxServiceFields.API_VERSION,
                            apiVersion
                        )
                    ),
                    new HashMap<>(Map.of()),
                    createRandomChunkingSettingsMap(),
                    getSecretSettingsMap(apiKey)
                ),
                modelListener
            );
        }
    }

    public void testParseRequestConfig_ThrowsUnsupportedModelType() throws IOException {
        try (var service = createIbmWatsonxService()) {
            var failureListener = getModelListenerForException(
                ElasticsearchStatusException.class,
                "The [watsonxai] service does not support task type [sparse_embedding]"
            );

            service.parseRequestConfig(
                "id",
                TaskType.SPARSE_EMBEDDING,
                getRequestConfigMap(
                    new HashMap<>(Map.of(ServiceFields.MODEL_ID, "model")),
                    new HashMap<>(Map.of()),
                    getSecretSettingsMap("secret")
                ),
                failureListener
            );
        }
    }

    public void testParseRequestConfig_ThrowsWhenAnExtraKeyExistsInSecretSettingsMap() throws IOException {
        try (var service = createIbmWatsonxService()) {
            Map<String, Object> secretSettings = getSecretSettingsMap("secret");
            secretSettings.put("extra_key", "value");

            var config = getRequestConfigMap(
                new HashMap<>(
                    Map.of(
                        ServiceFields.MODEL_ID,
                        modelId,
                        IbmWatsonxServiceFields.PROJECT_ID,
                        projectId,
                        ServiceFields.URL,
                        url,
                        IbmWatsonxServiceFields.API_VERSION,
                        apiVersion
                    )
                ),
                getTaskSettingsMapEmpty(),
                secretSettings
            );

            var failureListener = getModelListenerForException(
                ElasticsearchStatusException.class,
                "Configuration contains settings [{extra_key=value}] unknown to the [watsonxai] service"
            );
            service.parseRequestConfig("id", TaskType.TEXT_EMBEDDING, config, failureListener);
        }
    }

    public void testParsePersistedConfigWithSecrets_CreatesAIbmWatsonxEmbeddingsModel() throws IOException {
        try (var service = createIbmWatsonxService()) {
            var persistedConfig = getPersistedConfigMap(
                new HashMap<>(
                    Map.of(
                        ServiceFields.MODEL_ID,
                        modelId,
                        IbmWatsonxServiceFields.PROJECT_ID,
                        projectId,
                        ServiceFields.URL,
                        url,
                        IbmWatsonxServiceFields.API_VERSION,
                        apiVersion
                    )
                ),
                getTaskSettingsMapEmpty(),
                getSecretSettingsMap(apiKey)
            );

            var model = service.parsePersistedConfigWithSecrets(
                "id",
                TaskType.TEXT_EMBEDDING,
                persistedConfig.config(),
                persistedConfig.secrets()
            );

            assertThat(model, instanceOf(IbmWatsonxEmbeddingsModel.class));

            var embeddingsModel = (IbmWatsonxEmbeddingsModel) model;
            assertThat(embeddingsModel.getServiceSettings().modelId(), is(modelId));
            assertThat(embeddingsModel.getServiceSettings().projectId(), is(projectId));
            assertThat(embeddingsModel.getServiceSettings().url(), is(URI.create(url)));
            assertThat(embeddingsModel.getServiceSettings().apiVersion(), is(apiVersion));
            assertThat(embeddingsModel.getTaskSettings(), is(EmptyTaskSettings.INSTANCE));
            assertThat(embeddingsModel.getSecretSettings().apiKey().toString(), is(apiKey));
            assertThat(embeddingsModel.getConfigurations().getChunkingSettings(), instanceOf(ChunkingSettings.class));
        }
    }

    public void testParsePersistedConfigWithSecrets_CreatesAIbmWatsonxEmbeddingsModelWhenChunkingSettingsProvided() throws IOException {
        try (var service = createIbmWatsonxService()) {
            var persistedConfig = getPersistedConfigMap(
                new HashMap<>(
                    Map.of(
                        ServiceFields.MODEL_ID,
                        modelId,
                        IbmWatsonxServiceFields.PROJECT_ID,
                        projectId,
                        ServiceFields.URL,
                        url,
                        IbmWatsonxServiceFields.API_VERSION,
                        apiVersion
                    )
                ),
                getTaskSettingsMapEmpty(),
                createRandomChunkingSettingsMap(),
                getSecretSettingsMap(apiKey)
            );

            var model = service.parsePersistedConfigWithSecrets(
                "id",
                TaskType.TEXT_EMBEDDING,
                persistedConfig.config(),
                persistedConfig.secrets()
            );

            assertThat(model, instanceOf(IbmWatsonxEmbeddingsModel.class));

            var embeddingsModel = (IbmWatsonxEmbeddingsModel) model;
            assertThat(embeddingsModel.getServiceSettings().modelId(), is(modelId));
            assertThat(embeddingsModel.getServiceSettings().projectId(), is(projectId));
            assertThat(embeddingsModel.getServiceSettings().url(), is(URI.create(url)));
            assertThat(embeddingsModel.getServiceSettings().apiVersion(), is(apiVersion));
            assertThat(embeddingsModel.getTaskSettings(), is(EmptyTaskSettings.INSTANCE));
            assertThat(embeddingsModel.getSecretSettings().apiKey().toString(), is(apiKey));
            assertThat(embeddingsModel.getConfigurations().getChunkingSettings(), instanceOf(ChunkingSettings.class));
        }
    }

    public void testParsePersistedConfigWithSecrets_DoesNotThrowWhenAnExtraKeyExistsInConfig() throws IOException {
        try (var service = createIbmWatsonxService()) {
            var persistedConfig = getPersistedConfigMap(
                new HashMap<>(
                    Map.of(
                        ServiceFields.MODEL_ID,
                        modelId,
                        IbmWatsonxServiceFields.PROJECT_ID,
                        projectId,
                        ServiceFields.URL,
                        url,
                        IbmWatsonxServiceFields.API_VERSION,
                        apiVersion
                    )
                ),
                getTaskSettingsMapEmpty(),
                getSecretSettingsMap(apiKey)
            );
            persistedConfig.config().put("extra_key", "value");

            var model = service.parsePersistedConfigWithSecrets(
                "id",
                TaskType.TEXT_EMBEDDING,
                persistedConfig.config(),
                persistedConfig.secrets()
            );

            assertThat(model, instanceOf(IbmWatsonxEmbeddingsModel.class));

            var embeddingsModel = (IbmWatsonxEmbeddingsModel) model;
            assertThat(embeddingsModel.getServiceSettings().modelId(), is(modelId));
            assertThat(embeddingsModel.getServiceSettings().projectId(), is(projectId));
            assertThat(embeddingsModel.getServiceSettings().url(), is(URI.create(url)));
            assertThat(embeddingsModel.getServiceSettings().apiVersion(), is(apiVersion));
            assertThat(embeddingsModel.getTaskSettings(), is(EmptyTaskSettings.INSTANCE));
            assertThat(embeddingsModel.getSecretSettings().apiKey(), is(apiKey));
        }
    }

    public void testParsePersistedConfigWithSecrets_DoesNotThrowWhenAnExtraKeyExistsInSecretsSettings() throws IOException {
        try (var service = createIbmWatsonxService()) {
            var secretSettingsMap = getSecretSettingsMap(apiKey);
            secretSettingsMap.put("extra_key", "value");

            var persistedConfig = getPersistedConfigMap(
                new HashMap<>(
                    Map.of(
                        ServiceFields.MODEL_ID,
                        modelId,
                        IbmWatsonxServiceFields.PROJECT_ID,
                        projectId,
                        ServiceFields.URL,
                        url,
                        IbmWatsonxServiceFields.API_VERSION,
                        apiVersion
                    )
                ),
                getTaskSettingsMapEmpty(),
                secretSettingsMap
            );

            var model = service.parsePersistedConfigWithSecrets(
                "id",
                TaskType.TEXT_EMBEDDING,
                persistedConfig.config(),
                persistedConfig.secrets()
            );

            assertThat(model, instanceOf(IbmWatsonxEmbeddingsModel.class));

            var embeddingsModel = (IbmWatsonxEmbeddingsModel) model;
            assertThat(embeddingsModel.getServiceSettings().modelId(), is(modelId));
            assertThat(embeddingsModel.getServiceSettings().projectId(), is(projectId));
            assertThat(embeddingsModel.getServiceSettings().url(), is(URI.create(url)));
            assertThat(embeddingsModel.getServiceSettings().apiVersion(), is(apiVersion));
            assertThat(embeddingsModel.getTaskSettings(), is(EmptyTaskSettings.INSTANCE));
            assertThat(embeddingsModel.getSecretSettings().apiKey().toString(), is(apiKey));
        }
    }

    public void testParsePersistedConfigWithSecrets_DoesNotThrowWhenAnExtraKeyExistsInServiceSettings() throws IOException {
        try (var service = createIbmWatsonxService()) {
            Map<String, Object> serviceSettingsMap = new HashMap<>(
                Map.of(
                    ServiceFields.MODEL_ID,
                    modelId,
                    IbmWatsonxServiceFields.PROJECT_ID,
                    projectId,
                    ServiceFields.URL,
                    url,
                    IbmWatsonxServiceFields.API_VERSION,
                    apiVersion
                )
            );
            serviceSettingsMap.put("extra_key", "value");

            var persistedConfig = getPersistedConfigMap(serviceSettingsMap, getTaskSettingsMapEmpty(), getSecretSettingsMap(apiKey));

            var model = service.parsePersistedConfigWithSecrets(
                "id",
                TaskType.TEXT_EMBEDDING,
                persistedConfig.config(),
                persistedConfig.secrets()
            );

            assertThat(model, instanceOf(IbmWatsonxEmbeddingsModel.class));

            var embeddingsModel = (IbmWatsonxEmbeddingsModel) model;
            assertThat(embeddingsModel.getServiceSettings().modelId(), is(modelId));
            assertThat(embeddingsModel.getServiceSettings().projectId(), is(projectId));
            assertThat(embeddingsModel.getServiceSettings().url(), is(URI.create(url)));
            assertThat(embeddingsModel.getServiceSettings().apiVersion(), is(apiVersion));
            assertThat(embeddingsModel.getTaskSettings(), is(EmptyTaskSettings.INSTANCE));
            assertThat(embeddingsModel.getSecretSettings().apiKey().toString(), is(apiKey));
        }
    }

    public void testParsePersistedConfigWithSecrets_DoesNotThrowWhenAnExtraKeyExistsInTaskSettings() throws IOException {
        var modelId = "model";
        var apiKey = "apiKey";

        try (var service = createIbmWatsonxService()) {
            Map<String, Object> taskSettings = getTaskSettingsMapEmpty();
            taskSettings.put("extra_key", "value");

            var persistedConfig = getPersistedConfigMap(
                new HashMap<>(
                    Map.of(
                        ServiceFields.MODEL_ID,
                        modelId,
                        IbmWatsonxServiceFields.PROJECT_ID,
                        projectId,
                        ServiceFields.URL,
                        url,
                        IbmWatsonxServiceFields.API_VERSION,
                        apiVersion
                    )
                ),
                taskSettings,
                getSecretSettingsMap(apiKey)
            );

            var model = service.parsePersistedConfigWithSecrets(
                "id",
                TaskType.TEXT_EMBEDDING,
                persistedConfig.config(),
                persistedConfig.secrets()
            );

            assertThat(model, instanceOf(IbmWatsonxEmbeddingsModel.class));

            var embeddingsModel = (IbmWatsonxEmbeddingsModel) model;
            assertThat(embeddingsModel.getServiceSettings().modelId(), is(modelId));
            assertThat(embeddingsModel.getServiceSettings().projectId(), is(projectId));
            assertThat(embeddingsModel.getServiceSettings().url(), is(URI.create(url)));
            assertThat(embeddingsModel.getServiceSettings().apiVersion(), is(apiVersion));
            assertThat(embeddingsModel.getTaskSettings(), is(EmptyTaskSettings.INSTANCE));
            assertThat(embeddingsModel.getSecretSettings().apiKey().toString(), is(apiKey));
        }
    }

    public void testParsePersistedConfig_CreatesAIbmWatsonxEmbeddingsModelWhenChunkingSettingsNotProvided() throws IOException {
        try (var service = createIbmWatsonxService()) {
            var persistedConfig = getPersistedConfigMap(
                new HashMap<>(
                    Map.of(
                        ServiceFields.MODEL_ID,
                        modelId,
                        IbmWatsonxServiceFields.PROJECT_ID,
                        projectId,
                        ServiceFields.URL,
                        url,
                        IbmWatsonxServiceFields.API_VERSION,
                        apiVersion
                    )
                ),
                getTaskSettingsMapEmpty(),
                null
            );

            var model = service.parsePersistedConfig("id", TaskType.TEXT_EMBEDDING, persistedConfig.config());

            assertThat(model, instanceOf(IbmWatsonxEmbeddingsModel.class));

            var embeddingsModel = (IbmWatsonxEmbeddingsModel) model;
            assertThat(embeddingsModel.getServiceSettings().modelId(), is(modelId));
            assertThat(embeddingsModel.getServiceSettings().projectId(), is(projectId));
            assertThat(embeddingsModel.getServiceSettings().url(), is(URI.create(url)));
            assertThat(embeddingsModel.getServiceSettings().apiVersion(), is(apiVersion));
            assertThat(embeddingsModel.getTaskSettings(), is(EmptyTaskSettings.INSTANCE));
            assertThat(embeddingsModel.getConfigurations().getChunkingSettings(), instanceOf(ChunkingSettings.class));
        }
    }

    public void testParsePersistedConfig_CreatesAIbmWatsonxEmbeddingsModelWhenChunkingSettingsProvided() throws IOException {
        try (var service = createIbmWatsonxService()) {
            var persistedConfig = getPersistedConfigMap(
                new HashMap<>(
                    Map.of(
                        ServiceFields.MODEL_ID,
                        modelId,
                        IbmWatsonxServiceFields.PROJECT_ID,
                        projectId,
                        ServiceFields.URL,
                        url,
                        IbmWatsonxServiceFields.API_VERSION,
                        apiVersion
                    )
                ),
                getTaskSettingsMapEmpty(),
                createRandomChunkingSettingsMap(),
                null
            );

            var model = service.parsePersistedConfig("id", TaskType.TEXT_EMBEDDING, persistedConfig.config());

            assertThat(model, instanceOf(IbmWatsonxEmbeddingsModel.class));

            var embeddingsModel = (IbmWatsonxEmbeddingsModel) model;
            assertThat(embeddingsModel.getServiceSettings().modelId(), is(modelId));
            assertThat(embeddingsModel.getServiceSettings().projectId(), is(projectId));
            assertThat(embeddingsModel.getServiceSettings().url(), is(URI.create(url)));
            assertThat(embeddingsModel.getServiceSettings().apiVersion(), is(apiVersion));
            assertThat(embeddingsModel.getTaskSettings(), is(EmptyTaskSettings.INSTANCE));
            assertThat(embeddingsModel.getConfigurations().getChunkingSettings(), instanceOf(ChunkingSettings.class));
        }
    }

    public void testInfer_ThrowsErrorWhenModelIsNotIbmWatsonxModel() throws IOException {
        var sender = mock(Sender.class);

        var factory = mock(HttpRequestSender.Factory.class);
        when(factory.createSender()).thenReturn(sender);

        var mockModel = getInvalidModel("model_id", "service_name");

        try (var service = new IbmWatsonxService(factory, createWithEmptySettings(threadPool), mockClusterServiceEmpty())) {
            PlainActionFuture<InferenceServiceResults> listener = new PlainActionFuture<>();
            service.infer(
                mockModel,
                null,
                null,
                null,
                List.of(""),
                false,
                new HashMap<>(),
                InputType.INTERNAL_INGEST,
                InferenceAction.Request.DEFAULT_TIMEOUT,
                listener
            );

            var thrownException = expectThrows(ElasticsearchStatusException.class, () -> listener.actionGet(TIMEOUT));
            MatcherAssert.assertThat(
                thrownException.getMessage(),
                is("The internal model was invalid, please delete the service [service_name] with id [model_id] and add it again.")
            );

            verify(factory, times(1)).createSender();
            verify(sender, times(1)).start();
        }

        verify(sender, times(1)).close();
        verifyNoMoreInteractions(factory);
        verifyNoMoreInteractions(sender);
    }

    public void testInfer_ThrowsErrorWhenInputTypeIsSpecified() throws IOException {
        var sender = mock(Sender.class);

        var factory = mock(HttpRequestSender.Factory.class);
        when(factory.createSender()).thenReturn(sender);

        var model = IbmWatsonxEmbeddingsModelTests.createModel(modelId, projectId, URI.create(url), apiVersion, apiKey, getUrl(webServer));

        try (var service = new IbmWatsonxService(factory, createWithEmptySettings(threadPool), mockClusterServiceEmpty())) {
            PlainActionFuture<InferenceServiceResults> listener = new PlainActionFuture<>();

            var thrownException = expectThrows(
                ValidationException.class,
                () -> service.infer(
                    model,
                    null,
                    null,
                    null,
                    List.of(""),
                    false,
                    new HashMap<>(),
                    InputType.INGEST,
                    InferenceAction.Request.DEFAULT_TIMEOUT,
                    listener
                )
            );
            MatcherAssert.assertThat(
                thrownException.getMessage(),
                is("Validation Failed: 1: Invalid input_type [ingest]. The input_type option is not supported by this service;")
            );

            verify(factory, times(1)).createSender();
            verify(sender, times(1)).start();
        }

        verify(sender, times(1)).close();
        verifyNoMoreInteractions(factory);
        verifyNoMoreInteractions(sender);
    }

    public void testInfer_SendsEmbeddingsRequest() throws IOException {
        var input = "input";

        var senderFactory = HttpRequestSenderTests.createSenderFactory(threadPool, clientManager);

        try (var service = new IbmWatsonxServiceWithoutAuth(senderFactory, createWithEmptySettings(threadPool))) {
            String responseJson = """
                {
                     "results": [
                        {
                            "embedding": [
                               0.0123,
                               -0.0123
                            ],
                           "input": "input"
                        }
                     ]
                }
                """;

            webServer.enqueue(new MockResponse().setResponseCode(200).setBody(responseJson));

            var model = IbmWatsonxEmbeddingsModelTests.createModel(
                modelId,
                projectId,
                URI.create(url),
                apiVersion,
                apiKey,
                getUrl(webServer)
            );
            PlainActionFuture<InferenceServiceResults> listener = new PlainActionFuture<>();
            service.infer(
                model,
                null,
                null,
                null,
                List.of(input),
                false,
                new HashMap<>(),
                InputType.INTERNAL_INGEST,
                InferenceAction.Request.DEFAULT_TIMEOUT,
                listener
            );
            var result = listener.actionGet(TIMEOUT);

            assertThat(result.asMap(), is(buildExpectationFloat(List.of(new float[] { 0.0123F, -0.0123F }))));
            assertThat(webServer.requests(), hasSize(1));
            assertThat(webServer.requests().get(0).getHeader(HttpHeaders.CONTENT_TYPE), Matchers.equalTo(XContentType.JSON.mediaType()));

            var requestMap = entityAsMap(webServer.requests().get(0).getBody());
            assertThat(requestMap, aMapWithSize(3));
            assertThat(requestMap, Matchers.is(Map.of("project_id", projectId, "inputs", List.of(input), "model_id", modelId)));
        }
    }

    public void testChunkedInfer_ChunkingSettingsNotSet() throws IOException {
        testChunkedInfer_Batches(null);
    }

    public void testChunkedInfer_ChunkingSettingsSet() throws IOException {
        testChunkedInfer_Batches(createRandomChunkingSettings());
    }

    private void testChunkedInfer_Batches(ChunkingSettings chunkingSettings) throws IOException {
        var input = List.of(new ChunkInferenceInput("a"), new ChunkInferenceInput("bb"));

        var senderFactory = HttpRequestSenderTests.createSenderFactory(threadPool, clientManager);

        try (var service = new IbmWatsonxServiceWithoutAuth(senderFactory, createWithEmptySettings(threadPool))) {
            String responseJson = """
                {
                     "results": [
                        {
                            "embedding": [
                               0.0123,
                               -0.0123
                            ],
                           "input": "foo"
                        },
                         {
                            "embedding": [
                               0.0456,
                               -0.0456
                            ],
                           "input": "bar"
                        }
                     ]
                 }
                """;

            webServer.enqueue(new MockResponse().setResponseCode(200).setBody(responseJson));

            var model = IbmWatsonxEmbeddingsModelTests.createModel(
                modelId,
                projectId,
                URI.create(url),
                apiVersion,
                apiKey,
                getUrl(webServer)
            );
            PlainActionFuture<List<ChunkedInference>> listener = new PlainActionFuture<>();
            service.chunkedInfer(
                model,
                null,
                input,
                new HashMap<>(),
                InputType.INTERNAL_INGEST,
                InferenceAction.Request.DEFAULT_TIMEOUT,
                listener
            );

            var results = listener.actionGet(TIMEOUT);
            assertThat(results, hasSize(2));

            // first result
            {
                assertThat(results.get(0), instanceOf(ChunkedInferenceEmbedding.class));
                var floatResult = (ChunkedInferenceEmbedding) results.get(0);
                assertThat(floatResult.chunks(), hasSize(1));
                assertEquals(new ChunkedInference.TextOffset(0, input.get(0).input().length()), floatResult.chunks().get(0).offset());
                assertThat(floatResult.chunks().get(0).embedding(), Matchers.instanceOf(TextEmbeddingFloatResults.Embedding.class));
                assertTrue(
                    Arrays.equals(
                        new float[] { 0.0123f, -0.0123f },
                        ((TextEmbeddingFloatResults.Embedding) floatResult.chunks().get(0).embedding()).values()
                    )
                );
            }

            // second result
            {
                assertThat(results.get(1), instanceOf(ChunkedInferenceEmbedding.class));
                var floatResult = (ChunkedInferenceEmbedding) results.get(1);
                assertThat(floatResult.chunks(), hasSize(1));
                assertEquals(new ChunkedInference.TextOffset(0, input.get(1).input().length()), floatResult.chunks().get(0).offset());
                assertThat(floatResult.chunks().get(0).embedding(), Matchers.instanceOf(TextEmbeddingFloatResults.Embedding.class));
                assertTrue(
                    Arrays.equals(
                        new float[] { 0.0456f, -0.0456f },
                        ((TextEmbeddingFloatResults.Embedding) floatResult.chunks().get(0).embedding()).values()
                    )
                );
            }

            assertThat(webServer.requests(), hasSize(1));
            assertThat(webServer.requests().get(0).getHeader(HttpHeaders.CONTENT_TYPE), Matchers.equalTo(XContentType.JSON.mediaType()));

            var requestMap = entityAsMap(webServer.requests().get(0).getBody());
            assertThat(requestMap, aMapWithSize(3));
            assertThat(requestMap, is(Map.of("project_id", projectId, "inputs", List.of("a", "bb"), "model_id", modelId)));
        }
    }

    public void testInfer_ResourceNotFound() throws IOException {
        var senderFactory = HttpRequestSenderTests.createSenderFactory(threadPool, clientManager);

        try (var service = new IbmWatsonxServiceWithoutAuth(senderFactory, createWithEmptySettings(threadPool))) {

            String responseJson = """
                {
                    "error": {
                        "message": "error"
                    }
                }
                """;
            webServer.enqueue(new MockResponse().setResponseCode(404).setBody(responseJson));

            var model = IbmWatsonxEmbeddingsModelTests.createModel(
                modelId,
                projectId,
                URI.create(url),
                apiVersion,
                apiKey,
                getUrl(webServer)
            );
            PlainActionFuture<InferenceServiceResults> listener = new PlainActionFuture<>();
            service.infer(
                model,
                null,
                null,
                null,
                List.of("abc"),
                false,
                new HashMap<>(),
                InputType.INTERNAL_INGEST,
                InferenceAction.Request.DEFAULT_TIMEOUT,
                listener
            );

            var error = expectThrows(ElasticsearchException.class, () -> listener.actionGet(TIMEOUT));
            assertThat(error.getMessage(), containsString("Resource not found at "));
            assertThat(error.getMessage(), containsString("Error message: [error]"));
            assertThat(webServer.requests(), hasSize(1));
        }
    }

    public void testUpdateModelWithEmbeddingDetails_InvalidModelProvided() throws IOException {
        var senderFactory = HttpRequestSenderTests.createSenderFactory(threadPool, clientManager);

        try (var service = new IbmWatsonxServiceWithoutAuth(senderFactory, createWithEmptySettings(threadPool))) {
            var model = OpenAiChatCompletionModelTests.createCompletionModel(
                randomAlphaOfLength(10),
                randomAlphaOfLength(10),
                randomAlphaOfLength(10),
                randomAlphaOfLength(10),
                randomAlphaOfLength(10)
            );
            assertThrows(
                ElasticsearchStatusException.class,
                () -> { service.updateModelWithEmbeddingDetails(model, randomNonNegativeInt()); }
            );
        }
    }

    public void testUpdateModelWithEmbeddingDetails_NullSimilarityInOriginalModel() throws IOException {
        testUpdateModelWithEmbeddingDetails_Successful(null);
    }

    public void testUpdateModelWithEmbeddingDetails_NonNullSimilarityInOriginalModel() throws IOException {
        testUpdateModelWithEmbeddingDetails_Successful(randomFrom(SimilarityMeasure.values()));
    }

    private void testUpdateModelWithEmbeddingDetails_Successful(SimilarityMeasure similarityMeasure) throws IOException {
        var senderFactory = HttpRequestSenderTests.createSenderFactory(threadPool, clientManager);

        try (var service = new IbmWatsonxServiceWithoutAuth(senderFactory, createWithEmptySettings(threadPool))) {
            var embeddingSize = randomNonNegativeInt();
            var model = IbmWatsonxEmbeddingsModelTests.createModel(
                randomAlphaOfLength(10),
                randomAlphaOfLength(10),
                randomAlphaOfLength(10),
                URI.create(randomAlphaOfLength(10)),
                randomAlphaOfLength(10),
                randomAlphaOfLength(10),
                randomNonNegativeInt(),
                similarityMeasure
            );

            Model updatedModel = service.updateModelWithEmbeddingDetails(model, embeddingSize);

            SimilarityMeasure expectedSimilarityMeasure = similarityMeasure == null ? SimilarityMeasure.DOT_PRODUCT : similarityMeasure;
            assertEquals(expectedSimilarityMeasure, updatedModel.getServiceSettings().similarity());
            assertEquals(embeddingSize, updatedModel.getServiceSettings().dimensions().intValue());
        }
    }

    public void testGetConfiguration() throws Exception {
        try (var service = createIbmWatsonxService()) {
            String content = XContentHelper.stripWhitespace("""
                {
                       "service": "watsonxai",
                       "name": "IBM watsonx",
                       "task_types": ["text_embedding", "completion", "chat_completion"],
                       "configurations": {
                           "project_id": {
                               "description": "",
                               "label": "Project ID",
                               "required": true,
                               "sensitive": false,
                               "updatable": false,
                               "type": "str",
                               "supported_task_types": ["text_embedding", "completion", "chat_completion"]
                           },
                           "model_id": {
                               "description": "The name of the model to use for the inference task.",
                               "label": "Model ID",
                               "required": true,
                               "sensitive": false,
                               "updatable": false,
                               "type": "str",
                               "supported_task_types": ["text_embedding", "completion", "chat_completion"]
                           },
                           "api_version": {
                               "description": "The IBM watsonx API version ID to use.",
                               "label": "API Version",
                               "required": true,
                               "sensitive": false,
                               "updatable": false,
                               "type": "str",
                               "supported_task_types": ["text_embedding", "completion", "chat_completion"]
                           },
                           "max_input_tokens": {
                               "description": "Allows you to specify the maximum number of tokens per input.",
                               "label": "Maximum Input Tokens",
                               "required": false,
                               "sensitive": false,
                               "updatable": false,
                               "type": "int",
                               "supported_task_types": ["text_embedding"]
                           },
                           "url": {
                               "description": "",
                               "label": "URL",
                               "required": true,
                               "sensitive": false,
                               "updatable": false,
                               "type": "str",
                               "supported_task_types": ["text_embedding", "completion", "chat_completion"]
                           }
                       }
                   }
                """);
            InferenceServiceConfiguration configuration = InferenceServiceConfiguration.fromXContentBytes(
                new BytesArray(content),
                XContentType.JSON
            );
            boolean humanReadable = true;
            BytesReference originalBytes = toShuffledXContent(configuration, XContentType.JSON, ToXContent.EMPTY_PARAMS, humanReadable);
            InferenceServiceConfiguration serviceConfiguration = service.getConfiguration();
            assertToXContentEquivalent(
                originalBytes,
                toXContent(serviceConfiguration, XContentType.JSON, humanReadable),
                XContentType.JSON
            );
        }
    }

    private static ActionListener<Model> getModelListenerForException(Class<?> exceptionClass, String expectedMessage) {
        return ActionListener.<Model>wrap((model) -> fail("Model parsing should have failed"), e -> {
            assertThat(e, Matchers.instanceOf(exceptionClass));
            assertThat(e.getMessage(), is(expectedMessage));
        });
    }

    private Map<String, Object> getRequestConfigMap(
        Map<String, Object> serviceSettings,
        Map<String, Object> taskSettings,
        Map<String, Object> chunkingSettings,
        Map<String, Object> secretSettings
    ) {
        var requestConfigMap = getRequestConfigMap(serviceSettings, taskSettings, secretSettings);
        requestConfigMap.put(ModelConfigurations.CHUNKING_SETTINGS, chunkingSettings);

        return requestConfigMap;
    }

    private Map<String, Object> getRequestConfigMap(
        Map<String, Object> serviceSettings,
        Map<String, Object> taskSettings,
        Map<String, Object> secretSettings
    ) {
        var builtServiceSettings = new HashMap<>();
        builtServiceSettings.putAll(serviceSettings);
        builtServiceSettings.putAll(secretSettings);

        return new HashMap<>(
            Map.of(ModelConfigurations.SERVICE_SETTINGS, builtServiceSettings, ModelConfigurations.TASK_SETTINGS, taskSettings)
        );
    }

    private IbmWatsonxService createIbmWatsonxService() {
        return new IbmWatsonxService(mock(HttpRequestSender.Factory.class), createWithEmptySettings(threadPool), mockClusterServiceEmpty());
    }

    private static class IbmWatsonxServiceWithoutAuth extends IbmWatsonxService {
        IbmWatsonxServiceWithoutAuth(HttpRequestSender.Factory factory, ServiceComponents serviceComponents) {
            super(factory, serviceComponents, mockClusterServiceEmpty());
        }

        @Override
        protected IbmWatsonxActionCreator getActionCreator(Sender sender, ServiceComponents serviceComponents) {
            return new IbmWatsonxActionCreatorWithoutAuth(getSender(), getServiceComponents());
        }
    }

    private static class IbmWatsonxActionCreatorWithoutAuth extends IbmWatsonxActionCreator {
        IbmWatsonxActionCreatorWithoutAuth(Sender sender, ServiceComponents serviceComponents) {
            super(sender, serviceComponents);
        }

        @Override
        protected IbmWatsonxEmbeddingsRequestManager getEmbeddingsRequestManager(
            IbmWatsonxEmbeddingsModel model,
            Truncator truncator,
            ThreadPool threadPool
        ) {
            return new IbmWatsonxEmbeddingsRequestManagerWithoutAuth(model, truncator, threadPool);
        }
    }

    private static class IbmWatsonxEmbeddingsRequestManagerWithoutAuth extends IbmWatsonxEmbeddingsRequestManager {
        IbmWatsonxEmbeddingsRequestManagerWithoutAuth(IbmWatsonxEmbeddingsModel model, Truncator truncator, ThreadPool threadPool) {
            super(model, truncator, threadPool);
        }

        @Override
        protected IbmWatsonxEmbeddingsRequest getEmbeddingRequest(
            Truncator truncator,
            Truncator.TruncationResult truncatedInput,
            IbmWatsonxEmbeddingsModel model
        ) {
            return new IbmWatsonxEmbeddingsWithoutAuthRequest(truncator, truncatedInput, model);
        }

    }

    private static class IbmWatsonxEmbeddingsWithoutAuthRequest extends IbmWatsonxEmbeddingsRequest {
        private static final String AUTH_HEADER_VALUE = "foo";

        IbmWatsonxEmbeddingsWithoutAuthRequest(Truncator truncator, Truncator.TruncationResult input, IbmWatsonxEmbeddingsModel model) {
            super(truncator, input, model);
        }

        @Override
        public void decorateWithAuth(HttpPost httpPost) {
            httpPost.setHeader(HttpHeaders.AUTHORIZATION, AUTH_HEADER_VALUE);
        }

        @Override
        public Request truncate() {
            IbmWatsonxEmbeddingsRequest embeddingsRequest = (IbmWatsonxEmbeddingsRequest) super.truncate();
            return new IbmWatsonxEmbeddingsWithoutAuthRequest(
                embeddingsRequest.truncator(),
                embeddingsRequest.truncationResult(),
                embeddingsRequest.model()
            );
        }
    }
}
