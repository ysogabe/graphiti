#!/usr/bin/env python3
"""Unit tests for service factory provider detection and client routing."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Add the src directory to the path (mirrors the other factory tests)
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from graphiti_core.embedder.gemini import GeminiEmbedder
from graphiti_core.llm_client import OpenAIClient
from graphiti_core.llm_client.azure_openai_client import AzureOpenAILLMClient
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

from config.schema import (
    AzureOpenAIProviderConfig,
    EmbedderConfig,
    EmbedderProvidersConfig,
    GeminiProviderConfig,
    LLMConfig,
    LLMProvidersConfig,
    OpenAIProviderConfig,
)
from services.factories import (
    EmbedderFactory,
    LLMClientFactory,
    is_non_openai_provider,
    reasoning_effort_for_model,
)


class TestIsNonOpenAIProvider:
    """Tests for the base_url-based provider detection."""

    @pytest.mark.parametrize(
        'base_url',
        [
            None,
            '',
            'https://api.openai.com/v1',
            'https://api.openai.com',
            'https://my-resource.openai.azure.com',
        ],
    )
    def test_official_or_unset_is_openai(self, base_url):
        """Unset, empty, or official OpenAI/Azure endpoints are treated as OpenAI."""
        assert is_non_openai_provider(base_url) is False

    @pytest.mark.parametrize(
        'base_url',
        [
            'http://localhost:11434/v1',  # Ollama
            'http://localhost:1234/v1',  # LM Studio
            'http://localhost:8000/v1',  # vLLM
            'https://my-proxy.internal/v1',
        ],
    )
    def test_compatible_providers_are_non_openai(self, base_url):
        """OpenAI-compatible third-party endpoints are detected as non-OpenAI."""
        assert is_non_openai_provider(base_url) is True


class TestLLMClientFactoryRouting:
    """Tests that the factory selects the right client based on base_url."""

    @staticmethod
    def _config(api_url: str) -> LLMConfig:
        return LLMConfig(
            provider='openai',
            model='gpt-5.5',
            providers=LLMProvidersConfig(
                openai=OpenAIProviderConfig(api_key='test-key', api_url=api_url)
            ),
        )

    def test_official_openai_uses_openai_client(self):
        client = LLMClientFactory.create(self._config('https://api.openai.com/v1'))
        assert isinstance(client, OpenAIClient)
        assert not isinstance(client, OpenAIGenericClient)

    def test_ollama_uses_generic_client(self):
        client = LLMClientFactory.create(self._config('http://localhost:11434/v1'))
        assert isinstance(client, OpenAIGenericClient)

    def test_generic_client_uses_default_structured_output_mode(self):
        client = LLMClientFactory.create(self._config('http://localhost:11434/v1'))
        assert isinstance(client, OpenAIGenericClient)
        assert client.structured_output_mode == 'json_schema'

    def test_generic_client_uses_configured_structured_output_mode(self):
        config = self._config('http://localhost:11434/v1')
        config.structured_output_mode = 'json_object'

        client = LLMClientFactory.create(config)

        assert isinstance(client, OpenAIGenericClient)
        assert client.structured_output_mode == 'json_object'

    def test_generic_client_uses_configured_reasoning_effort(self):
        # b.ai (and any non-OpenAI-compatible endpoint) must forward a configured
        # reasoning_effort so reasoning models (gpt-5.6-luna etc.) can be tuned.
        config = self._config('https://api.b.ai/v1')
        config.reasoning_effort = 'high'

        client = LLMClientFactory.create(config)

        assert isinstance(client, OpenAIGenericClient)
        assert client.reasoning_effort == 'high'

    def test_generic_client_defaults_reasoning_effort_to_none(self):
        config = self._config('https://api.b.ai/v1')

        client = LLMClientFactory.create(config)

        assert isinstance(client, OpenAIGenericClient)
        assert client.reasoning_effort is None


class TestLLMClientReasoningEffort:
    """The OpenAI factory selects reasoning effort by model family."""

    @staticmethod
    def _config(model: str) -> LLMConfig:
        return LLMConfig(
            provider='openai',
            model=model,
            providers=LLMProvidersConfig(
                openai=OpenAIProviderConfig(api_key='test-key', api_url='https://api.openai.com/v1')
            ),
        )

    def test_gpt_5_5_uses_reasoning_none(self):
        """gpt-5.5 (the default) runs with reasoning off."""
        client = LLMClientFactory.create(self._config('gpt-5.5'))
        assert isinstance(client, OpenAIClient)
        assert client.reasoning == 'none'

    def test_earlier_reasoning_model_uses_minimal(self):
        """Earlier gpt-5 reasoning models keep the historical 'minimal' floor."""
        client = LLMClientFactory.create(self._config('gpt-5'))
        assert isinstance(client, OpenAIClient)
        assert client.reasoning == 'minimal'


class TestReasoningEffortForModel:
    """The shared effort selector used by both the OpenAI and Azure branches."""

    @pytest.mark.parametrize(
        ('model', 'expected'),
        [
            ('gpt-5.5', 'none'),
            ('gpt-5.5-2026-04-23', 'none'),
            ('gpt-5', 'minimal'),
            ('gpt-5-mini', 'minimal'),
            ('gpt-5.4-mini', 'minimal'),
            ('o1', 'minimal'),
            ('o3-mini', 'minimal'),
            ('gpt-4.1', None),
            ('gpt-4o-mini', None),
        ],
    )
    def test_effort_selection(self, model, expected):
        assert reasoning_effort_for_model(model) == expected


class TestAzureReasoningEffort:
    """The Azure OpenAI branch applies the same model-tied reasoning effort."""

    @staticmethod
    def _config(model: str) -> LLMConfig:
        return LLMConfig(
            provider='azure_openai',
            model=model,
            providers=LLMProvidersConfig(
                azure_openai=AzureOpenAIProviderConfig(
                    api_key='test-key',
                    api_url='https://example.openai.azure.com',
                )
            ),
        )

    def test_azure_gpt_5_5_uses_reasoning_none(self):
        client = LLMClientFactory.create(self._config('gpt-5.5'))
        assert isinstance(client, AzureOpenAILLMClient)
        assert client.reasoning == 'none'

    def test_azure_non_reasoning_model_sends_no_effort(self):
        client = LLMClientFactory.create(self._config('gpt-4.1'))
        assert isinstance(client, AzureOpenAILLMClient)
        assert client.reasoning is None


class TestGeminiEmbedderVertexAI:
    """The Gemini embedder must forward Vertex AI project/location when configured."""

    @staticmethod
    def _config() -> EmbedderConfig:
        return EmbedderConfig(
            provider='gemini',
            model='text-embedding-005',
            dimensions=768,
            providers=EmbedderProvidersConfig(
                gemini=GeminiProviderConfig(
                    api_key='vertex-key', project_id='proj-1', location='global', vertexai=True
                )
            ),
        )

    @patch('google.genai.Client')
    def test_gemini_embedder_uses_vertex_mode(self, mock_client):
        embedder = EmbedderFactory.create(self._config())

        assert isinstance(embedder, GeminiEmbedder)
        assert embedder.config.vertexai is True
        assert embedder.config.project_id == 'proj-1'
        assert embedder.config.location == 'global'
        # The client is built for Vertex AI (vertexai=True) so requests hit aiplatform.
        mock_client.assert_called_once_with(
            vertexai=True, project='proj-1', location='global', api_key='vertex-key'
        )
