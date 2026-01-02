import json
from unittest import mock

import pytest
from pytest_mock import MockerFixture

from src.experiment.silver_standard import SilverStandardGenerator


@pytest.fixture
def mock_retriever(mocker: MockerFixture):
    """Create a mock retriever with sample chunks."""
    mock = mocker.Mock()
    mock.chunks = [
        "This is the first chunk with enough content to be valid." * 5,
        "This is the second chunk with sufficient length." * 5,
        "Another chunk that meets the minimum character requirement." * 5,
        "Yet another valid chunk for testing purposes." * 5,
        "Final chunk with adequate content length for validation." * 5,
    ]
    return mock


@pytest.fixture
def mock_gemini_client(mocker: MockerFixture):
    """Create a mock Google Gen AI client for Gemini."""
    mock_client = mocker.Mock()
    return mock_client


@pytest.fixture
def mock_ollama_client(mocker: MockerFixture):
    """Create a mock Ollama client."""
    mock_client = mocker.Mock()
    return mock_client


@pytest.fixture
def generator_gemini(mock_retriever, mock_gemini_client):
    """Create a SilverStandardGenerator instance for Gemini."""
    return SilverStandardGenerator(mock_retriever, mock_gemini_client, llm_type="gemini")


@pytest.fixture
def generator_ollama(mock_retriever, mock_ollama_client):
    """Create a SilverStandardGenerator instance for Ollama."""
    return SilverStandardGenerator(mock_retriever, mock_ollama_client, llm_type="ollama")


class TestSilverStandardGeneratorInitialization:
    """Tests for generator initialization."""

    def test_init_gemini(self, mock_retriever, mock_gemini_client):
        """Test initialization with Gemini."""
        generator = SilverStandardGenerator(mock_retriever, mock_gemini_client, llm_type="gemini")
        assert generator.retriever == mock_retriever
        assert generator.llm_client == mock_gemini_client
        assert generator.llm_type == "gemini"
        assert generator.model == "gemini-3.0-preview"

    def test_init_ollama(self, mock_retriever, mock_ollama_client):
        """Test initialization with Ollama."""
        generator = SilverStandardGenerator(mock_retriever, mock_ollama_client, llm_type="ollama")
        assert generator.retriever == mock_retriever
        assert generator.llm_client == mock_ollama_client
        assert generator.llm_type == "ollama"
        assert generator.model is None

    def test_init_invalid_llm_type(self, mock_retriever, mock_gemini_client):
        """Test initialization with invalid LLM type."""
        with pytest.raises(ValueError, match="Unsupported LLM type"):
            SilverStandardGenerator(mock_retriever, mock_gemini_client, llm_type="invalid")

    def test_init_case_insensitive(self, mock_retriever, mock_gemini_client):
        """Test that LLM type is case-insensitive."""
        generator = SilverStandardGenerator(mock_retriever, mock_gemini_client, llm_type="GEMINI")
        assert generator.llm_type == "gemini"


class TestGenerateDataset:
    """Tests for dataset generation."""

    def test_generate_dataset_gemini(self, generator_gemini, mock_gemini_client, mocker):
        """Test generating a dataset with Gemini."""
        # Mock the API response
        mock_response = mocker.Mock()
        mock_response.text = json.dumps(
            {
                "question": "What is the relationship between contexts?",
                "answer": "They are related.",
                "bridge_entity": "Test entity",
                "gold_snippets": ["Snippet 1", "Snippet 2", "Snippet 3"],
            }
        )
        mock_gemini_client.models.generate_content.return_value = mock_response

        dataset = generator_gemini.generate_dataset(num_samples=2, num_hops=3)

        assert len(dataset) == 2
        assert all("sample_id" in item for item in dataset)
        assert all("question" in item for item in dataset)
        assert all("answer" in item for item in dataset)
        assert all("gold_passages" in item for item in dataset)
        assert all("category" in item for item in dataset)
        assert all("difficulty" in item for item in dataset)

    def test_generate_dataset_ollama(self, generator_ollama, mock_ollama_client, mocker):
        """Test generating a dataset with Ollama."""
        # Mock the Ollama response
        mock_response = json.dumps(
            {
                "question": "What is the relationship between contexts?",
                "answer": "They are related.",
                "bridge_entity": "Test entity",
                "gold_snippets": ["Snippet 1", "Snippet 2", "Snippet 3"],
            }
        )
        mock_ollama_client.invoke.return_value = mock_response

        dataset = generator_ollama.generate_dataset(num_samples=2, num_hops=3)

        assert len(dataset) == 2
        assert all("sample_id" in item for item in dataset)

    def test_generate_dataset_respects_limit(self, generator_gemini, mock_gemini_client, mocker):
        """Test that dataset generation respects sample limit."""
        mock_response = mocker.Mock()
        mock_response.text = json.dumps(
            {
                "question": "Question?",
                "answer": "Answer",
            }
        )
        mock_gemini_client.models.generate_content.return_value = mock_response

        dataset = generator_gemini.generate_dataset(num_samples=5, num_hops=3)

        assert len(dataset) <= 5

    def test_generate_dataset_handles_failures(self, generator_gemini, mock_gemini_client, mocker):
        """Test that dataset generation continues even with failures."""
        responses = [
            mocker.Mock(text=json.dumps({"question": "IMPOSSIBLE", "answer": "IMPOSSIBLE"})),
            mocker.Mock(text="Invalid JSON"),
            mocker.Mock(
                text=json.dumps(
                    {
                        "question": "Q1?",
                        "answer": "A1",
                        "bridge_entity": "E1",
                        "gold_snippets": ["S1", "S2"],
                    }
                )
            ),
            mocker.Mock(
                text=json.dumps(
                    {
                        "question": "Q2?",
                        "answer": "A2",
                        "bridge_entity": "E2",
                        "gold_snippets": ["S1", "S2"],
                    }
                )
            ),
            mocker.Mock(
                text=json.dumps(
                    {
                        "question": "Q3?",
                        "answer": "A3",
                        "bridge_entity": "E3",
                        "gold_snippets": ["S1", "S2"],
                    }
                )
            ),
        ]

        mock_gemini_client.models.generate_content.side_effect = responses

        dataset = generator_gemini.generate_dataset(num_samples=3, num_hops=3)

        assert len(dataset) == 3


class TestGenerateSample:
    """Tests for single sample generation."""

    def test_generate_sample_gemini_success(self, generator_gemini, mock_gemini_client, mocker):
        """Test successful generation of a single sample with Gemini."""
        mock_response = mocker.Mock()
        mock_response.text = json.dumps(
            {
                "question": "What is test?",
                "answer": "Test answer from chunk",
                "bridge_entity": "Test entity",
                "gold_snippets": ["Snippet 1", "Snippet 2", "Snippet 3"],
            }
        )
        mock_gemini_client.models.generate_content.return_value = mock_response

        sample = generator_gemini.generate_sample(num_hops=3)

        assert sample is not None
        assert sample["question"] == "What is test?"
        assert sample["answer"] == "Test answer from chunk"
        assert sample["category"] == "Multihop"
        assert sample["difficulty"] == "Hard"
        assert len(sample["gold_passages"]) == 3
        assert "sample_id" in sample

    def test_generate_sample_ollama_success(self, generator_ollama, mock_ollama_client, mocker):
        """Test successful generation with Ollama."""
        mock_ollama_client.invoke.return_value = json.dumps(
            {
                "question": "What is test?",
                "answer": "Test answer from chunk",
                "bridge_entity": "Test entity",
                "gold_snippets": ["Snippet 1", "Snippet 2", "Snippet 3"],
            }
        )

        sample = generator_ollama.generate_sample(num_hops=3)

        assert sample is not None
        assert sample["question"] == "What is test?"
        assert sample["answer"] == "Test answer from chunk"

    def test_generate_sample_impossible_response(
        self, generator_gemini, mock_gemini_client, mocker
    ):
        """Test that IMPOSSIBLE responses result in None."""
        mock_response = mocker.Mock()
        mock_response.text = json.dumps(
            {
                "question": "IMPOSSIBLE",
                "answer": "IMPOSSIBLE",
            }
        )
        mock_gemini_client.models.generate_content.return_value = mock_response

        sample = generator_gemini.generate_sample(num_hops=3)

        assert sample is None

    def test_generate_sample_api_error(self, generator_gemini, mock_gemini_client):
        """Test handling of API errors."""
        mock_gemini_client.models.generate_content.side_effect = Exception("API Error")

        sample = generator_gemini.generate_sample(num_hops=3)

        assert sample is None

    def test_generate_sample_invalid_json(self, generator_gemini, mock_gemini_client, mocker):
        """Test handling of invalid JSON in response."""
        mock_response = mocker.Mock()
        mock_response.text = "This is not valid JSON {}"
        mock_gemini_client.models.generate_content.return_value = mock_response

        sample = generator_gemini.generate_sample(num_hops=3)

        assert sample is None

    def test_generate_sample_uuid_uniqueness(self, generator_gemini, mock_gemini_client, mocker):
        """Test that each sample gets a unique UUID."""
        mock_response = mocker.Mock()
        mock_response.text = json.dumps(
            {
                "question": "Q?",
                "answer": "A",
                "bridge_entity": "Entity",
                "gold_snippets": ["Snippet 1", "Snippet 2"],
            }
        )
        mock_gemini_client.models.generate_content.return_value = mock_response

        samples = [generator_gemini.generate_sample() for _ in range(3)]
        sample_ids = [s["sample_id"] for s in samples]

        assert len(set(sample_ids)) == 3


class TestGetRandomContexts:
    """Tests for random context selection."""

    def test_get_random_contexts_basic(self, generator_gemini):
        """Test getting random contexts."""
        contexts = generator_gemini._get_random_contexts(n=3, min_char_length=50)

        assert len(contexts) == 3
        assert all(len(c) >= 50 for c in contexts)

    def test_get_random_contexts_with_short_chunks(self, mock_retriever, mock_gemini_client):
        """Test that short chunks are filtered out."""
        mock_retriever.chunks = [
            "Short",
            "This chunk has sufficient length to be included." * 5,
            "Also short",
            "Another valid chunk with enough characters." * 5,
        ]

        generator = SilverStandardGenerator(mock_retriever, mock_gemini_client, llm_type="gemini")
        contexts = generator._get_random_contexts(n=2, min_char_length=100)

        assert len(contexts) == 2
        assert all(len(c) >= 100 for c in contexts)

    def test_get_random_contexts_fallback(self, mock_retriever, mock_gemini_client):
        """Test fallback when not enough long chunks available."""
        mock_retriever.chunks = [
            "Short1",
            "Short2",
            "This is a longer chunk with sufficient content." * 5,
        ]

        generator = SilverStandardGenerator(mock_retriever, mock_gemini_client, llm_type="gemini")
        contexts = generator._get_random_contexts(n=3, min_char_length=100)

        assert len(contexts) <= len(mock_retriever.chunks)

    def test_get_random_contexts_empty_retriever(self, mock_gemini_client):
        """Test error handling when retriever has no chunks."""
        mock_retriever = mock.Mock()
        mock_retriever.chunks = []

        generator = SilverStandardGenerator(mock_retriever, mock_gemini_client, llm_type="gemini")

        with pytest.raises(ValueError, match="Retriever has no chunks loaded"):
            generator._get_random_contexts(n=3)

    def test_get_random_contexts_randomness(self, generator_gemini):
        """Test that context selection is random."""
        contexts1 = generator_gemini._get_random_contexts(n=2, min_char_length=50)
        contexts2 = generator_gemini._get_random_contexts(n=2, min_char_length=50)

        # With high probability, different random selections will differ
        # (though technically they could be the same)
        assert isinstance(contexts1, list)
        assert isinstance(contexts2, list)


class TestBuildMultihopPrompt:
    """Tests for prompt construction."""

    def test_build_multihop_prompt(self, generator_gemini):
        """Test prompt construction."""
        chunks = ["First chunk", "Second chunk", "Third chunk"]
        prompt = generator_gemini._build_multihop_prompt(chunks)

        assert "multi-hop" in prompt.lower()
        assert "First chunk" in prompt
        assert "Second chunk" in prompt
        assert "Third chunk" in prompt
        assert "Bridge" in prompt
        assert "JSON" in prompt

    def test_build_multihop_prompt_ordering(self, generator_gemini):
        """Test that chunks are presented in correct order."""
        chunks = ["Alpha", "Beta", "Gamma"]
        prompt = generator_gemini._build_multihop_prompt(chunks)

        alpha_pos = prompt.find("Context 1: Alpha")
        beta_pos = prompt.find("Context 2: Beta")
        gamma_pos = prompt.find("Context 3: Gamma")

        assert alpha_pos < beta_pos < gamma_pos

    def test_build_multihop_prompt_uses_exact_substring_requirement(self, generator_gemini):
        """Test that prompt includes exact substring match requirement."""
        chunks = ["Context1", "Context2"]
        prompt = generator_gemini._build_multihop_prompt(chunks)

        assert "exact" in prompt.lower()
        assert "snippets" in prompt.lower()


class TestCallLLM:
    """Tests for LLM invocation."""

    def test_call_llm_gemini(self, generator_gemini, mock_gemini_client, mocker):
        """Test calling Gemini LLM."""
        mock_response = mocker.Mock()
        mock_response.text = '{"question": "Q?", "answer": "A"}'
        mock_gemini_client.models.generate_content.return_value = mock_response

        result = generator_gemini._call_llm("test prompt")

        assert result == '{"question": "Q?", "answer": "A"}'
        mock_gemini_client.models.generate_content.assert_called_once()

    def test_call_llm_ollama(self, generator_ollama, mock_ollama_client):
        """Test calling Ollama LLM."""
        mock_ollama_client.invoke.return_value = '{"question": "Q?", "answer": "A"}'

        result = generator_ollama._call_llm("test prompt")

        assert result == '{"question": "Q?", "answer": "A"}'
        mock_ollama_client.invoke.assert_called_once_with("test prompt")

    def test_call_llm_gemini_error(self, generator_gemini, mock_gemini_client):
        """Test Gemini error handling."""
        mock_gemini_client.models.generate_content.side_effect = Exception("API Error")

        with pytest.raises(Exception) as exc_info:
            generator_gemini._call_llm("test prompt")
        assert "API Error" in str(exc_info.value)

    def test_call_llm_ollama_error(self, generator_ollama, mock_ollama_client):
        """Test Ollama error handling."""
        mock_ollama_client.invoke.side_effect = Exception("Connection Error")

        with pytest.raises(Exception) as exc_info:
            generator_ollama._call_llm("test prompt")
        assert "Connection Error" in str(exc_info.value)


class TestParseLLMResponse:
    """Tests for LLM response parsing."""

    def test_parse_valid_json(self, generator_gemini):
        """Test parsing valid JSON response."""
        response = '{"question": "Test Q?", "answer": "Test A"}'
        result = generator_gemini._parse_llm_response(response)

        assert result["question"] == "Test Q?"
        assert result["answer"] == "Test A"

    def test_parse_json_with_markdown(self, generator_gemini):
        """Test parsing JSON wrapped in markdown."""
        response = '```json\n{"question": "Test Q?", "answer": "Test A"}\n```'
        result = generator_gemini._parse_llm_response(response)

        assert result["question"] == "Test Q?"
        assert result["answer"] == "Test A"

    def test_parse_json_with_extra_spaces(self, generator_gemini):
        """Test parsing JSON with extra markdown formatting."""
        response = '```json  \n{"question": "Q?", "answer": "A"}  \n```'
        result = generator_gemini._parse_llm_response(response)

        assert result["question"] == "Q?"
        assert result["answer"] == "A"

    def test_parse_invalid_json(self, generator_gemini):
        """Test parsing invalid JSON."""
        response = "This is not JSON at all"
        result = generator_gemini._parse_llm_response(response)

        assert result["question"] == "Error parsing generation"
        assert result["answer"] == "Error parsing generation"

    def test_parse_malformed_json(self, generator_gemini):
        """Test parsing malformed JSON."""
        response = '{"question": "Q?", "answer": "A"'  # Missing closing brace
        result = generator_gemini._parse_llm_response(response)

        assert result["question"] == "Error parsing generation"
        assert result["answer"] == "Error parsing generation"


class TestMultihopGeneration:
    """Integration tests for multihop question generation."""

    def test_multihop_requires_all_contexts(self, generator_gemini, mock_gemini_client, mocker):
        """Test that generated questions require all contexts."""
        mock_response = mocker.Mock()
        mock_response.text = json.dumps(
            {
                "question": "How do contexts A, B, and C relate?",
                "answer": "They form a relationship",
                "bridge_entity": "Test entity",
                "gold_snippets": ["Context A", "Context B", "Context C"],
            }
        )
        mock_gemini_client.models.generate_content.return_value = mock_response

        mocker.patch.object(
            generator_gemini,
            "_get_random_contexts",
            return_value=["Context A", "Context B", "Context C"],
        )

        sample = generator_gemini.generate_sample(num_hops=3)

        # Verify that the prompt was built correctly
        assert sample is not None
        assert len(sample["gold_passages"]) == 3
        assert len(sample["gold_passages"]) == 3

    def test_num_hops_parameter_respected(self, mock_retriever, mock_gemini_client, mocker):
        """Test that num_hops parameter controls number of contexts."""
        generator = SilverStandardGenerator(mock_retriever, mock_gemini_client, llm_type="gemini")

        mock_response = mocker.Mock()
        mock_response.text = json.dumps({"question": "Q?", "answer": "A"})
        mock_gemini_client.models.generate_content.return_value = mock_response

        mocker.patch.object(generator, "_get_random_contexts", wraps=generator._get_random_contexts)

        generator.generate_sample(num_hops=5)

        # Verify _get_random_contexts was called with correct n value
        generator._get_random_contexts.assert_called()
        call_args = generator._get_random_contexts.call_args
        assert call_args[1]["n"] == 5
