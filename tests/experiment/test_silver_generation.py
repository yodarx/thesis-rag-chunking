"""
Unit tests for silver standard generation.
Tests the SilverStandardGenerator and load_documents_from_input_file functions.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.experiment.silver_generation import load_documents_from_input_file
from src.experiment.silver_standard import SilverStandardGenerator


class TestLoadDocumentsFromInputFile:
    """Tests for load_documents_from_input_file function."""

    def test_load_documents_document_text_field(self, tmp_path):
        """Test loading documents with 'document_text' field."""
        input_file = tmp_path / "test.jsonl"
        documents = [
            {"document_text": "Document 1", "metadata": "test"},
            {"document_text": "Document 2", "metadata": "test"},
        ]
        with open(input_file, "w") as f:
            for doc in documents:
                f.write(json.dumps(doc) + "\n")

        result = load_documents_from_input_file(str(input_file))
        assert len(result) == 2
        assert result[0] == "Document 1"
        assert result[1] == "Document 2"

    def test_load_documents_text_field(self, tmp_path):
        """Test loading documents with 'text' field."""
        input_file = tmp_path / "test.jsonl"
        documents = [
            {"text": "Text 1", "id": "1"},
            {"text": "Text 2", "id": "2"},
        ]
        with open(input_file, "w") as f:
            for doc in documents:
                f.write(json.dumps(doc) + "\n")

        result = load_documents_from_input_file(str(input_file))
        assert len(result) == 2
        assert result[0] == "Text 1"
        assert result[1] == "Text 2"

    def test_load_documents_content_field(self, tmp_path):
        """Test loading documents with 'content' field."""
        input_file = tmp_path / "test.jsonl"
        documents = [
            {"content": "Content 1"},
            {"content": "Content 2"},
        ]
        with open(input_file, "w") as f:
            for doc in documents:
                f.write(json.dumps(doc) + "\n")

        result = load_documents_from_input_file(str(input_file))
        assert len(result) == 2
        assert result[0] == "Content 1"
        assert result[1] == "Content 2"

    def test_load_documents_document_field(self, tmp_path):
        """Test loading documents with 'document' field."""
        input_file = tmp_path / "test.jsonl"
        documents = [
            {"document": "Doc 1"},
            {"document": "Doc 2"},
        ]
        with open(input_file, "w") as f:
            for doc in documents:
                f.write(json.dumps(doc) + "\n")

        result = load_documents_from_input_file(str(input_file))
        assert len(result) == 2
        assert result[0] == "Doc 1"
        assert result[1] == "Doc 2"

    def test_load_documents_priority_order(self, tmp_path):
        """Test that document_text takes priority over other fields."""
        input_file = tmp_path / "test.jsonl"
        # Has both document_text and text - should use document_text
        doc = {"document_text": "Priority", "text": "Secondary"}
        with open(input_file, "w") as f:
            f.write(json.dumps(doc) + "\n")

        result = load_documents_from_input_file(str(input_file))
        assert result[0] == "Priority"

    def test_load_documents_nonexistent_file(self):
        """Test loading from nonexistent file returns empty list."""
        result = load_documents_from_input_file("/nonexistent/path/file.jsonl")
        assert result == []

    def test_load_documents_invalid_json_line(self, tmp_path):
        """Test that invalid JSON lines are skipped."""
        input_file = tmp_path / "test.jsonl"
        with open(input_file, "w") as f:
            f.write(json.dumps({"document_text": "Valid 1"}) + "\n")
            f.write("Invalid JSON line\n")
            f.write(json.dumps({"document_text": "Valid 2"}) + "\n")

        result = load_documents_from_input_file(str(input_file))
        assert len(result) == 2
        assert result[0] == "Valid 1"
        assert result[1] == "Valid 2"

    def test_load_documents_empty_file(self, tmp_path):
        """Test loading from empty file."""
        input_file = tmp_path / "empty.jsonl"
        input_file.write_text("")

        result = load_documents_from_input_file(str(input_file))
        assert result == []

    def test_load_documents_empty_lines(self, tmp_path):
        """Test that empty lines are skipped."""
        input_file = tmp_path / "test.jsonl"
        with open(input_file, "w") as f:
            f.write(json.dumps({"document_text": "Doc 1"}) + "\n")
            f.write("\n")
            f.write("\n")
            f.write(json.dumps({"document_text": "Doc 2"}) + "\n")

        result = load_documents_from_input_file(str(input_file))
        assert len(result) == 2
        assert result[0] == "Doc 1"
        assert result[1] == "Doc 2"

    def test_load_documents_no_recognized_field(self, tmp_path):
        """Test that entries without recognized fields are skipped."""
        input_file = tmp_path / "test.jsonl"
        with open(input_file, "w") as f:
            f.write(json.dumps({"id": "1"}) + "\n")
            f.write(json.dumps({"document_text": "Valid"}) + "\n")
            f.write(json.dumps({"random_field": "value"}) + "\n")

        result = load_documents_from_input_file(str(input_file))
        assert len(result) == 1
        assert result[0] == "Valid"


class TestSilverStandardGenerator:
    """Tests for SilverStandardGenerator class."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        return Mock()

    @pytest.fixture
    def sample_documents(self):
        """Provide sample documents for testing."""
        return [
            "This is a long document about Company A. It was founded in 1995 and specializes in software development and enterprise solutions. Company A is known for its innovative approach.",
            "Company A produces innovative software solutions for enterprise clients worldwide. Their products are used by thousands of companies in the Fortune 500.",
            "The company was established to solve complex problems in data management systems. Over the years, they have become leaders in their field with significant market presence.",
        ]

    @pytest.fixture
    def generator(self, mock_llm_client, sample_documents):
        """Create a generator instance."""
        return SilverStandardGenerator(mock_llm_client, sample_documents)

    def test_initialization(self, mock_llm_client, sample_documents):
        """Test generator initialization."""
        gen = SilverStandardGenerator(mock_llm_client, sample_documents)
        assert gen.llm_client == mock_llm_client
        assert gen.documents == sample_documents
        assert gen.model == "gemini-3-pro-preview"

    def test_get_random_contexts(self, generator, sample_documents):
        """Test _get_random_contexts returns correct number of contexts."""
        contexts = generator._get_random_contexts(n=2)
        assert len(contexts) == 2
        assert all(ctx in sample_documents for ctx in contexts)

    def test_get_random_contexts_min_length_filter(self, mock_llm_client):
        """Test _get_random_contexts filters by minimum length."""
        documents = [
            "Short",  # Too short
            "This is a much longer document that should meet the minimum character requirement.",
            "Also short",  # Too short
            "Another reasonably long document about some topic.",
        ]
        gen = SilverStandardGenerator(mock_llm_client, documents)
        contexts = gen._get_random_contexts(n=2, min_char_length=30)
        assert len(contexts) == 2
        assert all(len(ctx) >= 30 for ctx in contexts)

    def test_get_random_contexts_insufficient_documents(self, mock_llm_client):
        """Test _get_random_contexts handles case with fewer docs than requested."""
        documents = [
            "Document 1 that is long enough.",
            "Document 2 that is long enough.",
        ]
        gen = SilverStandardGenerator(mock_llm_client, documents)
        contexts = gen._get_random_contexts(n=5)
        assert len(contexts) <= 2

    def test_build_multihop_prompt(self, generator):
        """Test that prompt is built correctly."""
        contexts = ["Context A content", "Context B content"]
        prompt = generator._build_multihop_prompt(contexts)
        assert "Context A content" in prompt
        assert "Context B content" in prompt
        assert "multi-hop" in prompt.lower()
        assert "bridge" in prompt.lower()

    def test_parse_llm_response_valid_json(self, generator):
        """Test parsing valid JSON response."""
        response = json.dumps({
            "bridge_entity": "Company A",
            "question": "What does Company A do?",
            "answer": "Software solutions",
            "gold_snippets": ["snippet1", "snippet2"],
        })
        result = generator._parse_llm_response(response)
        assert result["bridge_entity"] == "Company A"
        assert result["question"] == "What does Company A do?"

    def test_parse_llm_response_json_with_markdown(self, generator):
        """Test parsing JSON response with markdown code blocks."""
        response = """```json
{
    "bridge_entity": "Entity",
    "question": "Question?",
    "answer": "Answer",
    "gold_snippets": ["s1", "s2"]
}
```"""
        result = generator._parse_llm_response(response)
        assert result["bridge_entity"] == "Entity"
        assert result["question"] == "Question?"

    def test_parse_llm_response_invalid_json(self, generator):
        """Test parsing invalid JSON returns error dict."""
        response = "This is not JSON"
        result = generator._parse_llm_response(response)
        assert result["question"] == "Error parsing generation"
        assert result["answer"] == "Error parsing generation"

    def test_call_llm_calls_gemini(self, generator, mock_llm_client):
        """Test that _call_llm delegates to _call_gemini."""
        mock_response = Mock()
        mock_response.text = '{"question": "test"}'
        mock_llm_client.models.generate_content.return_value = mock_response

        result = generator._call_llm("test prompt")
        assert result == '{"question": "test"}'
        mock_llm_client.models.generate_content.assert_called_once()

    def test_call_gemini_success(self, generator, mock_llm_client):
        """Test successful Gemini API call."""
        mock_response = Mock()
        mock_response.text = '{"test": "response"}'
        mock_llm_client.models.generate_content.return_value = mock_response

        result = generator._call_gemini("prompt")
        assert result == '{"test": "response"}'
        mock_llm_client.models.generate_content.assert_called_once_with(
            model="gemini-3-pro-preview",
            contents="prompt",
        )

    def test_call_gemini_api_error(self, generator, mock_llm_client):
        """Test Gemini API error is raised."""
        mock_llm_client.models.generate_content.side_effect = Exception("API Error")

        with pytest.raises(Exception, match="API Error"):
            generator._call_gemini("prompt")

    def test_generate_sample_success(self, generator, mock_llm_client, mocker):
        """Test successful generation of a single sample."""
        # Mock the LLM response
        response_json = {
            "bridge_entity": "Company A",
            "question": "What industry is Company A in?",
            "answer": "Software",
            "gold_snippets": ["Company A specializes in software", "Software solutions"],
        }
        mock_response = Mock()
        mock_response.text = json.dumps(response_json)
        mock_llm_client.models.generate_content.return_value = mock_response

        # Mock random.sample to return consistent contexts
        mocker.patch("random.sample", return_value=[
            generator.documents[0],
            generator.documents[1],
        ])

        sample = generator.generate_sample(num_hops=2)
        assert sample is not None
        assert sample["question"] == "What industry is Company A in?"
        assert sample["answer"] == "Software"
        assert sample["category"] == "Multihop"
        assert sample["difficulty"] == "Hard"
        assert "sample_id" in sample
        assert "gold_passages" in sample
        assert "metadata" in sample

    def test_generate_sample_impossible_response(self, generator, mock_llm_client, mocker):
        """Test that IMPOSSIBLE responses are rejected."""
        response_json = {
            "bridge_entity": "N/A",
            "question": "IMPOSSIBLE",
            "answer": "IMPOSSIBLE",
            "gold_snippets": [],
        }
        mock_response = Mock()
        mock_response.text = json.dumps(response_json)
        mock_llm_client.models.generate_content.return_value = mock_response

        mocker.patch("random.sample", return_value=[
            generator.documents[0],
            generator.documents[1],
        ])

        sample = generator.generate_sample()
        assert sample is None

    def test_generate_sample_empty_snippets(self, generator, mock_llm_client, mocker):
        """Test that empty snippets are rejected."""
        response_json = {
            "bridge_entity": "Something",
            "question": "A question",
            "answer": "An answer",
            "gold_snippets": [],
        }
        mock_response = Mock()
        mock_response.text = json.dumps(response_json)
        mock_llm_client.models.generate_content.return_value = mock_response

        mocker.patch("random.sample", return_value=[
            generator.documents[0],
            generator.documents[1],
        ])

        sample = generator.generate_sample()
        assert sample is None

    def test_generate_sample_exception_handling(self, generator, mock_llm_client, mocker):
        """Test that exceptions during generation are caught."""
        mock_llm_client.models.generate_content.side_effect = Exception("LLM Error")
        mocker.patch("random.sample", return_value=[
            generator.documents[0],
            generator.documents[1],
        ])

        sample = generator.generate_sample()
        assert sample is None

    def test_generate_dataset_success(self, generator, mock_llm_client, mocker):
        """Test successful generation of multiple samples."""
        # Mock successful response
        response_json = {
            "bridge_entity": "Entity",
            "question": "Question",
            "answer": "Answer",
            "gold_snippets": ["snippet1", "snippet2"],
        }
        mock_response = Mock()
        mock_response.text = json.dumps(response_json)
        mock_llm_client.models.generate_content.return_value = mock_response

        mocker.patch("random.sample", return_value=[
            generator.documents[0],
            generator.documents[1],
        ])

        dataset = generator.generate_dataset(num_samples=3)
        assert len(dataset) == 3
        assert all("sample_id" in sample for sample in dataset)
        assert all("question" in sample for sample in dataset)

    def test_generate_dataset_high_failure_rate(self, generator, mock_llm_client, mocker):
        """Test that high failure rate terminates generation early."""
        # Mock failure response
        response_json = {
            "bridge_entity": "N/A",
            "question": "IMPOSSIBLE",
            "answer": "IMPOSSIBLE",
            "gold_snippets": [],
        }
        mock_response = Mock()
        mock_response.text = json.dumps(response_json)
        mock_llm_client.models.generate_content.return_value = mock_response

        mocker.patch("random.sample", return_value=[
            generator.documents[0],
            generator.documents[1],
        ])

        dataset = generator.generate_dataset(num_samples=5)
        # Should stop before reaching 5 samples due to high failure rate
        assert len(dataset) < 5

    def test_generate_sample_missing_snippets_key(self, generator, mock_llm_client, mocker):
        """Test handling of missing gold_snippets key in response."""
        response_json = {
            "bridge_entity": "Entity",
            "question": "Question",
            "answer": "Answer",
            # Missing gold_snippets
        }
        mock_response = Mock()
        mock_response.text = json.dumps(response_json)
        mock_llm_client.models.generate_content.return_value = mock_response

        mocker.patch("random.sample", return_value=[
            generator.documents[0],
            generator.documents[1],
        ])

        sample = generator.generate_sample()
        assert sample is None

