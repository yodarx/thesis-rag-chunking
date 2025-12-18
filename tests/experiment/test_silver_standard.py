from unittest.mock import Mock

import pytest
from langchain_community.llms import Ollama

from src.experiment.retriever import FaissRetriever
from src.experiment.silver_standard import SilverStandardGenerator


@pytest.fixture
def mock_retriever():
    retriever = Mock(spec=FaissRetriever)
    retriever.chunks = ["Chunk 1 content", "Chunk 2 content", "Chunk 3 content"]
    return retriever


@pytest.fixture
def mock_llm():
    llm = Mock(spec=Ollama)
    llm.invoke.return_value = '{"question": "What is the connection between Chunk 1 and Chunk 2?", "answer": "The connection is..."}'
    return llm


def test_generate_sample(mock_retriever, mock_llm):
    generator = SilverStandardGenerator(mock_retriever, mock_llm)

    sample = generator.generate_sample(num_hops=2)

    assert "sample_id" in sample
    assert "question" in sample
    assert "answer" in sample
    assert "gold_passages" in sample
    assert "category" in sample
    assert "difficulty" in sample
    assert sample["category"] == "Multihop"
    assert sample["difficulty"] == "Hard"
    assert sample["question"] == "What is the connection between Chunk 1 and Chunk 2?"
    assert sample["answer"] == "The connection is..."
    assert len(sample["gold_passages"]) == 2

    mock_llm.invoke.assert_called_once()


def test_generate_sample_custom_hops(mock_retriever, mock_llm):
    generator = SilverStandardGenerator(mock_retriever, mock_llm)

    # Test with 3 hops
    sample = generator.generate_sample(num_hops=3)

    assert len(sample["gold_passages"]) == 3
    mock_llm.invoke.assert_called()


def test_generate_dataset(mock_retriever, mock_llm):
    generator = SilverStandardGenerator(mock_retriever, mock_llm)

    dataset = generator.generate_dataset(3, num_hops=2)

    assert len(dataset) == 3
    assert dataset[0]["question"] == "What is the connection between Chunk 1 and Chunk 2?"
    assert dataset[0]["answer"] == "The connection is..."


def test_prompt_contains_constraints(mock_retriever, mock_llm):
    generator = SilverStandardGenerator(mock_retriever, mock_llm)
    chunks = ["Context A", "Context B"]
    prompt = generator._build_multihop_prompt(chunks)

    assert "requires information from ALL contexts" in prompt
    assert "based ONLY on the provided contexts" in prompt
    assert "1:1" in prompt
    assert "IMPOSSIBLE" in prompt
    assert "EXACT substring" in prompt


def test_get_random_contexts_error(mock_retriever, mock_llm):
    mock_retriever.chunks = []
    generator = SilverStandardGenerator(mock_retriever, mock_llm)

    with pytest.raises(ValueError, match="Retriever has no chunks loaded"):
        generator._get_random_contexts()


def test_parse_llm_response_error_handling(mock_retriever, mock_llm):
    # Test handling of invalid JSON
    mock_llm.invoke.return_value = "Invalid JSON response"
    generator = SilverStandardGenerator(mock_retriever, mock_llm)

    sample = generator.generate_sample()
    assert sample is None


def test_generate_sample_impossible(mock_retriever, mock_llm):
    mock_llm.invoke.return_value = '{"question": "IMPOSSIBLE", "answer": "IMPOSSIBLE"}'
    generator = SilverStandardGenerator(mock_retriever, mock_llm)

    sample = generator.generate_sample()
    assert sample is None


def test_generate_dataset_retry(mock_retriever, mock_llm):
    # First 2 calls return IMPOSSIBLE, 3rd returns valid
    mock_llm.invoke.side_effect = [
        '{"question": "IMPOSSIBLE", "answer": "IMPOSSIBLE"}',
        '{"question": "IMPOSSIBLE", "answer": "IMPOSSIBLE"}',
        '{"question": "Valid Q", "answer": "Valid A"}',
    ]
    generator = SilverStandardGenerator(mock_retriever, mock_llm)

    dataset = generator.generate_dataset(num_samples=1)

    assert len(dataset) == 1
    assert dataset[0]["question"] == "Valid Q"
    assert mock_llm.invoke.call_count == 3


def test_generate_dataset_failure_limit(mock_retriever, mock_llm):
    # Always returns IMPOSSIBLE
    mock_llm.invoke.return_value = '{"question": "IMPOSSIBLE", "answer": "IMPOSSIBLE"}'
    generator = SilverStandardGenerator(mock_retriever, mock_llm)

    # Ask for 1 sample, it should eventually give up
    dataset = generator.generate_dataset(num_samples=1)

    assert len(dataset) == 0
    # Should have tried max_failures (1 * 10) times
    assert mock_llm.invoke.call_count >= 10


def test_get_random_contexts_filtering(mock_retriever, mock_llm):
    # Setup mixed chunks: 2 short, 2 long
    long_chunk_1 = "A" * 101
    long_chunk_2 = "B" * 101
    mock_retriever.chunks = ["Short 1", "Short 2", long_chunk_1, long_chunk_2]

    generator = SilverStandardGenerator(mock_retriever, mock_llm)

    # Request 2 chunks, default min_char_length is 100
    contexts = generator._get_random_contexts(n=2)

    # Should only get the long ones
    assert len(contexts) == 2
    assert long_chunk_1 in contexts
    assert long_chunk_2 in contexts
    assert "Short 1" not in contexts


def test_get_random_contexts_fallback(mock_retriever, mock_llm):
    # Setup only short chunks
    mock_retriever.chunks = ["Short 1", "Short 2", "Short 3"]

    generator = SilverStandardGenerator(mock_retriever, mock_llm)

    # Request 2 chunks, default min_char_length is 100
    # Should fallback to using short chunks instead of returning fewer or crashing
    contexts = generator._get_random_contexts(n=2)

    assert len(contexts) == 2
    assert set(contexts).issubset(set(mock_retriever.chunks))
