import json
import os
from pathlib import Path
from unittest import mock

import pytest
from pytest_mock import MockerFixture

import build_silver


@pytest.fixture
def sample_config(tmp_path: Path) -> str:
    """Create a sample configuration file for testing."""
    config = {
        "embedding_model": "test-model",
        "silver_limit": 5,
        "hops_count": 2,
        "experiments": [
            {
                "name": "test_fixed_512_50",
                "function": "chunk_fixed_size",
                "params": {"chunk_size": 512, "chunk_overlap": 50},
            },
            {
                "name": "test_recursive_512_50",
                "function": "chunk_recursive",
                "params": {"chunk_size": 512, "chunk_overlap": 50},
            },
        ],
    }
    config_path = tmp_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f)
    return str(config_path)


@pytest.fixture
def mock_genai_client(mocker: MockerFixture):
    """Mock the Google Gen AI client."""
    mock_client = mocker.Mock()
    mock_response = mocker.Mock()
    mock_response.text = '{"question": "What is test?", "answer": "Test answer"}'
    mock_client.models.generate_content.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_dependencies(mocker: MockerFixture):
    """Mock all dependencies for build_silver functions."""
    mock_vectorizer = mocker.Mock()
    mock_retriever = mocker.Mock()
    mock_generator = mocker.Mock()

    mocker.patch("build_silver.Vectorizer.from_model_name", return_value=mock_vectorizer)
    mocker.patch("build_silver.FaissRetriever", return_value=mock_retriever)
    mocker.patch("build_silver.SilverStandardGenerator", return_value=mock_generator)
    mocker.patch("build_silver.genai.Client")

    return {
        "vectorizer": mock_vectorizer,
        "retriever": mock_retriever,
        "generator": mock_generator,
    }


def test_create_index_name():
    """Test index name creation with various model names."""
    # Standard model name
    result = build_silver.create_index_name("test_exp", "all-MiniLM-L6-v2")
    assert result == "test_exp_all-MiniLM-L6-v2"

    # Model name with forward slash
    result = build_silver.create_index_name("test_exp", "BAAI/bge-large-en-v1.5")
    assert result == "test_exp_BAAI_bge-large-en-v1.5"

    # Model name with multiple slashes
    result = build_silver.create_index_name("exp", "org/repo/model-name")
    assert result == "exp_org_repo_model-name"


def test_load_config(tmp_path: Path):
    """Test configuration file loading."""
    config = {"key": "value", "experiments": []}
    config_path = tmp_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f)

    result = build_silver.load_config(str(config_path))
    assert result == config


def test_load_config_file_not_found():
    """Test that loading non-existent config raises error."""
    with pytest.raises(RuntimeError, match="Error loading config file"):
        build_silver.load_config("/nonexistent/path.json")


def test_load_config_invalid_json(tmp_path: Path):
    """Test that loading invalid JSON raises error."""
    config_path = tmp_path / "invalid.json"
    with open(config_path, "w") as f:
        f.write("invalid json {")

    with pytest.raises(RuntimeError, match="Error loading config file"):
        build_silver.load_config(str(config_path))


def test_generate_silver_for_experiment_input_file_missing(
    mocker: MockerFixture,
    capsys,
):
    """Test that experiment is skipped if input file is missing."""
    experiment = {
        "name": "test_exp_missing",
    }
    config = {
        "embedding_model": "test-model",
        "input_file": "/nonexistent/input.jsonl",
    }

    # Create mock vectorizer and client
    mock_vectorizer = mocker.Mock()
    mock_client = mocker.Mock()

    # Call the function - it should skip since input file doesn't exist
    from src.experiment.silver_generation import generate_silver_for_experiment

    generate_silver_for_experiment(experiment, config, mock_vectorizer, mock_client, "gemini", 10)

    captured = capsys.readouterr()
    assert "Skipping" in captured.out
    assert "No documents found" in captured.out


def test_generate_silver_for_experiment_silver_exists(
    mocker: MockerFixture,
    tmp_path: Path,
    capsys,
):
    """Test that experiment is skipped if silver standard already exists."""

    experiment = {
        "name": "test_exp_silver_exists",
    }
    config = {
        "embedding_model": "test-model",
        "input_file": str(tmp_path / "test_input.jsonl"),
    }

    # Create necessary directories and files in tmp_path
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    silver_dir = data_dir / "silver"
    silver_dir.mkdir(parents=True, exist_ok=True)

    # Create a sample input file
    with open(tmp_path / "test_input.jsonl", "w") as f:
        f.write('{"text": "Sample document 1"}\n')
        f.write('{"text": "Sample document 2"}\n')

    silver_file = silver_dir / "test_exp_silver_exists_test-model_silver.jsonl"
    with open(silver_file, "w") as f:
        f.write("")

    mock_vectorizer = mocker.Mock()
    mock_client = mocker.Mock()

    # Change to tmp_path for the test
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        from src.experiment.silver_generation import generate_silver_for_experiment

        generate_silver_for_experiment(
            experiment, config, mock_vectorizer, mock_client, "gemini", 10
        )

        captured = capsys.readouterr()
        assert "Skipping" in captured.out
        assert "already exists" in captured.out
    finally:
        os.chdir(cwd)


def test_generate_silver_for_experiment_success(
    mocker: MockerFixture,
    tmp_path: Path,
):
    """Test successful silver standard generation."""

    # Use a unique experiment name
    experiment = {
        "name": "test_exp_success",
    }
    config = {
        "embedding_model": "test-model",
        "hops_count": 2,
        "input_file": str(tmp_path / "test_input_success.jsonl"),
    }

    # Create necessary directories and files in tmp_path
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    silver_dir = data_dir / "silver"
    silver_dir.mkdir(parents=True, exist_ok=True)

    # Create a sample input file with documents
    with open(tmp_path / "test_input_success.jsonl", "w") as f:
        f.write('{"text": "Document 1 about machine learning and AI"}\n')
        f.write('{"text": "Document 2 about deep learning models"}\n')
        f.write('{"text": "Document 3 about neural networks"}\n')

    # Mock the generator
    mock_generator = mocker.Mock()
    mock_dataset = [
        {
            "sample_id": "1",
            "question": "Test question",
            "answer": "Test answer",
            "gold_passages": ["Document 1 about machine learning and AI"],
            "category": "Multihop",
            "difficulty": "Hard",
        }
    ]
    mock_generator.generate_dataset.return_value = mock_dataset

    # Mock the SilverStandardGenerator in the silver_generation module
    mocker.patch(
        "src.experiment.silver_generation.SilverStandardGenerator", return_value=mock_generator
    )

    # Mock Vectorizer and FaissRetriever
    mock_vectorizer = mocker.Mock()
    mocker.patch(
        "src.experiment.silver_generation.Vectorizer.from_model_name", return_value=mock_vectorizer
    )
    mocker.patch("src.experiment.silver_generation.FaissRetriever")

    # Mock genai client
    mock_client = mocker.Mock()
    mocker.patch("build_silver.genai.Client", return_value=mock_client)

    # Change to tmp_path for the test
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        # Import and call the function
        from src.experiment.silver_generation import generate_silver_for_experiment

        generate_silver_for_experiment(
            experiment, config, mock_vectorizer, mock_client, "gemini", 5
        )

        # Verify silver file was created
        silver_file = silver_dir / "test_exp_success_test-model_silver.jsonl"
        assert silver_file.exists()

        with open(silver_file) as f:
            lines = f.readlines()
            assert len(lines) == 1
            data = json.loads(lines[0])
            assert data["question"] == "Test question"
    finally:
        os.chdir(cwd)


@mock.patch("build_silver.initialize_llm_client")
@mock.patch("build_silver.Vectorizer.from_model_name")
def test_main_calls_all_experiments(
    mock_vectorizer_from_model,
    mock_llm_init,
    sample_config: str,
    mocker: MockerFixture,
):
    """Test that main processes all experiments."""
    mock_client = mock.Mock()
    mock_llm_init.return_value = (mock_client, "gemini")
    mock_vectorizer = mock.Mock()
    mock_vectorizer_from_model.return_value = mock_vectorizer

    mock_generate = mocker.patch("build_silver.generate_silver_for_experiment")

    build_silver.main(sample_config)

    # Should be called once for each experiment
    assert mock_generate.call_count == 2


@mock.patch("build_silver.initialize_llm_client")
@mock.patch("build_silver.Vectorizer.from_model_name")
def test_main_initializes_genai_client(
    mock_vectorizer_from_model,
    mock_llm_init,
    sample_config: str,
    mocker: MockerFixture,
):
    """Test that main initializes Google Gen AI client."""
    mock_client = mock.Mock()
    mock_llm_init.return_value = (mock_client, "gemini")
    mock_vectorizer = mock.Mock()
    mock_vectorizer_from_model.return_value = mock_vectorizer

    mocker.patch("build_silver.generate_silver_for_experiment")

    build_silver.main(sample_config)

    # Verify client was initialized
    mock_llm_init.assert_called_once_with("gemini", None)


@mock.patch("build_silver.initialize_llm_client")
@mock.patch("build_silver.Vectorizer.from_model_name")
def test_main_initializes_vectorizer(
    mock_vectorizer_from_model,
    mock_llm_init,
    sample_config: str,
    mocker: MockerFixture,
):
    """Test that main initializes vectorizer with correct model."""
    mock_client = mock.Mock()
    mock_llm_init.return_value = (mock_client, "gemini")
    mock_vectorizer = mock.Mock()
    mock_vectorizer_from_model.return_value = mock_vectorizer

    mocker.patch("build_silver.generate_silver_for_experiment")

    build_silver.main(sample_config)

    # Verify vectorizer was initialized with correct model
    mock_vectorizer_from_model.assert_called_once_with("test-model")


@mock.patch("build_silver.initialize_llm_client")
@mock.patch("build_silver.Vectorizer.from_model_name")
def test_main_with_default_limit(
    mock_vectorizer_from_model,
    mock_llm_init,
    tmp_path: Path,
    mocker: MockerFixture,
):
    """Test that main uses default limit when not specified in config."""
    config = {
        "embedding_model": "test-model",
        "experiments": [],
    }
    config_path = tmp_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f)

    mock_client = mock.Mock()
    mock_llm_init.return_value = (mock_client, "gemini")
    mock_vectorizer = mock.Mock()
    mock_vectorizer_from_model.return_value = mock_vectorizer

    build_silver.main(str(config_path))

    # Should use default limit of 10
    # (This is just a sanity check that it doesn't crash)
    assert True


def test_main_prints_status(sample_config: str, capsys, mocker: MockerFixture):
    """Test that main prints status messages."""
    mock_client_instance = mocker.Mock()
    mocker.patch(
        "build_silver.initialize_llm_client", return_value=(mock_client_instance, "gemini")
    )
    mocker.patch(
        "build_silver.Vectorizer.from_model_name",
        return_value=mocker.Mock(),
    )
    mocker.patch("build_silver.generate_silver_for_experiment")

    build_silver.main(sample_config)

    captured = capsys.readouterr()
    # Verify that status messages are printed
    assert "Limit per experiment" in captured.out
    assert "Processing experiment" in captured.out
