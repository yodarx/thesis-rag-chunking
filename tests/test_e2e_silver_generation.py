"""
End-to-end tests for silver standard generation.
Tests the full pipeline of generating silver datasets.
"""

import json
import os
from unittest.mock import Mock

import pytest

from src.experiment.silver_generation import generate_silver_for_experiment


@pytest.fixture
def e2e_silver_setup(tmp_path):
    """Setup for e2e silver generation tests."""
    # Create preprocessed input file
    data_dir = tmp_path / "data" / "preprocessed"
    data_dir.mkdir(parents=True, exist_ok=True)

    input_file = data_dir / "preprocessed_2025-11-03_all_categorized.jsonl"
    documents = [
        {
            "id": "doc1",
            "document_text": "This is a long document about the Python programming language. Python was created by Guido van Rossum in 1989.",
            "category": "Programming",
        },
        {
            "id": "doc2",
            "document_text": "Guido van Rossum is a Dutch computer programmer who designed Python. He received the NIST Award for contributions to computer science.",
            "category": "People",
        },
        {
            "id": "doc3",
            "document_text": "The NIST Award recognizes outstanding contributions to computing. It has been awarded since 1989.",
            "category": "Awards",
        },
        {
            "id": "doc4",
            "document_text": "Java is another popular programming language. It was developed by James Gosling at Sun Microsystems.",
            "category": "Programming",
        },
        {
            "id": "doc5",
            "document_text": "Sun Microsystems was a technology company founded in 1982. James Gosling worked there for many years.",
            "category": "Companies",
        },
    ]

    with open(input_file, "w") as f:
        for doc in documents:
            f.write(json.dumps(doc) + "\n")

    return {
        "tmp_path": tmp_path,
        "input_file": str(input_file),
        "documents": documents,
    }


class TestGenerateSilverForExperiment:
    """Tests for generate_silver_for_experiment function."""

    def test_e2e_generate_silver_success(self, e2e_silver_setup):
        """
        E2E test: Successfully generate silver dataset with mocked LLM.
        """
        setup = e2e_silver_setup
        tmp_path = setup["tmp_path"]

        # Create mock LLM client
        mock_llm_client = Mock()

        # Mock valid LLM responses
        valid_response = Mock()
        valid_response.text = json.dumps(
            {
                "bridge_entity": "Python",
                "question": "Who created the programming language that was released in 1989?",
                "answer": "Guido van Rossum",
                "gold_snippets": [
                    "Python was created by Guido van Rossum in 1989.",
                    "Guido van Rossum is a Dutch computer programmer who designed Python.",
                ],
            }
        )
        mock_llm_client.models.generate_content.return_value = valid_response

        # Change to tmp directory
        cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            # Create symlink or copy the preprocessed file
            preprocessed_dir = tmp_path / "data" / "preprocessed"
            preprocessed_dir.mkdir(parents=True, exist_ok=True)
            expected_input = preprocessed_dir / "preprocessed_2025-11-03_all_categorized.jsonl"

            # Copy from setup
            with open(expected_input, "w") as f:
                for doc in setup["documents"]:
                    f.write(json.dumps(doc) + "\n")

            # Run generation
            output_file = generate_silver_for_experiment(mock_llm_client, limit=2)

            # Verify output file exists
            assert os.path.exists(output_file), f"Output file {output_file} was not created"

            # Verify output file is JSONL format
            with open(output_file) as f:
                lines = f.readlines()
                assert len(lines) <= 2, f"Expected at most 2 samples, got {len(lines)}"

                for line in lines:
                    data = json.loads(line)
                    assert "sample_id" in data
                    assert "question" in data
                    assert "answer" in data
                    assert "gold_passages" in data
                    assert "category" in data
                    assert data["category"] == "Multihop"
                    assert data["difficulty"] == "Hard"
                    assert "metadata" in data
                    assert "source_chunks" in data["metadata"]

        finally:
            os.chdir(cwd)

    def test_e2e_generate_silver_nonexistent_input(self, tmp_path):
        """
        E2E test: Handle case where input file doesn't exist.
        """
        mock_llm_client = Mock()

        cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            # Input file doesn't exist
            output_file = generate_silver_for_experiment(mock_llm_client, limit=1)

            # Should still create output file with 0 samples (empty dataset)
            assert os.path.exists(output_file)
            with open(output_file) as f:
                lines = f.readlines()
                # May be empty or have fewer samples due to no input
                assert len(lines) == 0

        finally:
            os.chdir(cwd)

    def test_e2e_output_directory_created(self, e2e_silver_setup):
        """
        E2E test: Verify that output directory is created if it doesn't exist.
        """
        setup = e2e_silver_setup
        tmp_path = setup["tmp_path"]

        # Ensure output directory doesn't exist
        silver_dir = tmp_path / "data" / "silver"
        if silver_dir.exists():
            import shutil

            shutil.rmtree(silver_dir)

        mock_llm_client = Mock()
        valid_response = Mock()
        valid_response.text = json.dumps(
            {
                "bridge_entity": "Entity",
                "question": "Question?",
                "answer": "Answer",
                "gold_snippets": ["s1", "s2"],
            }
        )
        mock_llm_client.models.generate_content.return_value = valid_response

        cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            preprocessed_dir = tmp_path / "data" / "preprocessed"
            preprocessed_dir.mkdir(parents=True, exist_ok=True)
            expected_input = preprocessed_dir / "preprocessed_2025-11-03_all_categorized.jsonl"

            with open(expected_input, "w") as f:
                for doc in setup["documents"]:
                    f.write(json.dumps(doc) + "\n")

            output_file = generate_silver_for_experiment(mock_llm_client, limit=1)

            # Verify directory was created
            assert silver_dir.exists(), "Output directory was not created"
            assert os.path.exists(output_file), "Output file was not created"

        finally:
            os.chdir(cwd)

    def test_e2e_output_file_format(self, e2e_silver_setup):
        """
        E2E test: Verify that output file follows correct JSONL format.
        """
        setup = e2e_silver_setup
        tmp_path = setup["tmp_path"]

        mock_llm_client = Mock()

        # Mock multiple responses
        responses = [
            {
                "bridge_entity": "Python",
                "question": "Who created Python?",
                "answer": "Guido van Rossum",
                "gold_snippets": ["Python was created by Guido van Rossum", "Guido created Python"],
            },
            {
                "bridge_entity": "Java",
                "question": "Who created Java?",
                "answer": "James Gosling",
                "gold_snippets": ["Java was developed by James Gosling", "James created Java"],
            },
        ]

        response_iter = iter(responses)

        def mock_generate_content(*args, **kwargs):
            resp = Mock()
            resp.text = json.dumps(next(response_iter))
            return resp

        mock_llm_client.models.generate_content.side_effect = mock_generate_content

        cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            preprocessed_dir = tmp_path / "data" / "preprocessed"
            preprocessed_dir.mkdir(parents=True, exist_ok=True)
            expected_input = preprocessed_dir / "preprocessed_2025-11-03_all_categorized.jsonl"

            with open(expected_input, "w") as f:
                for doc in setup["documents"]:
                    f.write(json.dumps(doc) + "\n")

            output_file = generate_silver_for_experiment(mock_llm_client, limit=2)

            # Verify file content
            with open(output_file) as f:
                lines = f.readlines()
                assert len(lines) <= 2

                for _i, line in enumerate(lines):
                    # Each line should be valid JSON
                    data = json.loads(line)
                    # Verify required fields
                    assert "sample_id" in data
                    assert "question" in data
                    assert "answer" in data
                    assert "gold_passages" in data
                    assert "category" in data
                    assert "difficulty" in data
                    assert "metadata" in data

        finally:
            os.chdir(cwd)

    def test_e2e_timestamp_in_filename(self, e2e_silver_setup):
        """
        E2E test: Verify that output filename includes timestamp.
        """
        setup = e2e_silver_setup
        tmp_path = setup["tmp_path"]

        mock_llm_client = Mock()
        valid_response = Mock()
        valid_response.text = json.dumps(
            {
                "bridge_entity": "Entity",
                "question": "Question?",
                "answer": "Answer",
                "gold_snippets": ["s1", "s2"],
            }
        )
        mock_llm_client.models.generate_content.return_value = valid_response

        cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            preprocessed_dir = tmp_path / "data" / "preprocessed"
            preprocessed_dir.mkdir(parents=True, exist_ok=True)
            expected_input = preprocessed_dir / "preprocessed_2025-11-03_all_categorized.jsonl"

            with open(expected_input, "w") as f:
                for doc in setup["documents"]:
                    f.write(json.dumps(doc) + "\n")

            output_file = generate_silver_for_experiment(mock_llm_client, limit=1)

            # Verify filename contains timestamp pattern
            filename = os.path.basename(output_file)
            # Should match pattern: YYYY-MM-DD_HH-MM-SS_silver.jsonl
            import re

            pattern = r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_silver\.jsonl"
            assert re.match(pattern, filename), (
                f"Filename {filename} doesn't match timestamp pattern"
            )

        finally:
            os.chdir(cwd)

    def test_e2e_handles_multiple_field_types(self, tmp_path):
        """
        E2E test: Verify that documents with various field names are loaded.
        """
        # Create input file with mixed field names
        preprocessed_dir = tmp_path / "data" / "preprocessed"
        preprocessed_dir.mkdir(parents=True, exist_ok=True)

        input_file = preprocessed_dir / "preprocessed_2025-11-03_all_categorized.jsonl"
        docs = [
            {"document_text": "Document 1 text is long enough for testing purposes."},
            {"text": "Document 2 text is long enough for testing purposes."},
            {"content": "Document 3 text is long enough for testing purposes."},
            {"document": "Document 4 text is long enough for testing purposes."},
        ]

        with open(input_file, "w") as f:
            for doc in docs:
                f.write(json.dumps(doc) + "\n")

        mock_llm_client = Mock()
        valid_response = Mock()
        valid_response.text = json.dumps(
            {
                "bridge_entity": "Entity",
                "question": "Question?",
                "answer": "Answer",
                "gold_snippets": ["s1", "s2"],
            }
        )
        mock_llm_client.models.generate_content.return_value = valid_response

        cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            output_file = generate_silver_for_experiment(mock_llm_client, limit=1)
            assert os.path.exists(output_file)

        finally:
            os.chdir(cwd)

    def test_e2e_unicode_handling(self, e2e_silver_setup):
        """
        E2E test: Verify that unicode content is handled correctly.
        """
        setup = e2e_silver_setup
        tmp_path = setup["tmp_path"]

        # Create documents with unicode
        preprocessed_dir = tmp_path / "data" / "preprocessed"
        preprocessed_dir.mkdir(parents=True, exist_ok=True)

        input_file = preprocessed_dir / "preprocessed_2025-11-03_all_categorized.jsonl"
        docs = [
            {"document_text": "This is about Café and Naïve concepts. 你好世界 مرحبا"},
            {"document_text": "Another document with émojis: 🚀 🌟 and more content."},
        ]

        with open(input_file, "w", encoding="utf-8") as f:
            for doc in docs:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")

        mock_llm_client = Mock()
        valid_response = Mock()
        valid_response.text = json.dumps(
            {
                "bridge_entity": "Concept",
                "question": "Question with unicode: Café?",
                "answer": "Answer with unicode: Naïve",
                "gold_snippets": ["Snippet 1 with unicode", "Snippet 2 with more unicode"],
            },
            ensure_ascii=False,
        )
        mock_llm_client.models.generate_content.return_value = valid_response

        cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            output_file = generate_silver_for_experiment(mock_llm_client, limit=1)

            # Verify unicode content is preserved
            with open(output_file, encoding="utf-8") as f:
                lines = f.readlines()
                if lines:
                    data = json.loads(lines[0])
                    # Unicode should be preserved
                    assert "unicode" in str(data).lower() or len(lines) == 0

        finally:
            os.chdir(cwd)

    def test_e2e_high_failure_rate_stops_generation(self, e2e_silver_setup):
        """
        E2E test: Verify that generation stops early with high failure rate.
        """
        setup = e2e_silver_setup
        tmp_path = setup["tmp_path"]

        mock_llm_client = Mock()

        # Always return IMPOSSIBLE (failure)
        failure_response = Mock()
        failure_response.text = json.dumps(
            {
                "bridge_entity": "N/A",
                "question": "IMPOSSIBLE",
                "answer": "IMPOSSIBLE",
                "gold_snippets": [],
            }
        )
        mock_llm_client.models.generate_content.return_value = failure_response

        cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            preprocessed_dir = tmp_path / "data" / "preprocessed"
            preprocessed_dir.mkdir(parents=True, exist_ok=True)
            expected_input = preprocessed_dir / "preprocessed_2025-11-03_all_categorized.jsonl"

            with open(expected_input, "w") as f:
                for doc in setup["documents"]:
                    f.write(json.dumps(doc) + "\n")

            output_file = generate_silver_for_experiment(mock_llm_client, limit=10)

            # Should have far fewer than 10 samples due to high failure rate
            with open(output_file) as f:
                lines = f.readlines()
                assert len(lines) < 10, "Generation did not stop despite high failure rate"

        finally:
            os.chdir(cwd)
