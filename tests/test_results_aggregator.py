import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

from results_aggregator import ResultProcessor, MetadataService, FileSystemManager


class TestPipeline(unittest.TestCase):

    def setUp(self):
        self.mock_meta_service = MagicMock()
        self.processor = ResultProcessor(self.mock_meta_service)

    def test_enrich_row_logic_preserves_retrieval_time(self):
        """
        Ensures that extra columns from CSV (like retrieval_time)
        are preserved during enrichment.
        """
        # Arrange
        row = {
            "experiment": "fixed_1024",
            "mrr": 0.8,
            "retrieval_time": 0.054  # <-- The specific column check
        }
        self.mock_meta_service.get_chunk_metadata.return_value = {"chunk_time": 10}
        self.mock_meta_service.get_index_metadata.return_value = {"index_time": 20}

        # Act
        result = self.processor._enrich_row(row, "bert", "Hard", 1000)

        # Assert
        self.assertEqual(result["experiment"], "fixed_1024")
        self.assertEqual(result["retrieval_time"], 0.054)
        self.assertEqual(result["chunk_time"], 10)
        self.assertEqual(result["index_time"], 20)
        self.assertEqual(result["config_difficulty"], "Hard")

        # Verify calls
        self.mock_meta_service.get_chunk_metadata.assert_called_with("fixed_1024")
        self.mock_meta_service.get_index_metadata.assert_called_with("fixed_1024", "bert")

    @patch("builtins.open", new_callable=mock_open, read_data='{"processing_time_seconds": 99}')
    @patch("json.load")
    def test_metadata_service_chunk(self, mock_json, mock_file):
        """Tests that chunk metadata is read correctly from JSON."""
        mock_json.return_value = {"processing_time_seconds": 99}
        service = MetadataService(Path("/tmp"))

        data = service.get_chunk_metadata("exp1")

        self.assertEqual(data["chunk_processing_time_s"], 99)
        mock_file.assert_called_with(Path("/tmp/chunks/exp1/metadata.json"), encoding='utf-8')

    @patch("builtins.open", new_callable=mock_open)
    @patch("json.load")
    def test_metadata_service_index_path_construction(self, mock_json, mock_file):
        """Ensures index path is constructed as $experimentName_$embeddingModel."""
        mock_json.return_value = {}
        service = MetadataService(Path("/tmp"))

        service.get_index_metadata("exp1", "modelA")

        expected_path = Path("/tmp/indices/exp1_modelA/metadata.json")
        mock_file.assert_called_with(expected_path, encoding='utf-8')

    @patch("builtins.open", side_effect=FileNotFoundError)
    def test_filesystem_manager_missing_file(self, mock_file):
        """Ensures missing files return empty dicts instead of crashing."""
        result = FileSystemManager.read_json(Path("fake.json"))
        self.assertEqual(result, {})


if __name__ == "__main__":
    unittest.main()
