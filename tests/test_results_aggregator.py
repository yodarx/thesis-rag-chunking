import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

from results_aggregator import ResultProcessor, MetadataService, FileSystemManager, GCSDownloader


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

    def test_process_all_results_ignores_archive(self):
        """Ensures that folders named 'archive' are skipped during processing."""
        # Mock filesystem
        mock_root = MagicMock(spec=Path)
        mock_root.exists.return_value = True

        dir1 = MagicMock(spec=Path)
        dir1.is_dir.return_value = True
        dir1.name = "valid_exp"

        dir2 = MagicMock(spec=Path)
        dir2.is_dir.return_value = True
        dir2.name = "archive"

        dir3 = MagicMock(spec=Path)
        dir3.is_dir.return_value = True
        dir3.name = "_archive"

        mock_root.iterdir.return_value = [dir1, dir2, dir3]

        # Mock internal processing method to avoid IO
        with patch.object(self.processor, '_process_experiment_folder', return_value=[{"res": 1}]) as mock_process:
            df = self.processor.process_all_results(mock_root)

            # Should have called process only for dir1
            mock_process.assert_called_once_with(dir1)
            self.assertEqual(len(df), 1)

    def test_gcs_downloader_is_relevant_file_ignores_archive(self):
        """Test that GCSDownloader ignores files in archive."""
        # We can test the private method directly or via public interface if feasible.
        # Since _is_relevant_file is what we changed and it is stateless logic (mostly),
        # passing mocks to init might be needed or just mocking.
        # But GCSDownloader init creates client. We should mock that.
        with patch("google.cloud.storage.Client"):
            downloader = GCSDownloader("test-bucket")

            self.assertTrue(downloader._is_relevant_file("data/results/exp1/metadata.json"))
            self.assertFalse(downloader._is_relevant_file("data/results/archive/exp1/metadata.json"))
            self.assertFalse(downloader._is_relevant_file("data/chunks/archive/exp1/metadata.json"))
            self.assertTrue(downloader._is_relevant_file("data/chunks/exp1/metadata.json"))


if __name__ == "__main__":
    unittest.main()
