import json
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch
import pandas as pd
import pytest

from results_aggregator import FileSystemManager, GCSDownloader, MetadataService, ResultProcessor

@pytest.fixture
def mock_meta_service():
    return MagicMock()

@pytest.fixture
def processor(mock_meta_service):
    return ResultProcessor(mock_meta_service)

def test_enrich_row_logic_preserves_retrieval_time(processor, mock_meta_service):
    """
    Ensures that extra columns from CSV (like retrieval_time)
    are preserved during enrichment.
    """
    # Arrange
    row = {
        "experiment": "fixed_1024",
        "mrr": 0.8,
        "retrieval_time": 0.054
    }
    mock_meta_service.get_chunk_metadata.return_value = {"chunk_time": 10}
    mock_meta_service.get_index_metadata.return_value = {"index_time": 20}

    # Act
    result = processor._enrich_row(row, "bert", "Hard", "exp_config_1")

    # Assert
    assert result["experiment"] == "fixed_1024"
    assert result["retrieval_time"] == 0.054
    assert result["chunk_time"] == 10
    assert result["index_time"] == 20
    assert result["config_difficulty"] == "Hard"
    assert result["experiment_config"] == "exp_config_1"

    # Verify calls
    mock_meta_service.get_chunk_metadata.assert_called_with("fixed_1024")
    mock_meta_service.get_index_metadata.assert_called_with("fixed_1024", "bert")

def test_process_folder_parsing(processor):
    """Tests that folder parser extracts embedding, difficulty, and config."""
    # Arrange
    folder = MagicMock(spec=Path)
    folder.name = "all-MiniLM-L6-v2_01_gold_easy_Easy"
    folder.glob.return_value = []  # No CSVs

    with patch("pandas.read_csv") as mock_read:
        # Mock CSV content
        df_mock = pd.DataFrame([{"experiment": "s1"}])
        mock_read.return_value = df_mock

        # Mock glob to return one file
        csv_file = MagicMock(spec=Path)
        csv_file.name = "res_detailed_results.csv"
        folder.glob.return_value = [csv_file]

        # Spy on _enrich_row to capture what was passed
        with patch.object(processor, '_enrich_row', return_value={}) as mock_enrich:
            processor._process_experiment_folder(folder)

            # Check what was passed to _enrich_row
            # Call args: (row, embedding, difficulty, config)
            args = mock_enrich.call_args[0]
            embedding = args[1]
            difficulty = args[2]
            config = args[3]

            assert embedding == "all-MiniLM-L6-v2"
            assert difficulty == "Easy"
            assert config == "01_gold_easy"

def test_process_folder_parsing_no_suffix(processor):
    """Tests folder parsing for case without difficulty suffix (e.g. _all)."""
    folder = MagicMock(spec=Path)
    folder.name = "BAAI_bge-base_01_gold_all"

    with patch("pandas.read_csv") as mock_read:
        mock_read.return_value = pd.DataFrame([{"experiment": "s1"}])
        folder.glob.return_value = [MagicMock()]

        with patch.object(processor, '_enrich_row', return_value={}) as mock_enrich:
            processor._process_experiment_folder(folder)

            args = mock_enrich.call_args[0]
            assert args[1] == "BAAI_bge-base"
            assert args[2] == "All"  # Default
            assert args[3] == "01_gold_all"  # Rest

@patch("builtins.open", new_callable=mock_open, read_data='{"processing_time_seconds": 99}')
@patch("json.load")
def test_metadata_service_chunk(mock_json_load, mock_file):
    """Tests that chunk metadata is read correctly from JSON."""
    mock_json_load.return_value = {"processing_time_seconds": 99}
    service = MetadataService(Path("/tmp"))

    data = service.get_chunk_metadata("exp1")

    assert data["chunk_processing_time_s"] == 99
    mock_file.assert_called_with(Path("/tmp/chunks/exp1/metadata.json"), encoding='utf-8')

@patch("builtins.open", new_callable=mock_open)
@patch("json.load")
def test_metadata_service_index_path_construction(mock_json_load, mock_file):
    """Ensures index path is constructed as $experimentName_$embeddingModel."""
    mock_json_load.return_value = {}
    service = MetadataService(Path("/tmp"))

    service.get_index_metadata("exp1", "modelA")

    expected_path = Path("/tmp/indices/exp1_modelA/index_metadata.json")
    mock_file.assert_called_with(expected_path, encoding='utf-8')

@patch("builtins.open", side_effect=FileNotFoundError)
def test_filesystem_manager_missing_file(mock_file):
    """Ensures missing files return empty dicts instead of crashing."""
    result = FileSystemManager.read_json(Path("fake.json"))
    assert result == {}

def test_process_all_results_ignores_archive(processor):
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
    with patch.object(processor, '_process_experiment_folder', return_value=[{"res": 1}]) as mock_process:
        df = processor.process_all_results(mock_root)

        # Should have called process only for dir1
        mock_process.assert_called_once_with(dir1)
        assert len(df) == 1

def test_gcs_downloader_is_relevant_file_ignores_archive():
    """Test that GCSDownloader ignores files in archive."""
    # We can test the private method directly or via public interface if feasible.
    # Since _is_relevant_file is what we changed and it is stateless logic (mostly),
    # passing mocks to init might be needed or just mocking.
    # But GCSDownloader init creates client. We should mock that.
    with patch("google.cloud.storage.Client"):
        downloader = GCSDownloader("test-bucket")

        assert downloader._is_relevant_file("data/results/exp1/metadata.json")
        assert not downloader._is_relevant_file("data/results/archive/exp1/metadata.json")
        assert not downloader._is_relevant_file("data/chunks/archive/exp1/metadata.json")
        assert downloader._is_relevant_file("data/chunks/exp1/metadata.json")

def test_gcs_downloader_path_construction():
    """Ensures that downloaded blobs are placed inside the local data directory."""
    with patch("google.cloud.storage.Client"):
        # Initialize with a custom base dir to verify it is prepended
        custom_base = Path("temp_test_data")
        downloader = GCSDownloader("test-bucket", local_base_dir=custom_base)

        # Mock a blob with a path like it appears in the bucket
        mock_blob = MagicMock()
        mock_blob.name = "results/experiment_1/metadata.json"

        with patch.object(FileSystemManager, "ensure_directory") as mock_ensure:
            with patch.object(Path, "exists", return_value=False):
                downloader._download_blob(mock_blob)

                # Expected path: temp_test_data/results/experiment_1/metadata.json
                expected_path = custom_base / "results/experiment_1/metadata.json"
                mock_blob.download_to_filename.assert_called_with(str(expected_path))
                mock_ensure.assert_called()

