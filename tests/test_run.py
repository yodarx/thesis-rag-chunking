import json
import os

import pandas as pd
import pytest
from pytest_mock import MockerFixture

import run

config_path = os.path.join(os.path.dirname(__file__), "test_experiment_config.json")


@pytest.fixture
def mock_dependencies(mocker: MockerFixture):
    """Mockt alle Klassen, die von main() initialisiert werden."""
    mock_vectorizer_cls = mocker.patch("run.Vectorizer")
    mock_retriever_cls = mocker.patch("run.FaissRetriever")
    mock_results_cls = mocker.patch("run.ResultsHandler")
    mock_runner_cls = mocker.patch("run.ExperimentRunner")

    # Mocke die Funktionen
    mock_load_data = mocker.patch("run.load_asqa_dataset")
    mock_visualize = mocker.patch("run.visualize_and_save_results")
    mock_create_dir = mocker.patch("run.create_output_directory")

    # Konfiguriere Rückgabewerte
    mock_create_dir.return_value = "mock_dir"
    mock_load_data.return_value = [{"id": 1}]

    # Mock datetime to control timestamp in main()
    mock_dt = mocker.patch("run.datetime")
    mock_now = mocker.MagicMock()
    mock_dt.now.return_value = mock_now
    mock_now.strftime.return_value = "mock_ts"
    mock_now.isoformat.return_value = "2024-01-01T12:00:00"
    # duration calculation
    mock_now.__sub__.return_value.total_seconds.return_value = 10.0
    mock_now.__sub__.return_value.__str__.return_value = "0:00:10"

    # Mocke die Instanzen, die von den Konstruktoren zurückgegeben werden
    mock_runner_inst = mock_runner_cls.return_value
    mock_runner_inst.dataset = []  # Ensure dataset attr exists for len() check

    # Konfiguriere den Runner, ein (Mock) Summary-DataFrame zurückzugeben
    mock_summary_df = mocker.Mock(spec=pd.DataFrame)
    mock_summary_df.empty = False
    mock_runner_inst.run_all.return_value = mock_summary_df

    return {
        "load_data": mock_load_data,
        "visualize": mock_visualize,
        "create_dir": mock_create_dir,
        "Vector": mock_vectorizer_cls,
        "Retriever": mock_retriever_cls,
        "Results": mock_results_cls,
        "Runner": mock_runner_cls,
        "runner_inst": mock_runner_inst,
        "summary_df": mock_summary_df,
    }


def test_run_main_workflow(mock_dependencies, mocker):
    with open(config_path) as f:
        config_data = json.load(f)

    config_data["input_file"] = "data/preprocessed/standard.jsonl"

    mocker.patch("builtins.open", mocker.mock_open(read_data=json.dumps(config_data)))
    mocker.patch("json.load", return_value=config_data)
    mocker.patch("shutil.copy")

    run.main(config_json=config_path)

    mock_dependencies["create_dir"].assert_called_once_with("test-model_test_experiment_config_standard")

    mock_dependencies["load_data"].assert_called_once_with(
        config_data["input_file"], config_data.get("limit")
    )
    # Chunking models are loaded on-demand by ExperimentRunner
    mock_dependencies["Vector"].from_model_name.assert_called_once_with(
        config_data["embedding_model"]
    )

    mock_dependencies["Retriever"].assert_called_once()
    mock_dependencies["Results"].assert_called_once_with("mock_dir", "mock_ts")
    mock_dependencies["Runner"].assert_called_once()
    mock_dependencies["runner_inst"].run_all.assert_called_once()
    mock_dependencies["visualize"].assert_called_once_with(
        mock_dependencies["summary_df"], "mock_dir", "mock_ts"
    )


def test_run_main_silver_suffix(mock_dependencies, mocker):
    with open(config_path) as f:
        config_data = json.load(f)
    config_data["input_file"] = "data/silver/test_silver.jsonl"

    mocker.patch("builtins.open", mocker.mock_open(read_data=json.dumps(config_data)))
    mocker.patch("json.load", return_value=config_data)
    mocker.patch("shutil.copy")

    run.main(config_json=config_path)

    mock_dependencies["create_dir"].assert_called_once_with("test-model_test_experiment_config_test_silver")


def test_run_main_gold_suffix(mock_dependencies, mocker):
    with open(config_path) as f:
        config_data = json.load(f)
    config_data["input_file"] = "data/gold/test_gold.jsonl"

    mocker.patch("builtins.open", mocker.mock_open(read_data=json.dumps(config_data)))
    mocker.patch("json.load", return_value=config_data)
    mocker.patch("shutil.copy")

    run.main(config_json=config_path)

    mock_dependencies["create_dir"].assert_called_once_with("test-model_test_experiment_config_test_gold")


def test_run_main_silver_dataset(mock_dependencies, mocker):
    """Regression test after removing --silver flag: using a silver file is now driven by config['input_file']."""
    with open(config_path) as f:
        config_data = json.load(f)

    config_data["silver_file"] = "data/silver/test_silver.jsonl"
    config_data["input_file"] = config_data["silver_file"]

    mocker.patch("builtins.open", mocker.mock_open(read_data=json.dumps(config_data)))
    mocker.patch("json.load", return_value=config_data)
    mocker.patch("shutil.copy")

    run.main(config_json=config_path)

    mock_dependencies["create_dir"].assert_called_once_with("test-model_test_experiment_config_test_silver")
    mock_dependencies["load_data"].assert_called_with(
        "data/silver/test_silver.jsonl", config_data.get("limit")
    )


def test_run_main_with_difficulty(mock_dependencies, mocker):
    with open(config_path) as f:
        config_data = json.load(f)
    config_data["input_file"] = "data/preprocessed/standard.jsonl"

    mocker.patch("builtins.open", mocker.mock_open(read_data=json.dumps(config_data)))
    mocker.patch("json.load", return_value=config_data)
    mocker.patch("shutil.copy")

    run.main(config_json=config_path, difficulty="Hard")

    # expected format: $embedding_$experimentName_$usedInputFile_$difficulty
    mock_dependencies["create_dir"].assert_called_once_with("test-model_test_experiment_config_standard_Hard")


def test_run_main_no_data(mock_dependencies, mocker):
    with open(config_path) as f:
        config_data = json.load(f)
    mocker.patch("builtins.open", mocker.mock_open(read_data=json.dumps(config_data)))
    mocker.patch("json.load", return_value=config_data)
    mocker.patch("shutil.copy")
    mock_dependencies["load_data"].return_value = []
    # Make runner return an empty DataFrame
    mock_dependencies["runner_inst"].run_all.return_value.empty = True

    run.main(config_json=config_path)

    mock_dependencies["Runner"].assert_called()
    mock_dependencies["visualize"].assert_not_called()
    mock_dependencies["load_data"].assert_called_once_with(
        config_data["input_file"], config_data.get("limit")
    )


def test_run_main_unknown_retriever_type(mock_dependencies, mocker, capsys):
    import sys

    with open(config_path) as f:
        config_data = json.load(f)
    config_data_invalid = dict(config_data)
    config_data_invalid["retriever_type"] = "invalid_type"
    mocker.patch("builtins.open", mocker.mock_open(read_data=json.dumps(config_data_invalid)))
    mocker.patch("json.load", return_value=config_data_invalid)
    mocker.patch("shutil.copy")
    original_exit = sys.exit
    sys.exit = lambda code=1: (_ for _ in ()).throw(SystemExit(code))
    try:
        with pytest.raises(SystemExit) as excinfo:
            run.main(config_json=config_path)
        assert excinfo.value.code == 1
        captured = capsys.readouterr()
        assert "Error: Unknown retriever type" in captured.out
    finally:
        sys.exit = original_exit


def test_config_is_copied_to_results_local_environment(mock_dependencies, mocker):
    """Test config copying in local environment (no /workspace directory)."""
    with open(config_path) as f:
        config_data = json.load(f)
    mocker.patch("builtins.open", mocker.mock_open(read_data=json.dumps(config_data)))
    mocker.patch("json.load", return_value=config_data)
    mock_copy = mocker.patch("shutil.copy")
    mocker.patch("os.path.join", side_effect=os.path.join)

    # Mock os.path.exists to simulate local environment (no /workspace)
    mock_exists = mocker.patch("os.path.exists")
    mock_exists.return_value = False

    run.main(config_json=config_path)

    # Check that the config file is copied to the correct results directory
    output_dir = mock_dependencies["create_dir"].return_value  # Use mock_dir directly
    expected_dest = os.path.join(output_dir, "experiment_config_test_experiment_config.json")
    mock_copy.assert_called_once_with(config_path, expected_dest)


def test_run_caching_skips_execution(mock_dependencies, mocker, capsys):
    """Test that execution is skipped if the results directory already exists."""
    with open(config_path) as f:
        config_data = json.load(f)

    # We set input_file to 'gold.jsonl' implies prefix 'test-model_test_experiment_config_gold'

    mocker.patch("builtins.open", mocker.mock_open(read_data=json.dumps(config_data)))
    mocker.patch("json.load", return_value=config_data)

    # Mock os.path.exists
    mock_exists = mocker.patch("os.path.exists")

    def side_effect(path):
        # Check if checking for results directory
        if "results" in path and "test-model_test_experiment_config_gold" in path:
            return True
        return False

    mock_exists.side_effect = side_effect

    run.main(config_json=config_path)

    # Verify that create_output_directory was NOT called (so no new results dir)
    mock_dependencies["create_dir"].assert_not_called()

    # Verify processing was skipped (Runner not initialized)
    mock_dependencies["Runner"].assert_not_called()

    # Verify message
    captured = capsys.readouterr()
    assert "Skipping experiment" in captured.out
    assert "already exists" in captured.out

