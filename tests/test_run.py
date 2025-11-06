import json
import os

import pandas as pd
import pytest
from pytest_mock import MockerFixture

import run


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
    mock_create_dir = mocker.patch("run._create_output_directory")

    # Konfiguriere Rückgabewerte
    mock_create_dir.return_value = ("mock_dir", "mock_ts")
    mock_load_data.return_value = [{"id": 1}]

    # Mocke die Instanzen, die von den Konstruktoren zurückgegeben werden
    mock_runner_inst = mock_runner_cls.return_value

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
    config_path = "configs/base_experiment_limit_5.json"
    with open(config_path) as f:
        config_data = json.load(f)
    mocker.patch("builtins.open", mocker.mock_open(read_data=json.dumps(config_data)))
    mocker.patch("json.load", return_value=config_data)
    mocker.patch("shutil.copy")

    run.main(config_json=config_path)

    mock_dependencies["create_dir"].assert_called_once()
    mock_dependencies["load_data"].assert_called_once_with(
        config_data["input_file"], limit=config_data["limit"]
    )
    mock_dependencies["Vector"].from_model_name.assert_called_once()
    mock_dependencies["Retriever"].assert_called_once()
    mock_dependencies["Results"].assert_called_once_with("mock_dir", "mock_ts")
    mock_dependencies["Runner"].assert_called_once()
    mock_dependencies["runner_inst"].run_all.assert_called_once()
    mock_dependencies["visualize"].assert_called_once_with(
        mock_dependencies["summary_df"], "mock_dir", "mock_ts"
    )


def test_run_main_no_data(mock_dependencies, mocker):
    config_path = "configs/base_experiment_limit_5.json"
    with open(config_path) as f:
        config_data = json.load(f)
    mocker.patch("builtins.open", mocker.mock_open(read_data=json.dumps(config_data)))
    mocker.patch("json.load", return_value=config_data)
    mocker.patch("shutil.copy")
    mock_dependencies["load_data"].return_value = []

    run.main(config_json=config_path)

    mock_dependencies["Runner"].assert_not_called()
    mock_dependencies["visualize"].assert_not_called()
    mock_dependencies["load_data"].assert_called_once_with(
        config_data["input_file"], limit=config_data["limit"]
    )


def test_run_main_unknown_retriever_type(mock_dependencies, mocker, capsys):
    import sys

    config_path = "configs/base_experiment_limit_5.json"
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


def test_config_is_copied_to_results(mock_dependencies, mocker):
    config_path = "configs/base_experiment_limit_5.json"
    with open(config_path) as f:
        config_data = json.load(f)
    mocker.patch("builtins.open", mocker.mock_open(read_data=json.dumps(config_data)))
    mocker.patch("json.load", return_value=config_data)
    mock_copy = mocker.patch("shutil.copy")
    mocker.patch("os.path.join", side_effect=os.path.join)

    run.main(config_json=config_path)

    # Check that the config file is copied to the results directory
    output_dir = mock_dependencies["create_dir"].return_value[0]
    expected_dest = os.path.join(output_dir, "experiment_config.json")
    mock_copy.assert_called_once_with(config_path, expected_dest)


def test_run_all_configs_sequential(mocker, mock_dependencies):
    config_files = [
        "1_full_parameter_sweep.json",
        "2_model_sensitivity_bge_large.json",
        "3_top_k_sensitivity_k1.json",
    ]
    config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs")
    mocker.patch("os.listdir", return_value=config_files)
    mocker.patch("os.path.dirname", return_value=os.path.dirname(os.path.dirname(__file__)))

    # Simulate CLI args
    import sys

    sys.argv = ["run.py", "--run-all-configs"]
    import importlib

    importlib.reload(run)
    mock_main = mocker.patch("run.main")
    run.cli_entry()
    expected_calls = [
        mocker.call(config_json=os.path.join(config_dir, f)) for f in sorted(config_files)
    ]
    actual_calls = [call for call in mock_main.call_args_list]
    for expected, actual in zip(expected_calls, actual_calls, strict=True):
        if expected != actual:
            assert actual.args[0] == expected.kwargs["config_json"]
    assert mock_main.call_count == len(config_files)
