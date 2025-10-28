from unittest.mock import Mock

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
    mock_get_exps = mocker.patch("run.get_experiments")
    mock_visualize = mocker.patch("run.visualize_and_save_results")
    mock_create_dir = mocker.patch("run._create_output_directory")

    # Konfiguriere Rückgabewerte
    mock_create_dir.return_value = ("mock_dir", "mock_ts")
    mock_get_exps.return_value = [{"name": "mock_exp"}]
    mock_load_data.return_value = [{"id": 1}]

    # Mocke die Instanzen, die von den Konstruktoren zurückgegeben werden
    mock_runner_inst = mock_runner_cls.return_value

    # Konfiguriere den Runner, ein (Mock) Summary-DataFrame zurückzugeben
    mock_summary_df = mocker.Mock(spec=pd.DataFrame)
    mock_summary_df.empty = False
    mock_runner_inst.run_all.return_value = mock_summary_df

    return {
        "load_data": mock_load_data,
        "get_exps": mock_get_exps,
        "visualize": mock_visualize,
        "create_dir": mock_create_dir,
        "Vector": mock_vectorizer_cls,
        "Retriever": mock_retriever_cls,
        "Results": mock_results_cls,
        "Runner": mock_runner_cls,
        "runner_inst": mock_runner_inst,
        "summary_df": mock_summary_df,
    }


def test_run_main_workflow(mock_dependencies):
    """Testet den gesamten Ablauf der main()-Funktion in run.py."""

    dummy_input_path = "dummy/input.jsonl"
    run.main(
        input_filepath=dummy_input_path,
        limit=None,
        embedding_model_name="mock_model",
        retriever_type="faiss",
        top_k=5,
    )

    # --- 1. Setup-Phase prüfen ---
    mock_dependencies["create_dir"].assert_called_once()
    mock_dependencies["get_exps"].assert_called_once()
    mock_dependencies["load_data"].assert_called_once_with(dummy_input_path, limit=None)

    # --- 2. Initialisierungs-Phase prüfen ---
    mock_dependencies["Vector"].from_model_name.assert_called_once()
    mock_dependencies["Retriever"].assert_called_once()
    mock_dependencies["Results"].assert_called_once_with("mock_dir", "mock_ts")

    # --- 3. Runner-Initialisierung prüfen ---
    mock_dependencies["Runner"].assert_called_once()
    runner_call_args = mock_dependencies["Runner"].call_args[1]  # Check keyword args
    assert runner_call_args["dataset"] == [{"id": 1}]
    assert runner_call_args["experiments"] == [{"name": "mock_exp"}]
    assert isinstance(runner_call_args["vectorizer"], Mock)
    assert isinstance(runner_call_args["retriever"], Mock)
    assert isinstance(runner_call_args["results_handler"], Mock)

    # --- 4. Ausführungs-Phase prüfen ---
    mock_dependencies["runner_inst"].run_all.assert_called_once()

    # --- 5. Visualisierungs-Phase prüfen ---
    mock_dependencies["visualize"].assert_called_once_with(
        mock_dependencies["summary_df"], "mock_dir", "mock_ts"
    )


def test_run_main_no_data(mock_dependencies):
    """Testet, ob das Skript korrekt abbricht, wenn keine Daten geladen werden."""
    mock_dependencies["load_data"].return_value = []
    dummy_input_path = "dummy/input.jsonl"
    run.main(
        input_filepath=dummy_input_path,
        limit=None,
        embedding_model_name="mock_model",
        retriever_type="faiss",
        top_k=5,
    )

    mock_dependencies["Runner"].assert_not_called()
    mock_dependencies["visualize"].assert_not_called()
    mock_dependencies["load_data"].assert_called_once_with(dummy_input_path, limit=None)


def test_run_main_unknown_retriever_type(mock_dependencies, capsys):
    """Test that main() exits with error for unknown retriever_type."""
    import sys

    dummy_input_path = "dummy/input.jsonl"
    original_exit = sys.exit
    sys.exit = lambda code=1: (_ for _ in ()).throw(SystemExit(code))
    try:
        with pytest.raises(SystemExit) as excinfo:
            run.main(
                input_filepath=dummy_input_path,
                limit=None,
                embedding_model_name="mock_model",
                retriever_type="unknown",
                top_k=5,
            )
        assert excinfo.value.code == 1
        captured = capsys.readouterr()
        assert "Error: Unknown retriever type" in captured.out
    finally:
        sys.exit = original_exit
