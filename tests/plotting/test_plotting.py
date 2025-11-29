import pandas as pd
import pytest
from pytest_mock import MockerFixture

from plotting import plotting

# --- Fixture: Test-Daten ---


@pytest.fixture
def sample_summary_df() -> pd.DataFrame:
    """Erstellt ein Beispieldaten-DataFrame für die Tests."""
    data = {
        "experiment": ["strategy_A", "strategy_B"],
        "chunking_time_s": [0.1, 0.5],
        "mrr": [0.85, 0.90],
        "map": [0.80, 0.85],
        "ndcg_at_k": [0.82, 0.88],
        "precision_at_k": [0.75, 0.80],
        "recall_at_k": [0.88, 0.92],
        "f1_score_at_k": [0.81, 0.85],
    }
    return pd.DataFrame(data)


# --- Mocks für die Tests ---


@pytest.fixture
def mock_plotting(mocker: MockerFixture):
    """
    Mockt die globalen Plotting-Objekte (plt, sns) und os.path.join,
    damit keine echten Dateien erstellt werden.
    """
    mock_plt = mocker.patch("plotting.plotting.plt")
    mock_sns = mocker.patch("plotting.plotting.sns")

    mock_os_path_join = mocker.patch(
        "plotting.plotting.os.path.join", side_effect=lambda *args: "/".join(args)
    )

    # Mock die bar_label-Funktionalität, da sie aufgerufen wird
    mock_bar_plot = mocker.Mock()
    mock_bar_plot.containers = [mocker.Mock()]
    mock_sns.barplot.return_value = mock_bar_plot

    return {
        "plt": mock_plt,
        "sns": mock_sns,
        "os_path_join": mock_os_path_join,
        "bar_plot": mock_bar_plot,
    }


# --- Tests ---


def test_visualize_and_save_results_main_loop(
    mocker: MockerFixture, sample_summary_df: pd.DataFrame
):
    """
    Testet die Hauptfunktion, um sicherzustellen, dass sie die
    Helferfunktionen für jede Metrik korrekt aufruft.
    """
    # Mocke die *privaten Helferfunktionen* des Moduls, nicht die Plots selbst
    mock_bar = mocker.patch("plotting.plotting._create_and_save_bar_plot")
    mock_print = mocker.patch("builtins.print")

    test_dir = "test_output"
    test_ts = "2025-10-23"

    plotting.visualize_and_save_results(sample_summary_df, test_dir, test_ts)

    # Überprüfen, ob die Helfer für jede Metrik aufgerufen wurden
    assert mock_bar.call_count == len(plotting.METRICS_TO_PLOT)

    # Stichprobenartig den ersten Aufruf prüfen
    first_metric = plotting.METRICS_TO_PLOT[0]
    first_display_name = plotting.METRIC_DISPLAY_NAMES[first_metric]

    mock_bar.assert_any_call(
        df=sample_summary_df,
        metric=first_metric,
        display_name=first_display_name,
        output_dir=test_dir,
        timestamp=test_ts,
    )

    # Überprüfen, ob die Erfolgsmeldung gedruckt wurde
    mock_print.assert_any_call("\nAll plots were successfully saved.")


def test_visualize_skips_missing_metrics(mocker: MockerFixture, sample_summary_df: pd.DataFrame):
    """
    Testet, ob Metriken, die nicht im DataFrame sind, korrekt
    übersprungen werden und eine Warnung ausgegeben wird.
    """
    mock_bar = mocker.patch("plotting.plotting._create_and_save_bar_plot")
    mock_print = mocker.patch("builtins.print")

    # Entferne eine Metrik aus dem Test-DataFrame
    df_missing = sample_summary_df.drop(columns=["mrr"])

    plotting.visualize_and_save_results(df_missing, "test_dir", "test_ts")

    # Sollte für alle Metriken *außer* der fehlenden aufgerufen werden
    expected_calls = len(plotting.METRICS_TO_PLOT) - 1
    assert mock_bar.call_count == expected_calls

    # Überprüfe, ob die Warnung gedruckt wurde
    mock_print.assert_any_call("Warning: Metric 'mrr' not found in summary. Skipping plot.")


def test_create_and_save_bar_plot(sample_summary_df: pd.DataFrame, mock_plotting: dict):
    """Testet die Erstellung des Balkendiagramms im Detail."""
    test_dir = "test_output"
    test_ts = "2025-10-23"
    test_metric = "mrr"
    test_display = "MRR Test"

    plotting._create_and_save_bar_plot(
        sample_summary_df, test_metric, test_display, test_dir, test_ts
    )

    mock_sns = mock_plotting["sns"]
    mock_plt = mock_plotting["plt"]
    mock_os_path_join = mock_plotting["os_path_join"]
    mock_bar_plot = mock_plotting["bar_plot"]

    # 1. Wurde das Plotting korrekt aufgerufen?
    mock_sns.barplot.assert_called_once()
    assert mock_sns.barplot.call_args.kwargs["x"] == test_metric
    assert mock_sns.barplot.call_args.kwargs["y"] == "experiment"

    # 2. Wurden Titel und Labels gesetzt?
    mock_plt.title.assert_called_with(
        f"Performance Comparison: {test_display}", fontsize=16, pad=20
    )
    mock_plt.xlabel.assert_called_with(test_display, fontsize=12)

    # 3. Wurden die Bar-Labels aufgerufen?
    mock_bar_plot.bar_label.assert_called_once()

    # 4. Wurde die Datei korrekt gespeichert?
    expected_filename = f"{test_dir}/{test_ts}_{test_metric}_barplot.png"
    mock_os_path_join.assert_called_with(test_dir, f"{test_ts}_{test_metric}_barplot.png")
    mock_plt.savefig.assert_called_with(expected_filename)
    mock_plt.close.assert_called_once()
