import pandas as pd
import pytest
from pytest_mock import MockerFixture

from src.experiment.results import ResultsHandler


@pytest.fixture
def sample_data_point() -> dict:
    return {"sample_id": "s1", "question": "Test?"}


@pytest.fixture
def sample_metrics() -> dict:
    return {
        "mrr": 0.5,
        "map": 0.5,
        "ndcg_at_k": 0.5,
        "precision_at_k": 0.5,
        "recall_at_k": 0.5,
        "f1_score_at_k": 0.5,
    }


@pytest.fixture
def mock_pandas(mocker: MockerFixture):
    """Mockt pandas DataFrame-Erstellung und .to_csv()."""
    mock_df_instance = mocker.Mock(spec=pd.DataFrame)
    mock_df_constructor = mocker.patch("pandas.DataFrame", return_value=mock_df_instance)
    return mock_df_constructor, mock_df_instance


def test_results_handler_init():
    handler = ResultsHandler("test_dir", "test_ts")
    assert handler.output_dir == "test_dir"
    assert handler.timestamp == "test_ts"
    assert handler.all_results == []


def test_add_result_record(sample_data_point, sample_metrics):
    handler = ResultsHandler("test_dir", "test_ts")
    handler.add_result_record(
        data_point=sample_data_point,
        experiment_name="test_exp",
        chunking_time=0.123,
        num_chunks=10,
        metrics=sample_metrics,
    )

    assert len(handler.all_results) == 1
    result = handler.all_results[0]
    assert result["sample_id"] == "s1"
    assert result["experiment"] == "test_exp"
    assert result["chunking_time_s"] == 0.123
    assert result["mrr"] == 0.5


def test_save_detailed_results(mock_pandas, sample_data_point, sample_metrics, mocker):
    mock_df_constructor, mock_df_instance = mock_pandas
    mock_join = mocker.patch("os.path.join", return_value="test_dir/test_ts_detailed.csv")

    handler = ResultsHandler("test_dir", "test_ts")
    handler.add_result_record(sample_data_point, "test_exp", 0.1, 10, sample_metrics)

    df = handler.save_detailed_results()

    mock_join.assert_called_once_with("test_dir", "test_ts_detailed_results.csv")
    mock_df_constructor.assert_called_once_with(handler.all_results)
    mock_df_instance.to_csv.assert_called_once_with("test_dir/test_ts_detailed.csv", index=False)
    assert df is mock_df_instance


def test_create_and_save_summary(mocker):
    mock_join = mocker.patch("os.path.join", return_value="test_dir/test_ts_summary.csv")

    input_data = [
        {
            "experiment": "exp1",
            "mrr": 1.0,
            "map": 1.0,
            "ndcg_at_k": 1.0,
            "precision_at_k": 1.0,
            "recall_at_k": 1.0,
            "f1_score_at_k": 1.0,
            "chunking_time_s": 0.1,
            "num_chunks": 10,
        },
        {
            "experiment": "exp1",
            "mrr": 0.0,
            "map": 0.0,
            "ndcg_at_k": 0.0,
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "f1_score_at_k": 0.0,
            "chunking_time_s": 0.3,
            "num_chunks": 20,
        },
    ]
    input_df = pd.DataFrame(input_data)  # Real DataFrame

    mock_final_df = mocker.Mock(spec=pd.DataFrame)
    mock_agg_result = mocker.Mock()
    mock_agg_result.reset_index.return_value = mock_final_df  # Link to final mock

    mock_grouped_object = mocker.Mock()
    mock_grouped_object.agg.return_value = mock_agg_result  # Link to agg result
    mocker.patch.object(input_df, "groupby", return_value=mock_grouped_object)

    # --- Execute ---
    handler = ResultsHandler("test_dir", "test_ts")
    df = handler.create_and_save_summary(input_df)

    # --- Assertions ---
    input_df.groupby.assert_called_once_with("experiment")  # This should now pass!
    mock_grouped_object.agg.assert_called_once()
    mock_agg_result.reset_index.assert_called_once()

    mock_join.assert_called_once_with("test_dir", "test_ts_summary.csv")
    mock_final_df.to_csv.assert_called_once_with("test_dir/test_ts_summary.csv", index=False)

    assert df is mock_final_df
