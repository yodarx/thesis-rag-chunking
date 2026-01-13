import pytest
import os
from run import generate_experiment_prefix

def test_generate_experiment_prefix_standard():
    config = {
        "embedding_model": "BAAI/bge-large-en-v1.5",
        "input_file": "data/gold/test_gold.jsonl"
    }
    config_path = "configs/experiment.json"

    prefix = generate_experiment_prefix(config, config_path)
    # embedding: BAAI_bge-large-en-v1.5 -> BAAI_bge-large-en-v1.5 (replace only if needed, but my logic was replace("/", "_"))
    # BAAI/bge-large-en-v1.5 -> BAAI_bge-large-en-v1.5
    # experiment: experiment
    # input: test_gold
    # expected: BAAI_bge-large-en-v1.5_experiment_test_gold
    assert prefix == "BAAI_bge-large-en-v1.5_experiment_test_gold"

def test_generate_experiment_prefix_with_difficulty():
    config = {
        "embedding_model": "mini",
        "input_file": "data/silver/silver_data.jsonl"
    }
    config_path = "configs/h2.json"
    difficulty = "Hard"

    prefix = generate_experiment_prefix(config, config_path, difficulty=difficulty)
    # embedding: mini
    # experiment: h2
    # input: silver_data
    # difficulty: Hard
    # expected: mini_h2_silver_data_Hard
    assert prefix == "mini_h2_silver_data_Hard"

def test_generate_experiment_prefix_missing_input_file():
    config = {
        "embedding_model": "model"
    }
    config_path = "config.json"

    prefix = generate_experiment_prefix(config, config_path)
    # embedding: model
    # experiment: config
    # input: unknown_input
    assert prefix == "model_config_unknown_input"

def test_generate_experiment_prefix_no_config_path():
    config = {
        "embedding_model": "model",
        "input_file": "data.json"
    }

    prefix = generate_experiment_prefix(config, "")
    # embedding: model
    # experiment: unknown_experiment (if empty string)
    # input: data
    assert prefix == "model_unknown_experiment_data"

