import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

import build_indicies_SORTED


@pytest.fixture
def sorted_config_and_inputs(tmp_path: Path):
    config = {
        "embedding_model": "dummy-model",
        "experiments": [
            {
                "name": "exp1",
                "function": "chunk_fixed_size",
                "params": {"chunk_size": 10, "chunk_overlap": 0},
            }
        ],
    }

    dataset = [
        {"document_text": "aaaa"},
        {"document_text": "bbbbbbbb"},
    ]

    cache_dir = tmp_path / "data" / "chunks"
    out_dir = tmp_path / "data" / "indices"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    return config, dataset, out_dir, cache_dir


def test_process_experiment_writes_sorted_cache_and_artifacts(
    sorted_config_and_inputs, monkeypatch
):
    config, dataset, out_dir, cache_dir = sorted_config_and_inputs
    exp = config["experiments"][0]

    # Make chunking deterministic and *unsorted* so we can verify sorting persisted.
    monkeypatch.setattr(
        build_indicies_SORTED,
        "get_chunking_function",
        lambda name: (lambda text, **kwargs: ["zzzz", "a", "mmmm"]),
    )

    # Avoid progress bar noise and dependency on tqdm internals.
    monkeypatch.setattr(build_indicies_SORTED, "tqdm", lambda x, **kwargs: x)

    # Avoid heavy FAISS/vectorizer GPU work.
    dummy_index = SimpleNamespace(ntotal=123)

    def fake_build_index_dynamic(chunks, vec, model_name):
        # ensure chunks given to builder are already globally sorted by length
        assert chunks == sorted(chunks, key=len)
        return dummy_index, 0.01

    monkeypatch.setattr(build_indicies_SORTED, "build_index_dynamic", fake_build_index_dynamic)

    # Capture save_artifacts call and write minimal expected files.
    def fake_save_artifacts(index, index_dir, chunks, sorted_filename, build_time):
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)

        # mimic the real function writing metadata
        metadata = {
            "indexing_duration": build_time,
            "num_chunks": len(chunks),
            "linked_cache_file": sorted_filename,
            "faiss_ntotal": index.ntotal,
        }
        (index_dir / "metadata.json").write_text(json.dumps(metadata))
        (index_dir / "index.faiss").write_text("dummy")

    monkeypatch.setattr(build_indicies_SORTED, "save_artifacts", fake_save_artifacts)

    # Minimal vec placeholder (unused due to patched builder)
    vec = mock.Mock()

    build_indicies_SORTED.process_experiment(
        exp=exp,
        config=config,
        dataset=dataset,
        vec=vec,
        out_dir=str(out_dir),
        cache_dir=str(cache_dir),
    )

    # Verify sorted cache file exists and is sorted by length
    cache_name = f"{exp['name']}_{exp['name']}_{exp['function']}_chunks.json"
    sorted_name = cache_name.replace(".json", "_SORTED.json")
    sorted_path = cache_dir / sorted_name
    assert sorted_path.exists(), "Expected sorted cache file to be written"

    sorted_chunks = json.loads(sorted_path.read_text())
    assert sorted_chunks == sorted(sorted_chunks, key=len)

    # Verify artifacts were written to the expected index dir
    index_dir = out_dir / build_indicies_SORTED.create_index_name(
        exp["name"], config["embedding_model"]
    )
    assert (index_dir / "metadata.json").exists()
    assert (index_dir / "index.faiss").exists()

    metadata = json.loads((index_dir / "metadata.json").read_text())
    assert metadata["num_chunks"] == len(sorted_chunks)
    assert metadata["linked_cache_file"] == sorted_name


def test_process_experiment_skips_if_index_exists(sorted_config_and_inputs, monkeypatch):
    config, dataset, out_dir, cache_dir = sorted_config_and_inputs
    exp = config["experiments"][0]

    # Create existing index marker
    index_dir = out_dir / build_indicies_SORTED.create_index_name(
        exp["name"], config["embedding_model"]
    )
    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / "index.faiss").write_text("already-there")

    # Ensure we would fail if builder was called
    monkeypatch.setattr(
        build_indicies_SORTED,
        "build_index_dynamic",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("should not build")),
    )

    # Avoid tqdm
    monkeypatch.setattr(build_indicies_SORTED, "tqdm", lambda x, **kwargs: x)

    vec = mock.Mock()

    # Should return without raising
    build_indicies_SORTED.process_experiment(
        exp=exp,
        config=config,
        dataset=dataset,
        vec=vec,
        out_dir=str(out_dir),
        cache_dir=str(cache_dir),
    )
