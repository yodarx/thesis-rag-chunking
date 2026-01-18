import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from google.cloud import storage  # type: ignore

# --- Configuration & Constants ---
LOCAL_DATA_DIR = Path("data")
CHUNKS_DIR = LOCAL_DATA_DIR / "chunks"
INDICES_DIR = LOCAL_DATA_DIR / "indices"
RESULTS_DIR = LOCAL_DATA_DIR / "results"
MASTER_CSV_PATH = RESULTS_DIR / "master_results.csv"

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FileSystemManager:
    """Handles low-level file system operations."""

    @staticmethod
    def ensure_directory(path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def read_json(path: Path) -> dict[str, Any]:
        try:
            with open(path, encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.debug(f"Could not read JSON at {path}: {e}")
            return {}


class GCSDownloader:
    """Responsible for downloading relevant artifacts from Google Cloud Storage."""

    def __init__(self, bucket_name: str, prefix: str = "results", local_base_dir: Path = LOCAL_DATA_DIR):
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.local_base_dir = local_base_dir
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)

    def download_metadata_and_results(self) -> None:
        logger.info(f"Starting download from bucket: {self.bucket_name}")

        # We need to check results, chunks, and indices folders
        prefixes = ["results", "chunks", "indices"]

        for prefix in prefixes:
            logger.info(f"Checking prefix: {prefix}")
            blobs = self.bucket.list_blobs(prefix=prefix)

            for blob in blobs:
                if self._is_relevant_file(blob.name):
                    self._download_blob(blob)

    def _is_relevant_file(self, filename: str) -> bool:
        if "/archive/" in filename:
            return False

        return (
                filename.endswith("metadata.json") or
                filename.endswith("_detailed_results.csv")
        )

    def _download_blob(self, blob: Any) -> None:
        # Determine local path with potential renaming
        blob_name = blob.name

        # Apply renaming rules
        if blob_name.startswith("chunks/") and blob_name.endswith("/metadata.json"):
            # chunks/experiment/metadata.json -> chunks/experiment/chunks_metadata.json
            blob_name = blob_name.replace("/metadata.json", "/chunks_metadata.json")
        elif blob_name.startswith("indices/") and blob_name.endswith("/metadata.json"):
            # indices/exp_model/metadata.json -> indices/exp_model/index_metadata.json
            blob_name = blob_name.replace("/metadata.json", "/index_metadata.json")

        # Prepend local_base_dir (e.g. "data") to the blob name so it lands in data/results/...
        local_path = self.local_base_dir / blob_name
        FileSystemManager.ensure_directory(local_path.parent)

        if not local_path.exists():
            blob.download_to_filename(str(local_path))
            logger.info(f"Downloaded: {local_path}")


class MetadataService:
    """Provides specific metadata lookups for Experiments and Indices."""

    def __init__(self, base_dir: Path):
        self.chunks_dir = base_dir / "chunks"
        self.indices_dir = base_dir / "indices"

    def get_chunk_metadata(self, experiment_name: str) -> dict[str, Any]:
        """Reads /data/chunks/$experimentName/metadata.json"""
        path = self.chunks_dir / experiment_name / "metadata.json"
        data = FileSystemManager.read_json(path)

        return {
            "chunk_processing_time_s": data.get("processing_time_seconds"),
            "chunk_avg_chars": data.get("avg_chars_per_chunk"),
            "chunk_total_count": data.get("total_chunks")
        }

    def get_index_metadata(self, experiment_name: str, embedding_model: str) -> dict[str, Any]:
        """Reads /data/indices/$experimentName_$embeddingModel/index_metadata.json"""
        folder_name = f"{experiment_name}_{embedding_model}"
        path = self.indices_dir / folder_name / "index_metadata.json"
        data = FileSystemManager.read_json(path)

        return {
            "indexing_duration_s": data.get("indexing_duration"),
            "index_faiss_ntotal": data.get("faiss_ntotal"),
            "index_optimization": data.get("optimization")
        }


class ResultProcessor:
    """Aggregates individual result CSVs and enriches them with metadata."""

    def __init__(self, metadata_service: MetadataService):
        self.metadata_service = metadata_service

    def process_all_results(self, results_root: Path) -> pd.DataFrame:
        if not results_root.exists():
            logger.warning(f"Results directory not found: {results_root}")
            return pd.DataFrame()

        all_rows = []
        for folder in results_root.iterdir():
            if folder.is_dir() and "archive" not in folder.name:
                all_rows.extend(self._process_experiment_folder(folder))

        return pd.DataFrame(all_rows)

    def _process_experiment_folder(self, folder_path: Path) -> list[dict[str, Any]]:
        folder_name = folder_path.name

        match = re.search(r'^(.*?)_(0[12]_.*)$', folder_name)
        if match:
            embedding_model = match.group(1)
            rest = match.group(2)
        else:
            logger.warning(f"Could not parse folder name: {folder_name}")
            # Fallback to metadata if available
            folder_meta = FileSystemManager.read_json(folder_path / "metadata.json")
            embedding_model = folder_meta.get("embedding_model", "unknown")
            rest = folder_name

        # Parse difficulty from 'rest'
        difficulty = "All"
        experiment_config = rest

        for diff in ["Easy", "Hard", "Moderate"]:
            if rest.endswith(f"_{diff}"):
                difficulty = diff
                # Remove suffix: 01_gold_easy_Easy -> 01_gold_easy
                experiment_config = rest[:-(len(diff)+1)]
                break

        # 2. Find CSVs (handle naming inconsistencies)
        csv_files = list(folder_path.glob("*_detailed_results.csv"))

        enriched_rows = []
        for csv_file in csv_files:
            try:
                # pandas reads ALL columns automatically.
                # Use sep='\t' for tab-separated files as per user sample
                # The prompt shows tab-like structure for the CSV.
                try:
                    df = pd.read_csv(csv_file, sep='\t')
                    if len(df.columns) < 2:
                        # Fallback to comma if tab parse failed (e.g. single col read)
                        df = pd.read_csv(csv_file)
                except Exception:
                    df = pd.read_csv(csv_file)

                for _, row in df.iterrows():
                    enriched_rows.append(
                        self._enrich_row(row.to_dict(), embedding_model, difficulty, experiment_config)
                    )
            except Exception as e:
                logger.error(f"Error processing {csv_file}: {e}")

        return enriched_rows

    def _enrich_row(self, row: dict[str, Any], embedding: str, difficulty: str, config_name: str) -> dict[str, Any]:
        chunking_strategy = row.get("experiment") # e.g. 'sentence_s1'

        # Add Context
        row["embedding_model"] = embedding
        row["config_difficulty"] = difficulty
        row["experiment_config"] = config_name

        # Determine dataset type (Gold/Silver)
        config_lower = config_name.lower()
        if "gold" in config_lower:
            row["dataset_type"] = "Gold"
        elif "silver" in config_lower:
            row["dataset_type"] = "Silver"
        else:
            row["dataset_type"] = "Unknown"

        if chunking_strategy:
            # Add Chunking Data (avg_chars, etc not in CSV)
            # Row already has chunking_time_s, num_chunks
            chunk_meta = self.metadata_service.get_chunk_metadata(str(chunking_strategy))
            # Only add keys that don't exist? Or overwrite?
            # CSV has `chunking_time_s`. Metadata has `chunk_processing_time_s`.
            # I'll add all, user can decide which to use.
            row.update(chunk_meta)

            # Add Indexing Data
            row.update(self.metadata_service.get_index_metadata(str(chunking_strategy), embedding))

        return row


def main(bucket_name: str, skip_download: bool = False):
    if not skip_download:
        downloader = GCSDownloader(bucket_name)
        downloader.download_metadata_and_results()

    metadata_service = MetadataService(LOCAL_DATA_DIR)
    processor = ResultProcessor(metadata_service)

    logger.info("Aggregating results...")
    master_df = processor.process_all_results(RESULTS_DIR)

    if not master_df.empty:
        master_df.to_csv(MASTER_CSV_PATH, index=False)
        logger.info(f"Saved master CSV to {MASTER_CSV_PATH} ({len(master_df)} rows)")
    else:
        logger.warning("No data found.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("Usage: python rag_aggregator.py <bucket_name>")
