import json
import logging
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
MASTER_CSV_PATH = LOCAL_DATA_DIR / "master_results.csv"

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

    def __init__(self, bucket_name: str, prefix: str = "data"):
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)

    def download_metadata_and_results(self) -> None:
        logger.info(f"Starting download from bucket: {self.bucket_name}")
        blobs = self.bucket.list_blobs(prefix=self.prefix)

        for blob in blobs:
            if self._is_relevant_file(blob.name):
                self._download_blob(blob)

    def _is_relevant_file(self, filename: str) -> bool:
        if "/archive/" in filename:
            return False

        return (
                filename.endswith("metadata.json") or
                filename.endswith("_detailed_resulsts.csv")
        )

    def _download_blob(self, blob: Any) -> None:
        local_path = Path(blob.name)
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
        """Reads /data/indices/$experimentName_$embeddingModel/metadata.json"""
        folder_name = f"{experiment_name}_{embedding_model}"
        path = self.indices_dir / folder_name / "metadata.json"
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
            if folder.is_dir() and folder.name not in ("archive", "_archive"):
                all_rows.extend(self._process_experiment_folder(folder))

        return pd.DataFrame(all_rows)

    def _process_experiment_folder(self, folder_path: Path) -> list[dict[str, Any]]:
        # 1. Read Folder Metadata
        folder_meta = FileSystemManager.read_json(folder_path / "metadata.json")
        embedding_model = folder_meta.get("embedding_model", "unknown")
        dataset_size = folder_meta.get("dataset_size")
        difficulty = folder_meta.get("difficulty")

        # 2. Find CSVs (handle naming inconsistencies)
        csv_files = list(folder_path.glob("*_detailed_resulsts.csv")) + \
                    list(folder_path.glob("*_detailed_results.csv"))

        enriched_rows = []
        for csv_file in csv_files:
            try:
                # pandas reads ALL columns automatically, including 'retrieval_time'
                df = pd.read_csv(csv_file)
                for _, row in df.iterrows():
                    enriched_rows.append(
                        self._enrich_row(row.to_dict(), embedding_model, difficulty, dataset_size)
                    )
            except Exception as e:
                logger.error(f"Error processing {csv_file}: {e}")

        return enriched_rows

    def _enrich_row(self, row: dict[str, Any], embedding: str, difficulty: str, size: Any) -> dict[str, Any]:
        experiment_name = row.get("experiment")

        # Add Context
        row["embedding_model"] = embedding
        row["config_difficulty"] = difficulty
        row["dataset_size"] = size

        if experiment_name:
            # Add Chunking Data
            row.update(self.metadata_service.get_chunk_metadata(str(experiment_name)))
            # Add Indexing Data
            row.update(self.metadata_service.get_index_metadata(str(experiment_name), embedding))

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
