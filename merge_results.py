import os
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# --- PFADE ---
BASE_DIR = Path(__file__).parent.resolve()
RESULTS_DIR = BASE_DIR / 'results'
INDICES_DIR = BASE_DIR / 'data/indices'
OUTPUT_FILE = BASE_DIR / "merged_thesis_data_final.csv"

# Mapping: Wie heißt der Ordner für das jeweilige Modell in der CSV?
# Links: Name in der CSV ("model_name")
# Rechts: Name des Ordners unter data/indices
MODEL_FOLDER_MAP = {
    "all-MiniLM-L6-v2": "all_MiniLM",
    "BAAI/bge-base-en-v1.5": "bge-base",
    "BAAI/bge-large-en-v1.5": "bge-large"
}


def get_index_data(model_name_csv, experiment_name):
    """
    Sucht die JSON-Datei, indem es in den Ordner schaut und prüft,
    ob eine Datei mit '{experiment_name}_' beginnt.
    """
    # 1. Bestimme den Ordner
    folder_name = MODEL_FOLDER_MAP.get(model_name_csv)
    if not folder_name:
        return None, None  # Unbekanntes Modell

    target_dir = INDICES_DIR / folder_name
    if not target_dir.exists():
        return None, None

    # 2. Suche Datei, die mit dem Experiment-Namen beginnt
    # Wir suchen nach "fixed_128_0_" (mit Underscore am Ende, um s1 vs s10 zu trennen)
    prefix = f"{experiment_name}_"

    for file_path in target_dir.glob("*.json"):
        if file_path.name.startswith(prefix):
            # TREFFER! Laden und zurückgeben.
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                t = data.get('build_time_seconds', data.get('indexing_duration'))
                c = data.get('num_chunks')
                return t, c
            except Exception:
                return None, None

    return None, None


def main():
    print("🚀 Starte Matching mit Prefix-Logik...")

    # CSVs suchen
    csv_files = list(RESULTS_DIR.rglob("*summary.csv"))
    if not csv_files:
        csv_files = list(RESULTS_DIR.rglob("*detailed_results.csv"))

    all_rows = []
    found_count = 0
    missing_count = 0

    for csv_path in tqdm(csv_files, desc="Verarbeite Dateien"):
        try:
            # Metadata laden für Modell-Namen
            meta_path = csv_path.parent / "metadata.json"
            model = "unknown"
            diff = "unknown"
            dtype = "Gold"

            if meta_path.exists():
                with open(meta_path, 'r') as f:
                    m = json.load(f)
                    model = m.get("embedding_model", "unknown")
                    diff = m.get("difficulty", "unknown")
                    if "silver" in m.get("input_file", "").lower(): dtype = "Silver"
            else:
                # Fallback aus Pfad
                path_str = str(csv_path)
                if "MiniLM" in path_str:
                    model = "all-MiniLM-L6-v2"
                elif "bge-base" in path_str:
                    model = "BAAI/bge-base-en-v1.5"
                elif "bge-large" in path_str:
                    model = "BAAI/bge-large-en-v1.5"

            # Difficulty Fix
            if dtype == "Silver": diff = "Hard"
            if diff == "unknown":
                if "easy" in str(csv_path).lower():
                    diff = "Easy"
                elif "hard" in str(csv_path).lower():
                    diff = "Hard"
                elif "moderate" in str(csv_path).lower():
                    diff = "Moderate"

            # CSV laden
            df = pd.read_csv(csv_path)

            df['model_name'] = model
            df['dataset_difficulty'] = diff
            df['dataset_type'] = dtype
            df['source_file'] = csv_path.name

            # Index Matching Zeile für Zeile
            if 'experiment' in df.columns:
                for idx, row in df.iterrows():
                    exp = row['experiment']

                    # Hier rufen wir die neue Funktion auf
                    time_val, chunks_val = get_index_data(model, exp)

                    df.at[idx, 'indexing_time_s'] = time_val
                    df.at[idx, 'num_chunks'] = chunks_val

                    if time_val is not None:
                        found_count += 1
                    else:
                        missing_count += 1

            all_rows.append(df)

        except Exception as e:
            print(f"Fehler bei {csv_path}: {e}")

    # Speichern
    if all_rows:
        final_df = pd.concat(all_rows, ignore_index=True)
        final_df = final_df[final_df['model_name'] != 'unknown']

        # Spalten sortieren
        cols = ['model_name', 'dataset_type', 'dataset_difficulty', 'experiment', 'ndcg_at_k', 'map', 'indexing_time_s',
                'num_chunks']
        rest = [c for c in final_df.columns if c not in cols]
        final_df = final_df[cols + rest]

        final_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\n✅ FERTIG: {OUTPUT_FILE}")
        print(f"📊 Statistik: {found_count} gefunden, {missing_count} fehlen.")

        # Check BGE
        bge_data = final_df[final_df['model_name'].str.contains("bge", na=False)]
        bge_found = bge_data['indexing_time_s'].notna().sum()
        print(f"🔎 BGE-Check: {bge_found} Index-Zeiten für BGE-Modelle gefunden.")

    else:
        print("Keine Daten.")


if __name__ == "__main__":
    main()