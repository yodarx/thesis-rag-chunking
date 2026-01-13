 #!/bin/bash

# Script to run all experiments with all difficulties against Gold and Silver datasets

LOGFILE="results/run_all_experiments_$(date +%Y%m%d_%H%M%S).log"
DATA_GOLD="data/preprocessed/gold.jsonl"
DATA_SILVER="data/preprocessed/silver.jsonl"

# Ensure results directory exists
mkdir -p results

echo ">>> Logging started: $LOGFILE"
echo "========================================================" | tee -a "$LOGFILE"
echo "START ALL EXPERIMENTS: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOGFILE"
echo "========================================================" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"

# Find all config files, excluding _archive and hidden files
CONFIG_FILES=$(find configs -name "*.json" -not -path "*/_archive/*" -not -path "*/.*" | sort)

for config in $CONFIG_FILES; do
    echo "--------------------------------------------------------" | tee -a "$LOGFILE"
    echo "PROCESSING CONFIG: $config" | tee -a "$LOGFILE"
    echo "--------------------------------------------------------" | tee -a "$LOGFILE"

    # Datasets loop
    for dataset_path in "$DATA_GOLD" "$DATA_SILVER"; do
        dataset_name=$(basename "$dataset_path" .jsonl)

        # Difficulties loop
        for difficulty in "Hard" "Moderate" "Easy"; do
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] >>> RUNNING: Config=$config | Dataset=$dataset_name | Difficulty=$difficulty" | tee -a "$LOGFILE"

            python run.py \
                --config-json "$config" \
                --difficulty "$difficulty" \
                --input-file "$dataset_path" \
                2>&1 | tee -a "$LOGFILE"

            echo "" | tee -a "$LOGFILE"
        done
    done
done

echo "========================================================" | tee -a "$LOGFILE"
echo "ALL EXPERIMENTS FINISHED: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOGFILE"
echo "Log file saved to: $LOGFILE" | tee -a "$LOGFILE"
echo "========================================================" | tee -a "$LOGFILE"

