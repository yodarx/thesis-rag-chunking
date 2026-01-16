 #!/bin/bash

LOGFILE="results/run_all_experiments_$(date +%Y%m%d_%H%M%S).log"

echo ">>> Logging started: $LOGFILE"
echo "========================================================" | tee -a "$LOGFILE"
echo "START ALL EXPERIMENTS: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOGFILE"
echo "========================================================" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"

# Find all config files, excluding _archive and hidden files
CONFIG_FILES=$(find configs -name "*.json" -not -name "00_sanity_check.json" -not -path "*/_archive/*" -not -path "*/.*" | sort)

for config in $CONFIG_FILES; do
    echo "--------------------------------------------------------" | tee -a "$LOGFILE"
    echo "PROCESSING CONFIG: $config" | tee -a "$LOGFILE"
    echo "--------------------------------------------------------" | tee -a "$LOGFILE"

    # Extract filename to check for silver status
    config_filename=$(basename "$config")

    # Check if config filename is silver (case insensitive check)
    if [[ "$config_filename" == *"SILVER"* ]] || [[ "$config_filename" == *"silver"* ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] >>> RUNNING SILVER: Config=$config" | tee -a "$LOGFILE"

        python run.py \
            --config-json "$config" \
            2>&1 | tee -a "$LOGFILE"

        echo "" | tee -a "$LOGFILE"
    else
        for difficulty in "Hard" "Moderate" "Easy"; do
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] >>> RUNNING: Config=$config | Difficulty=$difficulty" | tee -a "$LOGFILE"

            python run.py \
                --config-json "$config" \
                --difficulty "$difficulty" \
                2>&1 | tee -a "$LOGFILE"

            echo "" | tee -a "$LOGFILE"
        done
    fi
done

echo "========================================================" | tee -a "$LOGFILE"
echo "ALL EXPERIMENTS FINISHED: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOGFILE"
echo "Log file saved to: $LOGFILE" | tee -a "$LOGFILE"
echo "========================================================" | tee -a "$LOGFILE"

