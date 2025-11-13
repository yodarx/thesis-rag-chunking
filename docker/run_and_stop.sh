#!/bin/bash
set -e

echo "Starting experiments..."
python run.py --run-all-configs
EXIT_CODE=$?
echo "Experiments completed with exit code: $EXIT_CODE"

# Stop the pod after completion if running on RunPod
if [ ! -z "$RUNPOD_POD_ID" ]; then
    echo "Stopping RunPod pod: $RUNPOD_POD_ID"

    # Configure runpodctl with API key if available
    if [ ! -z "$RUNPOD_API_KEY" ]; then
        echo "Configuring runpodctl with API key..."
        runpodctl config --apiKey $RUNPOD_API_KEY
    else
        echo "Warning: RUNPOD_API_KEY not set, attempting to use existing config..."
    fi

    # Get and display pod details for debugging
    echo "Getting pod details..."
    runpodctl get pod $RUNPOD_POD_ID || echo "Failed to get pod details"

    # Give a moment for logs to flush
    sleep 2
    # Use remove instead of stop to fully terminate the pod
    echo "Attempting to remove pod..."
    runpodctl remove pod $RUNPOD_POD_ID || echo "Failed to remove pod, but experiments completed"
    # Keep the process alive for a bit to ensure the command executes
    sleep 5
else
    echo "Not running on RunPod, skipping pod stop"
fi

exit $EXIT_CODE
