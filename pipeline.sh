#!/usr/bin/env bash

set -e  # Exit immediately if a command exits with a non-zero status

# File to store the last successful step
STATUS_FILE="pipeline_status.txt"

# Array of scripts to run in order
STEPS=(
    "Parsing/Parse_ESTA_LAN.py"
    "Compute_Features/Journey_Dwell_Compute.py"
    "Compute_Features/Journey_Dwell_Metrics_Compute.py"
    "Compute_Features/Refgem_Compute.py"
    "Compute_Features/Collate_Features.py"
)

# Parse optional resume argument
RESUME_FROM=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --resume)
            RESUME_FROM="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Determine starting point
START_INDEX=0
if [[ -n "$RESUME_FROM" ]]; then
    FOUND=0
    for i in "${!STEPS[@]}"; do
        if [[ "${STEPS[$i]}" == "$RESUME_FROM" ]]; then
            START_INDEX=$((i + 1))
            FOUND=1
            break
        fi
    done
    if [[ $FOUND -eq 0 ]]; then
        echo "Error: Step '$RESUME_FROM' not found in pipeline."
        exit 1
    fi
elif [[ -f "$STATUS_FILE" ]]; then
    LAST_COMPLETED=$(cat "$STATUS_FILE")
    FOUND=0
    for i in "${!STEPS[@]}"; do
        if [[ "${STEPS[$i]}" == "$LAST_COMPLETED" ]]; then
            START_INDEX=$((i + 1))
            FOUND=1
            echo "Resuming from after last completed step: $LAST_COMPLETED"
            break
        fi
    done
    if [[ $FOUND -eq 0 ]]; then
        echo "Warning: Last completed step '$LAST_COMPLETED' not found. Starting from scratch."
    fi
fi

# Execute pipeline
for (( i=$START_INDEX; i<${#STEPS[@]}; i++ )); do
    STEP="${STEPS[$i]}"
    SCRIPT_DIR=$(dirname "$STEP")
    SCRIPT_FILE=$(basename "$STEP")

    echo "Running: python3 $SCRIPT_FILE in $SCRIPT_DIR"
    if (cd "$SCRIPT_DIR" && python3 "$SCRIPT_FILE"); then
        echo "$STEP" > "$STATUS_FILE"
        echo "Completed: $STEP"
    else
        echo "Error occurred in step: $STEP"
        if [[ -f "$STATUS_FILE" ]]; then
            echo "Last successful step: $(cat $STATUS_FILE)"
        else
            echo "No successful steps completed yet."
        fi
        exit 1
    fi
done


echo "Pipeline completed successfully!"
# Cleanup if needed
rm -f "$STATUS_FILE"
