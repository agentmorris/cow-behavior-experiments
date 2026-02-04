#!/bin/bash
# Run all cattle behavior classification models sequentially.
# Failures are logged but do not stop subsequent models.
# Models that already have result files are skipped.
#
# Usage:
#   ./run_all_models.sh [--output-dir DIR]
#
# If --output-dir is not specified, results go to the default results folder.
# For datestamped batch runs, use e.g.:
#   ./run_all_models.sh --output-dir C:/temp/cow-experiments/cow-vlm-experiments/results/20260127

cd "c:/git/agentmorrisprivate/archive/cow_experiments"
export PYTHONUNBUFFERED=1

DEFAULT_RESULTS_DIR="C:/temp/cow-experiments/cow-vlm-experiments/results"
RESULTS_DIR="$DEFAULT_RESULTS_DIR"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--output-dir DIR]"
            exit 1
            ;;
    esac
done

mkdir -p "$RESULTS_DIR"

LOG="$RESULTS_DIR/run_all_models.log"
LOCKFILE="C:/temp/cow-experiments/cow-vlm-experiments/run_all_models.lock"

# Prevent concurrent execution
if [ -f "$LOCKFILE" ]; then
    existing_pid=$(cat "$LOCKFILE" 2>/dev/null)
    if kill -0 "$existing_pid" 2>/dev/null; then
        echo "ERROR: Another instance is already running (PID $existing_pid)."
        echo "If this is stale, remove $LOCKFILE and retry."
        exit 1
    else
        echo "Removing stale lock file (PID $existing_pid no longer running)."
        rm -f "$LOCKFILE"
    fi
fi
echo $$ > "$LOCKFILE"
trap 'rm -f "$LOCKFILE"' EXIT

echo "=== Starting full model comparison run ===" | tee "$LOG"
echo "Started at: $(date)" | tee -a "$LOG"
echo "Results directory: $RESULTS_DIR" | tee -a "$LOG"
echo "" | tee -a "$LOG"

FAILED_MODELS=""
SUCCEEDED_MODELS=""
SKIPPED_MODELS=""

# Check if a result file already exists for a model.
# Sanitizes model name (replace : and / with -) and looks for matching JSON.
has_results() {
    local model="$1"
    local safe_name
    safe_name=$(echo "$model" | sed 's|models/||; s/:/-/g; s|/|-|g')
    local count
    count=$(ls "$RESULTS_DIR"/${safe_name}*.json 2>/dev/null | wc -l)
    [ "$count" -gt 0 ]
}

run_gemini() {
    local model="$1"
    if has_results "$model"; then
        echo "----------------------------------------------" | tee -a "$LOG"
        echo "SKIPPED: $model (result file already exists)" | tee -a "$LOG"
        SKIPPED_MODELS="$SKIPPED_MODELS $model"
        echo "" | tee -a "$LOG"
        return
    fi
    echo "----------------------------------------------" | tee -a "$LOG"
    echo "Starting Gemini model: $model" | tee -a "$LOG"
    echo "Time: $(date)" | tee -a "$LOG"
    echo "" | tee -a "$LOG"
    python run_gemini_classification.py --model "$model" --sync -y --output-dir "$RESULTS_DIR" 2>&1 | tee -a "$LOG"
    local exit_code=${PIPESTATUS[0]}
    if [ $exit_code -eq 0 ]; then
        echo "SUCCESS: $model (exit code $exit_code)" | tee -a "$LOG"
        SUCCEEDED_MODELS="$SUCCEEDED_MODELS $model"
    else
        echo "FAILED: $model (exit code $exit_code)" | tee -a "$LOG"
        FAILED_MODELS="$FAILED_MODELS $model"
    fi
    echo "" | tee -a "$LOG"
}

run_ollama() {
    local model="$1"
    local extra_args="$2"
    if has_results "$model"; then
        echo "----------------------------------------------" | tee -a "$LOG"
        echo "SKIPPED: $model (result file already exists)" | tee -a "$LOG"
        SKIPPED_MODELS="$SKIPPED_MODELS $model"
        echo "" | tee -a "$LOG"
        return
    fi
    echo "----------------------------------------------" | tee -a "$LOG"
    echo "Starting Ollama model: $model" | tee -a "$LOG"
    echo "Time: $(date)" | tee -a "$LOG"
    echo "" | tee -a "$LOG"
    python run_ollama_classification.py --model "$model" --output-dir "$RESULTS_DIR" $extra_args 2>&1 | tee -a "$LOG"
    local exit_code=${PIPESTATUS[0]}
    if [ $exit_code -eq 0 ]; then
        echo "SUCCESS: $model (exit code $exit_code)" | tee -a "$LOG"
        SUCCEEDED_MODELS="$SUCCEEDED_MODELS $model"
    else
        echo "FAILED: $model (exit code $exit_code)" | tee -a "$LOG"
        FAILED_MODELS="$FAILED_MODELS $model"
    fi
    echo "" | tee -a "$LOG"
}

# Gemini models first (synchronous API)
run_gemini "gemini-3-flash-preview"
run_gemini "gemini-3-pro-preview"

# Ollama models in download order
run_ollama "qwen3-vl:8b"
run_ollama "qwen3-vl:32b"
run_ollama "qwen2.5vl:7b"
run_ollama "qwen2.5vl:32b"
# llama3.2-vision only supports one image per message, so query-batch-size must be 1
run_ollama "llama3.2-vision:11b" "--query-batch-size 1"
run_ollama "ministral-3:14b"
run_ollama "ministral-3:8b"
run_ollama "mistral-small3.2:24b"
run_ollama "gemma3:27b"
run_ollama "gemma3:12b"

# Summary
echo "==============================================" | tee -a "$LOG"
echo "=== ALL JOBS COMPLETE ===" | tee -a "$LOG"
echo "Finished at: $(date)" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "SUCCEEDED:$SUCCEEDED_MODELS" | tee -a "$LOG"
echo "" | tee -a "$LOG"
if [ -n "$FAILED_MODELS" ]; then
    echo "FAILED:$FAILED_MODELS" | tee -a "$LOG"
else
    echo "No failures." | tee -a "$LOG"
fi
if [ -n "$SKIPPED_MODELS" ]; then
    echo "SKIPPED (already had results):$SKIPPED_MODELS" | tee -a "$LOG"
fi
echo "" | tee -a "$LOG"
echo "Results saved to: $RESULTS_DIR" | tee -a "$LOG"
echo "" | tee -a "$LOG"
