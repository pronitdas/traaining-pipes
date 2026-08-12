#!/usr/bin/env bash
# Train a MiniCPM5-1B agentic LoRA locally on the RTX 3070 Ti. No pod, no rsync.
#
#   ./scripts/local_train_minicpm5.sh                     # native <tool_call> format (primary)
#   ./scripts/local_train_minicpm5.sh markers             # in-band [TOOL_CALL] regression run
#   ./scripts/local_train_minicpm5.sh native smoke        # 200-sample smoke test, foreground
#   ./scripts/local_train_minicpm5.sh <path/to/config.json>
#
# The full run goes to background under nohup and prints its logfile; tail that to watch.
# Detaching never touches training -- same convention as the RunPod runs behind watch.sh.
#
# Why .venv-train and not .venv: Unsloth pins transformers 5.5 and trl 0.24, while .venv
# holds transformers 5.15 + vllm 0.26 for serving. Installing Unsloth into .venv would drag
# the serving side backwards, so training gets its own interpreter.

set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv-train/bin/python
CONFIGS_DIR=configs
LOG_DIR=output/logs

TARGET="${1:-native}"
MODE="${2:-full}"

case "$TARGET" in
    native)  CONFIG="$CONFIGS_DIR/training_agentic_minicpm5_1b_q4.json" ;;
    markers) CONFIG="$CONFIGS_DIR/training_agentic_minicpm5_1b_q4_markers.json" ;;
    *)       CONFIG="$TARGET" ;;
esac

[ -f "$CONFIG" ] || { echo "ERROR: no such config: $CONFIG"; exit 1; }
[ -x "$PY" ] || { echo "ERROR: $PY missing -- create it with: uv venv .venv-train --python 3.12 && VIRTUAL_ENV=.venv-train uv pip install unsloth unsloth_zoo"; exit 1; }

DATASET=$("$PY" -c "import json,sys; print(json.load(open('$CONFIG'))['dataset_path'])")
OUTPUT=$("$PY" -c "import json,sys; print(json.load(open('$CONFIG'))['output_dir'])")
[ -f "$DATASET" ] || { echo "ERROR: dataset missing: $DATASET"; echo "  build it with: .venv/bin/python src/format/format_agentic_minicpm5.py"; exit 1; }

# 8GB card with a desktop already on it leaves ~5GB. expandable_segments keeps the
# allocator from fragmenting that margin away over a long run.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
echo "════════════════════════════════════════════════════════════"
echo "  MiniCPM5-1B agentic QLoRA — local"
echo "  config:  $CONFIG"
echo "  dataset: $DATASET"
echo "  output:  $OUTPUT"
echo "  GPU free: ${FREE_MB}MB"
echo "════════════════════════════════════════════════════════════"
if [ "$FREE_MB" -lt 4200 ]; then
    echo "WARNING: under 4.2GB free. Close GPU consumers (browser, other sessions) or this will OOM."
fi

if [ "$MODE" = "smoke" ]; then
    echo "[smoke] 200 samples, foreground. Watch peak VRAM in another shell with: nvidia-smi -l 2"
    exec "$PY" src/train/train_unsloth.py --config "$CONFIG" --max-samples 200 --no-wandb
fi

mkdir -p "$LOG_DIR"
STAMP=$(date +%Y%m%d-%H%M%S)
LOG="$LOG_DIR/minicpm5-$(basename "$CONFIG" .json)-$STAMP.log"

echo "[full] launching under nohup -> $LOG"
nohup "$PY" src/train/train_unsloth.py --config "$CONFIG" --no-wandb > "$LOG" 2>&1 &
echo "  pid $!"
echo ""
echo "  watch:  tail -f $LOG"
echo "  vram:   nvidia-smi -l 2"
