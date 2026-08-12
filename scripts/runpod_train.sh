#!/usr/bin/env bash
set -euo pipefail

GPU_TYPE="${1:-RTX_4090}"
POD_ID="${2:-}"
RUNPOD_API_KEY="${RUNPOD_API_KEY:-}"

if [ -z "$RUNPOD_API_KEY" ]; then
    echo "ERROR: Set RUNPOD_API_KEY env var"
    echo "  export RUNPOD_API_KEY=your_key_here"
    exit 1
fi

echo "=== Training Pipeline: Qwen3.6-35B-A3B LoRA ==="
echo "GPU: $GPU_TYPE"
echo ""

if [ -z "$POD_ID" ]; then
    echo "Creating RunPod pod..."
    POD_ID=$(curl -s -X POST "https://api.runpod.io/v2/pods" \
        -H "Authorization: Bearer $RUNPOD_API_KEY" \
        -H "Content-Type: application/json" \
        -d "{
            \"cloud_type\": \"ALL\",
            \"gpu_type_id\": \"$GPU_TYPE\",
            \"gpu_count\": 1,
            \"container_disk_in_gb\": 100,
            \"volume_in_gb\": 200,
            \"volume_mount_path\": \"/workspace\",
            \"image_name\": \"runpod/pytorch:2.4.0-py3.11-cuda12.4-cudnn-devel\",
            \"name\": \"training-pipe-qwen36\",
            \"ports\": \"8000/http\",
            \"env\": [
                {\"key\": \"HF_HOME\", \"value\": \"/workspace/huggingface\"},
                {\"key\": \"WANDB_MODE\", \"value\": \"online\"}
            ]
        }" | python3 -c "import sys,json; print(json.load(sys.stdin).get('id',''))")
    
    echo "Pod ID: $POD_ID"
    echo "Waiting for pod to start..."
    sleep 60
fi

echo ""
echo "=== Setup commands (run inside pod): ==="
echo ""
echo "# Install deps"
echo "pip install uv"
echo "cd /workspace"
echo ""
echo "# Download base model (35B MoE, ~70GB in 4-bit)"
echo "huggingface-cli download Qwen/Qwen3.6-35B-A3B --local-dir /workspace/models/qwen3.6-35b-a3b"
echo ""
echo "# Upload training data from local machine:"
echo "rsync -avz data/formatted/ root@<POD_IP>:/workspace/training-pipe/data/formatted/"
echo ""
echo "# Train agentic LoRA (A100 80GB recommended)"
echo "python src/train/train_lora.py --model /workspace/models/qwen3.6-35b-a3b --dataset data/formatted/agentic_sft.jsonl --no-wandb"
echo ""
echo "# For RTX 4090 (24GB): reduce LoRA rank and seq length"
echo "python src/train/train_lora.py --model /workspace/models/qwen3.6-35b-a3b --dataset data/formatted/agentic_sft.jsonl --lora_r 32 --max-seq-len 8192 --no-wandb"
echo ""
echo "# Merge LoRA"
echo "python src/train/merge_lora.py --base /workspace/models/qwen3.6-35b-a3b --adapter output/lora_qwen36_agentic/final --output output/merged"
echo ""
echo "# Deploy inference"
echo "vllm serve output/merged --served-model-name qwen3.6-coding --tool-call-parser qwen3_xml --reasoning-parser qwen3 --max-model-len 262144 --gpu-memory-utilization 0.90 --enable-prefix-caching --enable-chunked-prefill --trust-remote-code"
echo ""
echo "=== Expected training time ==="
echo "  A100 80GB: ~1-2 hours (2002 agentic examples, 3 epochs, LoRA r=128)"
echo "  RTX 4090:  ~4-6 hours (with LoRA r=32, seq_len=8192)"
echo "  H100 80GB: ~30-60 min"
echo ""
echo "=== GPU requirements ==="
echo "  Qwen3.6-35B-A3B is MoE: 35B total but only 3B active per token"
echo "  4-bit QLoRA VRAM: ~24GB (fits RTX 4090 with reduced seq_len)"
echo "  bf16 LoRA VRAM:    ~80GB (needs A100/H100 80GB)"
echo "  Inference VRAM:    ~20GB in 4-bit, ~70GB in bf16"
