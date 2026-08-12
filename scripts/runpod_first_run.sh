#!/usr/bin/env bash
set -euo pipefail

# RunPod first run: Qwen3.6-35B-A3B LoRA on RTX Pro 6000 Blackwell 96GB
# Uses Unsloth for 12x faster MoE training
# bf16 LoRA (NOT QLoRA — MoE QLoRA not recommended per Unsloth)

RUNPOD_API_KEY="${RUNPOD_API_KEY:-}"
if [ -z "$RUNPOD_API_KEY" ]; then
    echo "ERROR: Set RUNPOD_API_KEY env var"
    echo "  export RUNPOD_API_KEY=your_key_here"
    exit 1
fi

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  TRAINING PIPE — FIRST RUN ON RTX PRO 6000 BLACKWELL 96GB  ║"
echo "║  Model: Qwen3.6-35B-A3B (35B MoE / 3B active)              ║"
echo "║  Method: Unsloth bf16 LoRA (r=128, α=256)                 ║"
echo "║  Data: 16K agentic SFT examples, ~57.6M tokens            ║"
echo "║  GPU: RTX Pro 6000 Blackwell, 96GB GDDR7, 1.69/hr        ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Step 1: Create pod
echo "[1/5] Creating RunPod pod (RTX PRO 6000, Community Cloud)..."
POD_JSON=$(curl -s -X POST "https://api.runpod.io/v2/pods" \
    -H "Authorization: Bearer $RUNPOD_API_KEY" \
    -H "Content-Type: application/json" \
    -d '{
        "cloud_type": "ALL",
        "gpu_type_id": "RTX PRO 6000",
        "gpu_count": 1,
        "container_disk_in_gb": 200,
        "volume_in_gb": 500,
        "volume_mount_path": "/workspace",
        "image_name": "runpod/pytorch:2.5.1-py3.11-cuda12.4-cudnn-devel",
        "name": "training-pipe-qwen36",
        "ports": "22/tcp,8000/http,6006/http",
        "env": [
            {"key": "HF_HOME", "value": "/workspace/huggingface"},
            {"key": "WANDB_MODE", "value": "online"},
            {"key": "UNSLOTH_MOE_BACKEND", "value": "auto"}
        ]
    }')

POD_ID=$(echo "$POD_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin).get('id',''))" 2>/dev/null)
echo "  Pod ID: $POD_ID"
echo "  Monitor: https://console.runpod.io/pods/$POD_ID"
echo ""

# Step 2: Wait for pod to be ready
echo "[2/5] Waiting for pod to initialize..."
for i in $(seq 1 30); do
    STATUS=$(curl -s "https://api.runpod.io/v2/pods/$POD_ID" \
        -H "Authorization: Bearer $RUNPOD_API_KEY" \
        | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('desiredStatus',''))" 2>/dev/null)
    echo "  Status: $STATUS ($i/30)"
    if [ "$STATUS" = "RUNNING" ]; then
        break
    fi
    sleep 10
done

# Get SSH and HTTP ports
POD_INFO=$(curl -s "https://api.runpod.io/v2/pods/$POD_ID" \
    -H "Authorization: Bearer $RUNPOD_API_KEY")

SSH_PORT=$(echo "$POD_INFO" | python3 -c "import sys,json; d=json.load(sys.stdin); ports=d.get('runtime',{}).get('ports',{}); print(ports.get('22/tcp','').split(':')[1] if '22/tcp' in ports else '')" 2>/dev/null)
HTTP_PORT=$(echo "$POD_INFO" | python3 -c "import sys,json; d=json.load(sys.stdin); ports=d.get('runtime',{}).get('ports',{}); print(ports.get('8000/http','').split(':')[1] if '8000/http' in ports else '')" 2>/dev/null)
POD_IP=$(echo "$POD_INFO" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('runtime',{}).get('ports',{}).get('8000/http','').split(':')[0] if d.get('runtime') else '')" 2>/dev/null)

echo "  IP: $POD_IP"
echo "  SSH port: $SSH_PORT"
echo "  HTTP port: $HTTP_PORT"
echo ""

# Step 3: Print setup commands to run inside pod
echo "[3/5] Run these commands inside the pod (SSH or RunPod terminal):"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "# SSH: ssh root@$POD_IP -p $SSH_PORT"
echo ""
echo "# Install dependencies"
echo "pip install unsloth"
echo "pip install --upgrade 'transformers>=5.2.0,<=5.5.0'"
echo "pip install datasets trl peft bitsandbytes accelerate wandb"
echo ""
echo "# Download base model (Unsloth's optimized version)"
echo "huggingface-cli download unsloth/Qwen3.6-35B-A3B --local-dir /workspace/models/qwen3.6-35b-a3b"
echo ""
echo "# Upload training data from your machine (run locally):"
echo "rsync -avzP data/formatted/ root@$POD_IP:/workspace/data/formatted/ -e 'ssh -p $SSH_PORT'"
echo ""
echo "# Start training (smoke test first)"
echo "cd /workspace/training-pipe"
echo "python src/train/train_unsloth.py \\"
echo "  --model /workspace/models/qwen3.6-35b-a3b \\"
echo "  --dataset data/formatted/agentic_sft.jsonl \\"
echo "  --max-samples 100 \\"
echo "  --max-seq-len 4096 \\"
echo "  --no-wandb"
echo ""
echo "# If smoke test works, full training:"
echo "python src/train/train_unsloth.py \\"
echo "  --model /workspace/models/qwen3.6-35b-a3b \\"
echo "  --dataset data/formatted/agentic_sft.jsonl \\"
echo "  --config configs/training_agentic.json"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Step 4: Print cost estimate
echo "[4/5] Cost estimate:"
echo "  Model download: ~10 min"
echo "  Smoke test: ~5 min"
echo "  Full training (3 epochs, 16K examples):"
echo "    RTX Pro 6000 @ 1.69/hr:"
echo "      Unsloth MoE speedup: ~2-3 hours estimated"
echo "      Cost: ~$3.40 - $5.07"
echo "    A100 80GB @ $2.10/hr (for comparison):"
echo "      ~3-4 hours estimated"
echo "      Cost: ~$6.30 - $8.40"
echo ""

# Step 5: Inference deploy
echo "[5/5] After training, deploy inference:"
echo "  vllm serve output/lora_qwen36_agentic/merged \\"
echo "    --served-model-name qwen3.6-coding \\"
echo "    --tool-call-parser qwen3_xml \\"
echo "    --reasoning-parser qwen3 \\"
echo "    --max-model-len 262144 \\"
echo "    --gpu-memory-utilization 0.90 \\"
echo "    --enable-prefix-caching \\"
echo "    --enable-chunked-prefill \\"
echo "    --trust-remote-code"
echo ""
echo "  Test: curl http://$POD_IP:8000/v1/models"
echo ""
echo "Pod ID for cleanup: $POD_ID"
echo "Stop pod: curl -X POST 'https://api.runpod.io/v2/pods/$POD_ID/stop' -H 'Authorization: Bearer $RUNPOD_API_KEY'"
echo "Terminate: curl -X POST 'https://api.runpod.io/v2/pods/$POD_ID/terminate' -H 'Authorization: Bearer $RUNPOD_API_KEY'"
