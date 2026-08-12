#!/usr/bin/env bash
set -euo pipefail

GPU_TYPE="${1:-RTX_4090}"
MODEL_PATH="${2:-/workspace/output/lora_qwen36_agentic/merged}"
RUNPOD_API_KEY="${RUNPOD_API_KEY:-}"

if [ -z "$RUNPOD_API_KEY" ]; then
    echo "ERROR: Set RUNPOD_API_KEY env var"
    exit 1
fi

echo "=== Deploying vLLM Inference Server on RunPod ==="
echo "GPU: $GPU_TYPE"
echo "Model: $MODEL_PATH"
echo "Base: Qwen3.6-35B-A3B (35B MoE, 3B active, 256K context)"
echo ""

# Create inference pod
POD_ID=$(curl -s -X POST "https://api.runpod.io/v2/pods" \
    -H "Authorization: Bearer $RUNPOD_API_KEY" \
    -H "Content-Type: application/json" \
    -d "{
        \"cloud_type\": \"ALL\",
        \"gpu_type_id\": \"$GPU_TYPE\",
        \"gpu_count\": 1,
        \"container_disk_in_gb\": 50,
        \"volume_in_gb\": 200,
        \"volume_mount_path\": \"/workspace\",
        \"image_name\": \"vllm/vllm-openai:latest\",
        \"name\": \"qwen36-inference\",
        \"ports\": \"8000/http\",
        \"env\": [
            {\"key\": \"MODEL_PATH\", \"value\": \"$MODEL_PATH\"}
        ],
        \"args\": [
            \"--model\", \"$MODEL_PATH\",
            \"--host\", \"0.0.0.0\",
            \"--port\", \"8000\",
            \"--max-model-len\", \"262144\",
            \"--gpu-memory-utilization\", \"0.90\",
            \"--served-model-name\", \"qwen3.6-coding\",
            \"--enable-auto-tool-choice\",
            \"--tool-call-parser\", \"qwen3_xml\",
            \"--reasoning-parser\", \"qwen3\",
            \"--enable-prefix-caching\",
            \"--enable-chunked-prefill\",
            \"--trust-remote-code\"
        ]
    }" | python3 -c "import sys,json; print(json.load(sys.stdin).get('id',''))")

echo "Inference Pod ID: $POD_ID"
echo ""
echo "=== opencode.json config ==="
echo 'Add to your opencode.json provider section:'
echo '{
  "runpod-local": {
    "url": "http://<POD_IP>:8000/v1",
    "models": {
      "qwen3.6-coding": { "name": "qwen3.6-coding" }
    }
  }
}'
echo ""
echo "=== Zed settings.json for autocomplete ==="
echo 'Add edit prediction provider pointing to FIM server:'
echo '{
  "edit_predictions": {
    "disabled": false,
    "copilot": { "disabled": true },
    "zed": { "disabled": true },
    "custom": {
      "url": "http://<FIM_POD_IP>:8080/v1/completions"
    }
  }
}'
