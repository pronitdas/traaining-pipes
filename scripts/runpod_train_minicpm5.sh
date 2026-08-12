#!/usr/bin/env bash
# Drive a MiniCPM5-1B agentic QLoRA run on a RunPod H100 end to end.
#
#   ./scripts/runpod_train_minicpm5.sh            # create pod, upload, install, train
#   ./scripts/runpod_train_minicpm5.sh <POD_ID>   # reuse a pod that is already running
#
# Unlike runpod_first_run.sh, which prints the commands for you to paste, this one executes
# them over SSH -- the pod bills from creation, so the setup should not wait on a human.
#
# Teardown is deliberately NOT automatic. The run ends with the pod alive so you can pull
# artifacts and eval on it; terminate it yourself when done (the script prints the command).

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG=configs/training_agentic_minicpm5_1b_h100_q4.json
DATASET=data/formatted/agentic_sft_minicpm5.jsonl
EVALSET=data/formatted/agentic_eval_minicpm5.jsonl
GPU_TYPE="${GPU_TYPE:-NVIDIA H100 80GB HBM3}"
# The frozen base from .github/workflows/build-base-image.yml. Deps are already installed, so
# there is no pip step on the pod -- that is the whole point of the image.
IMAGE="${IMAGE:-ghcr.io/pronitdas/training-pipe:base}"
# Only needed while the GHCR package is private: a RunPod "container registry credential" ID
# (create one in the RunPod console with your GitHub username + a PAT holding read:packages).
# Leave empty if the package is public.
REGISTRY_ID="${REGISTRY_ID:-}"
SSH_KEY=/home/pronit/.runpod/ssh/RunPod-Key-Go
REMOTE=/workspace/training-pipe

KEY=$(python3 -c "
import re,sys
for l in open('/home/pronit/.runpod/config.toml'):
    m=re.match(r'\s*apikey\s*=\s*[\"\x27]?([^\"\x27\s]+)',l)
    if m: print(m.group(1)); sys.exit()
")
[ -n "$KEY" ] || { echo "ERROR: no apikey in ~/.runpod/config.toml"; exit 1; }

# Without PUBLIC_KEY in the pod env these images never provision authorized_keys and never
# start sshd, so the pod boots RUNNING but refuses every connection. Costly to rediscover.
PUBKEY=$(cat "$SSH_KEY.pub")
[ -n "$PUBKEY" ] || { echo "ERROR: no public key at $SSH_KEY.pub"; exit 1; }
[ -f "$DATASET" ] || { echo "ERROR: missing $DATASET -- run src/format/format_agentic_minicpm5.py"; exit 1; }

api() { curl -s -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" "$@"; }
jqp() { python3 -c "import sys,json;d=json.load(sys.stdin);$1" 2>/dev/null; }

POD_ID="${1:-}"

if [ -z "$POD_ID" ]; then
    echo "[1/6] Creating H100 pod ($GPU_TYPE)..."
    # Schema per https://api.runpod.io/v2/openapi.json (POST /v2/pods): requires name+image,
    # nests the GPU under gpu:{id,count} and the volume under mounts.persistent:{size,path}.
    # runpod_first_run.sh's flat snake_case body (gpu_type_id, volume_in_gb, image_name, a
    # string `ports`, a list `env`) is rejected 422 by the current API.
    # runpod/pytorch retired the "2.5.1-py3.11-cuda12.4-cudnn-devel" tag scheme; the current
    # family is 1.1.0-rc.<n>-cu<ver>-torch<ver>-ubuntu<ver> and there is no stable tag yet.
    for CLOUD in SECURE COMMUNITY; do
        RESP=$(api -X POST "https://api.runpod.io/v2/pods" -d "{
            \"name\": \"minicpm5-agentic\",
            \"image\": \"$IMAGE\",
            \"cloud\": \"$CLOUD\",
            \"gpu\": {\"id\": \"$GPU_TYPE\", \"count\": 1},
            \"disk\": 100,
            \"ports\": [\"22/tcp\", \"8000/http\"],
            \"env\": {\"HF_HOME\": \"/workspace/huggingface\", \"PUBLIC_KEY\": \"$PUBKEY\"},
            ${REGISTRY_ID:+\"registry\": \"$REGISTRY_ID\",}
            \"mounts\": {\"persistent\": {\"size\": 100, \"path\": \"/workspace\"}}
        }")
        POD_ID=$(echo "$RESP" | jqp "print(d.get('id',''))")
        [ -n "$POD_ID" ] && { echo "  cloud: $CLOUD"; break; }
        echo "  $CLOUD rejected: $(echo "$RESP" | head -c 240)"
    done
    [ -n "$POD_ID" ] || { echo "ERROR: pod creation failed"; exit 1; }
    echo "  pod: $POD_ID   console: https://console.runpod.io/pods/$POD_ID"
else
    echo "[1/6] Reusing pod $POD_ID"
fi

# 60x10s was not enough for a cold pull of a ~10GB image; the pod reports RUNNING long before
# the container is up, and ports appear later still.
echo "[2/6] Waiting for SSH (first pull of a cold image can take ~15 min)..."
for i in $(seq 1 150); do
    INFO=$(api "https://api.runpod.io/v2/pods/$POD_ID")
    # runtime.ports is a list of {ip, private, public, type}; runtime itself is null until
    # networking is assigned, a few minutes after the pod first reports RUNNING.
    read -r IP PORT <<<"$(echo "$INFO" | jqp "
rt = d.get('runtime') or {}
for p in (rt.get('ports') or []):
    if p.get('private') == 22 and p.get('type') == 'tcp':
        print(p['ip'], p['public']); break
" || true)" || true   # empty read returns 1, which set -e would treat as fatal
    if [ -n "${IP:-}" ] && [ -n "${PORT:-}" ] && [ "$PORT" != "None" ]; then
        if ssh -i "$SSH_KEY" -p "$PORT" -o StrictHostKeyChecking=no -o ConnectTimeout=8 \
               root@"$IP" true 2>/dev/null; then
            echo "  ssh root@$IP -p $PORT"
            break
        fi
    fi
    [ $((i % 12)) -eq 0 ] && echo "  ...still booting ($i/150)"
    sleep 10
done
[ -n "${IP:-}" ] || { echo "ERROR: pod never became reachable"; exit 1; }

SSHC="ssh -i $SSH_KEY -p $PORT -o StrictHostKeyChecking=no root@$IP"

echo "[3/6] Seeding /workspace from the image, then uploading data..."
# The image bakes code at /opt/training-pipe because the persistent volume mounts over
# /workspace. Copy it across, then rsync only what differs locally.
$SSHC "cp -rn /opt/training-pipe/. $REMOTE/ 2>/dev/null; mkdir -p $REMOTE/data/formatted $REMOTE/output"
rsync -rlptz --no-o --no-g --exclude '__pycache__' --exclude 'unsloth_compiled_cache' \
    --info=progress2 -e "ssh -i $SSH_KEY -p $PORT -o StrictHostKeyChecking=no" \
    src configs scripts root@"$IP":$REMOTE/
rsync -rlptz --no-o --no-g --exclude '__pycache__' --exclude 'unsloth_compiled_cache' \
    --info=progress2 -e "ssh -i $SSH_KEY -p $PORT -o StrictHostKeyChecking=no" \
    "$DATASET" "$EVALSET" root@"$IP":$REMOTE/data/formatted/

# No pip step: the image ships the stack. Verify it against the real GPU before spending an
# hour on a broken pod, then prefetch weights (deliberately not baked, so one image serves
# every model in the ladder).
echo "[4/6] Verifying stack + prefetching base model..."
$SSHC "verify-stack" || { echo "ERROR: image stack is broken on this GPU"; exit 1; }

MODEL=$(python3 -c "import json;print(json.load(open('$CONFIG'))['model_name'])")
$SSHC "nohup python -c \"
from huggingface_hub import snapshot_download
snapshot_download('$MODEL')
open('/workspace/.setup_done','w').close()
\" > /workspace/setup.log 2>&1 &"

for i in $(seq 1 60); do
    $SSHC "test -f /workspace/.setup_done" 2>/dev/null && break
    [ $((i % 6)) -eq 0 ] && echo "  ...downloading $MODEL ($i/60)"
    sleep 10
done
$SSHC "test -f /workspace/.setup_done" || { echo "ERROR: model download stalled; see /workspace/setup.log"; exit 1; }

echo "[5/6] Launching training (3 epochs, q4, effective batch 16)..."
$SSHC "cd $REMOTE && nohup python src/train/train_unsloth.py --config $CONFIG --no-wandb \
    > /workspace/train.log 2>&1 & echo started"

echo "[6/6] Live. Pod stays up after training so you can pull artifacts."
cat <<EOF

  watch:     ssh -i $SSH_KEY -p $PORT root@$IP 'tail -f /workspace/train.log'
  gpu:       ssh -i $SSH_KEY -p $PORT root@$IP 'nvidia-smi'
  pull:      rsync -avzP -e 'ssh -i $SSH_KEY -p $PORT' \\
               root@$IP:$REMOTE/output/agentic_minicpm5_1b_q4_h100/ \\
               output/agentic_minicpm5_1b_q4_h100/
  terminate: curl -s -X DELETE https://api.runpod.io/v2/pods/$POD_ID -H "Authorization: Bearer \$KEY"

  POD_ID=$POD_ID  IP=$IP  PORT=$PORT
EOF
echo "$POD_ID $IP $PORT" > /tmp/minicpm5_pod.txt
