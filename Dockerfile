# Frozen training base for RunPod.
#
# Everything that used to be rediscovered per-run lives here instead: the CUDA/torch base,
# the Unsloth stack, and the repo code. A pod built from this image is ready to train in the
# time it takes to pull, with no pip install step and no version drift between runs.
#
# What is deliberately NOT baked in:
#   model weights  downloaded per run (~1-2 min for a 1B on a pod's network). Keeping them
#                  out means one image serves every model instead of one tag per model.
#   training data  it is personal conversation history. It is rsync'd to the pod at run time
#                  and must never enter an image layer or a git object.
#
# Build: GitHub Actions -> ghcr.io/pronitdas/training-pipe:base (see .github/workflows/).
# Local Docker is not required and is currently broken on the dev machine.

# runpod/pytorch retired the "2.5.1-py3.11-cuda12.4-cudnn-devel" scheme; the current family is
# 1.1.0-rc.<n>-cu<ver>-torch<ver>-ubuntu<ver>. cu12.9 + torch 2.9.1 covers H100 (sm_90).
FROM runpod/pytorch:1.1.0-rc.154-cu1290-torch291-ubuntu2404

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/workspace/huggingface \
    PYTHONUNBUFFERED=1 \
    # 8GB-class fragmentation guard is harmless on an H100 and helps small-card reuse.
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN apt-get update && apt-get install -y --no-install-recommends \
        rsync tmux git curl \
    && rm -rf /var/lib/apt/lists/*

# Unsloth pins its own transformers/trl; let it resolve rather than fighting it. Versions are
# pinned so a rebuild six months from now produces the same trainer that produced the weights.
#
# Do NOT `import unsloth` here. unsloth_zoo resolves the accelerator at import time and raises
# NotImplementedError("Unsloth cannot find any torch accelerator? You need a GPU.") on a
# CPU-only machine -- which every CI builder is. The import check belongs at runtime on the
# pod (see verify-stack below); at build time we can only assert the packages are present.
RUN pip install --no-cache-dir \
        unsloth==2026.8.15 \
        unsloth_zoo==2026.8.10 \
    && pip install --no-cache-dir \
        datasets trl peft bitsandbytes accelerate \
    && python -c "import importlib.metadata as m; \
print('unsloth', m.version('unsloth'), '| unsloth_zoo', m.version('unsloth_zoo'), \
'| torch', m.version('torch'), '| trl', m.version('trl'), '| peft', m.version('peft'))"

# Code lives in /opt, NOT /workspace. A pod mounts its persistent volume at /workspace, which
# mounts straight over anything baked there -- code copied to /workspace/training-pipe in the
# image is invisible the moment the pod starts. /opt survives the mount; the run script seeds
# /workspace from it at boot.
WORKDIR /opt/training-pipe
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
RUN mkdir -p data/formatted output/logs && chmod +x scripts/*.sh

# Fail fast and loudly if a pod ever comes up with a broken stack, rather than 10 minutes into
# a run. `docker run <image> verify` executes this; the default command keeps the pod alive.
COPY <<'EOF' /usr/local/bin/verify-stack
#!/usr/bin/env bash
set -e
python - <<'PY'
import torch
from unsloth import FastLanguageModel
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| gpu", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY
EOF
RUN chmod +x /usr/local/bin/verify-stack

# Must stay /start.sh. That is the base image's own command: it provisions authorized_keys
# from the PUBLIC_KEY env var and starts sshd. Overriding it with a shell produces a pod that
# reports RUNNING and refuses every SSH connection, with no error anywhere to explain why.
CMD ["/start.sh"]
