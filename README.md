# Training Pipe — Personal Agentic Coding Model

Train custom LoRA adapters on SOTA open models from your own conversation history
(opencode.db + Claude Code transcripts), then serve on RunPod for inference and
build a custom autocomplete model for Zed.

## Models (SOTA as of Aug 2026)

| Purpose | Base Model | Size | Why |
|---------|-----------|------|-----|
| Agentic coding | Qwen/Qwen3.6-35B-A3B | 35B MoE / 3B active | SWE-bench 73.4, Terminal-Bench 51.5, 256K context, qwen3_xml tool parser |
| Zed autocomplete | zed-industries/zeta-2 | 8B (Seed-Coder base) | Zed's own open-weight edit-prediction model, Apache-2.0 |
| Light autocomplete | sweepai/sweep-next-edit-1.5B | 1.5B | Sub-500ms local, 67.8% accuracy on next-edit bench |

## Pipeline

```
Extract → Format → Train → Merge → Deploy
  opencode.db     SFT LoRA    merge    vLLM (RunPod)
  Claude logs     Sharegpt            OpenAI API
  GPT/Kimi        FIM data            Zed edit prediction
```

## Data Inventory

- **opencode.db** (204MB): 38 sessions, 2,383 turns, 3,584 tool calls
- **Claude transcripts** (2.2GB): 7,904 files, 691K lines, 1,078 conversations
- **Claude project files** (225MB): 235 conversations with thinking blocks
- **Total**: 1,116 conversations, 44,120 turns, 13,631 tool calls, 6,360 thinking blocks

## Quick Start

```bash
# 1. Extract all conversations
uv run python src/extract/extract_all.py

# 2. Format for SFT training
uv run python src/format/format_sft.py

# 3. Extract FIM data from your codebases
uv run python src/autocomplete/extract_fim.py

# 4. Upload to RunPod and train (agentic LoRA)
bash scripts/runpod_train.sh A100_80GB

# 5. Deploy inference server
bash scripts/runpod_deploy.sh A100_80GB

# 6. Train FIM autocomplete model
bash scripts/runpod_train_fim.sh A100_80GB
```

## Training Config

### Agentic LoRA (Qwen3.6-35B-A3B)
- LoRA rank: 128, alpha: 256 (high rank for MoE)
- 4-bit NF4 QLoRA (fits on A100 80GB)
- 3 epochs, cosine schedule, LR 2e-4
- Max seq length: 32K
- Tool-call format: qwen3_xml (native Qwen3.6 format)
- Target modules: all attention + MLP projections

### FIM Autocomplete (zeta-2)
- LoRA rank: 32, alpha: 64
- 4-bit QLoRA
- 2 epochs, LR 5e-4
- Max seq length: 4K
- Purpose-built for Zed edit prediction

## Deploy

### Agentic inference (vLLM)
```bash
vllm serve output/merged \
  --served-model-name qwen3.6-coding \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --trust-remote-code
```

### FIM autocomplete server
```bash
python src/autocomplete/serve_fim.py --model output/fim_autocomplete/merged --port 8080
```

## Requirements

- RunPod account with GPU pods (A100 80GB recommended, RTX 4090 works with reduced config)
- Python 3.12+, uv package manager
- ~70GB disk for model weights, ~10GB for training data
