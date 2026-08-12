# Context Model — Design (locked 2026-08-12)

A third model for the tardis pipe: a **"where to look next + why"** model. Not a
code-writer. Given an explicit trigger, it surfaces relevant context and points at
the next upstream/downstream consumer to visit, with a short sequential thought.

## Locked decisions

| Decision | Choice |
|---|---|
| Harness | Editor-agnostic **LSP server**; **Zed** is the first client (thin extension) |
| Triggers | **Everything explicit** — two commands, nothing fires automatically |
| Output | **Structured JSON** contract (below) |
| Retrieval | **Codegraph + co-change stats retrieve; model only ranks + thinks** |
| Phase-1 brain | **Qwen3.8-27B via `ai.tardis.digital`** (prompted, $0 GPU spend) |
| Final model | **Extreme-mini LoRA** (size TBD, ~1–4B class), served RunPod serverless |
| Training data | **Blended**: usage logs + co-change transitions w/ distilled thoughts + opencode.db error traces |
| Spend gate | No GPU money until the prompted phase has logged real usage and a hit@3 baseline |

## Triggers (explicit only)

- `tardis.whereNext` — invoked when a code block is done. Input: current file,
  enclosing symbol, recent edit history (server-tracked `didChange`).
  Output: ranked upstream/downstream targets + thought.
- `tardis.explainError` — invoked with a selected/pasted error + cursor position.
  Output: relevant surrounding context (snippets) + thought.

LSP constraint that forced this: our server cannot see other language servers'
diagnostics, so the error path must be explicit (selection/paste).

## Output contract (locked — do not change without re-locking)

```json
{
  "trigger": "block_done | error",
  "thought": "1-2 sentence next-thought, sequential-thinking style",
  "targets": [
    {"file": "src/server.py", "symbol": "load_app", "dir": "upstream",
     "why": "calls parse_config, unwraps old dict shape"}
  ],
  "snippets": [
    {"file": "src/server.py", "lines": "40-55", "text": "..."}
  ]
}
```

## Architecture

```
Zed ──(extension)──> LSP server ──> retrieval layer ──> brain ──> JSON ──> Zed
                        │              codegraph (callers/callees/blast radius)
                        │              co-change stats (data/mined/clusters.json)
                        │              recent-edit tracker (didChange history)
                        └── logger: (context, candidates, output, actual-next-jump) → JSONL
```

The retrieval layer is deterministic and spends zero tokens. The model never
searches; it receives a candidate set and only ranks + writes the thought.
This is what lets the final model be tiny.

## Phases

1. **Harness ($0):** LSP server + Zed extension + codegraph/co-change retrieval +
   prompted Qwen3.8-27B on ai.tardis.digital, forced into the JSON contract.
   Live with it during real work.
2. **Collect (free byproduct):** every invocation logs the full context, the
   candidates, the model's answer, and **where the user actually navigated next**
   (observed via subsequent didOpen/didChange). The actual-next-jump is
   ground-truth for the ranking objective.
3. **Train (only spend):** extreme-mini LoRA on the blend:
   - usage logs (primary — real preferences)
   - co-change transitions from mined commits, with "why" thoughts distilled
     offline by a big model
   - error→reasoning→fix traces extracted from opencode.db
   Serve on RunPod serverless. Reuse pipe lessons: manual prepack, sha-only
   dedupe, four-axis split verification, adapter saved first.

## Success metric

**hit@3**: the place the user actually went next appears in the model's top-3
targets. Computed automatically from logs. The prompted 27B sets the baseline;
the mini model must match or beat it at a fraction of the serving cost.

## Explicitly out of scope (v1)

- Automatic/ambient triggering of any kind
- Editors other than Zed (server is LSP-generic, but only Zed gets a client)
- Model-side retrieval / unindexed-repo fallback
- Code generation or edit application
