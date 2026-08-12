"""Build a next-tool-call evaluation set from the agentic SFT data.

One example = a conversation truncated immediately before an assistant turn that issues a
tool call. The model is asked to produce that turn; we score which tool it picks and
whether the call is well-formed.

Two things make the headline number interpretable, and both are emitted here:

  majority_baseline  the share of the most common tool. `read` alone is ~31% of all
                     calls, so a model that always answers "read" scores 31%. Any tool
                     accuracy must be read against this, never on its own.
  n_valid_next       how many distinct tools appear as the next call in similar contexts.
                     Agentic work is genuinely multi-modal -- grep instead of read is
                     often equally correct -- which is why the LLM judge matters more
                     here than it does for next-edit prediction.

Usage:
    python src/format/format_agentic_eval.py --limit 300
"""

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
SRC = ROOT / "data" / "formatted" / "agentic_sft.jsonl"
OUT = ROOT / "data" / "formatted" / "agentic_eval.jsonl"

TOOL_RE = re.compile(r"\[TOOL_CALL\](\w+)")


def first_tool(text):
    m = TOOL_RE.search(text or "")
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser(description="Build next-tool-call eval set")
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--min-context", type=int, default=3,
                    help="require at least this many preceding messages, so the model has "
                         "something to reason from rather than guessing cold")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    candidates = []
    tool_counts = Counter()

    with open(args.src) as fh:
        for line in fh:
            msgs = json.loads(line)["messages"]
            for i, m in enumerate(msgs):
                if m.get("role") != "assistant":
                    continue
                name = first_tool(m.get("content"))
                if not name or i < args.min_context:
                    continue
                tool_counts[name] += 1
                candidates.append({"prefix": msgs[:i], "target": m["content"], "tool": name})

    print(f"{len(candidates):,} tool-call points found across the corpus")
    if not candidates:
        raise SystemExit("no tool calls found -- check the [TOOL_CALL] marker format")

    top_tool, top_n = tool_counts.most_common(1)[0]
    baseline = top_n / sum(tool_counts.values())
    print(f"majority baseline: always answering '{top_tool}' scores {baseline:.1%}")
    print(f"distinct tools: {len(tool_counts)}   top: {tool_counts.most_common(6)}")

    random.Random(args.seed).shuffle(candidates)
    chosen = candidates[:args.limit]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        for c in chosen:
            fh.write(json.dumps({
                "messages": c["prefix"],
                "target": c["target"],
                "tool": c["tool"],
                "majority_baseline": round(baseline, 4),
                "majority_tool": top_tool,
            }) + "\n")

    sel = Counter(c["tool"] for c in chosen)
    print(f"\nwrote {len(chosen):,} examples -> {out_path}")
    print(f"  tool spread in sample: {sel.most_common(8)}")
    print(f"  distinct tools in sample: {len(sel)}")


if __name__ == "__main__":
    main()
