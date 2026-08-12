"""Extract conversations from Codex CLI rollout sessions (~/.codex/sessions).

Codex writes one JSONL per session under sessions/YYYY/MM/DD/rollout-<ts>-<uuid>.jsonl.
Each line is {timestamp, type, payload} where type is one of:

    session_meta   once, carries id / cwd / git commit+branch
    turn_context   per turn, carries model + effort
    event_msg      UI-level events (duplicates of message content) -- ignored
    response_item  the actual conversation

response_item payloads seen in the corpus:
    message                  role + content[{type: input_text|output_text, text}]
    reasoning                model-private chain of thought -- skipped by default
    function_call            name + arguments + call_id
    function_call_output     call_id + output
    custom_tool_call         name + input + call_id   (e.g. apply_patch)
    custom_tool_call_output  call_id + output

Tool calls and their outputs arrive as separate items linked by call_id, so calls are
buffered and resolved when the matching output appears.

Output matches the schema used by extract_claude.py / extract_opencode_j.py so
extract_all.py:merge_sources() can fold it in unchanged.

Usage:
    python src/extract/extract_codex.py [--sessions DIR] [--limit N] [--keep-reasoning]
"""

import argparse
import json
from pathlib import Path

import jsonlines
from tqdm import tqdm

from common import truncate_output

SESSIONS_DIR = Path.home() / ".codex" / "sessions"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "raw" / "codex"

TOOL_CALL_TYPES = {"function_call", "custom_tool_call"}
TOOL_OUTPUT_TYPES = {"function_call_output", "custom_tool_call_output"}


def _text_of(content):
    """Join the text fields of a Codex message content list."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    out = []
    for c in content:
        if isinstance(c, dict) and c.get("text"):
            out.append(c["text"])
        elif isinstance(c, str):
            out.append(c)
    return "\n".join(out)


def parse_session(path, stats, keep_reasoning=False):
    """Parse one rollout file into a conversation dict, or None if too short."""
    meta = {}
    turns = []
    pending = {}          # call_id -> tool_call dict awaiting its output
    assistant_buf = []    # content parts accumulating for the current assistant turn

    def flush_assistant():
        if assistant_buf:
            turns.append({"role": "assistant", "content": list(assistant_buf)})
            assistant_buf.clear()

    with open(path, "r", errors="replace") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rtype = rec.get("type")
            payload = rec.get("payload") or {}

            if rtype == "session_meta":
                meta = payload
                continue
            if rtype == "turn_context":
                meta.setdefault("model", payload.get("model"))
                continue
            if rtype != "response_item":
                continue    # event_msg duplicates message content

            ptype = payload.get("type")

            if ptype == "message":
                role = payload.get("role")
                text = _text_of(payload.get("content"))
                if not text:
                    continue
                if role == "user":
                    flush_assistant()
                    turns.append({"role": "user", "content": text})
                elif role == "assistant":
                    assistant_buf.append({"type": "text", "text": text})

            elif ptype == "reasoning":
                if keep_reasoning:
                    summary = _text_of(payload.get("summary"))
                    if summary:
                        assistant_buf.append({"type": "reasoning", "text": summary})

            elif ptype in TOOL_CALL_TYPES:
                raw_args = payload.get("arguments") or payload.get("input") or ""
                args, removed = truncate_output(raw_args)
                stats["chars_elided"] += removed
                call = {
                    "type": "tool_call",
                    "name": payload.get("name"),
                    "input": args,
                    "output": None,
                    "status": payload.get("status"),
                }
                assistant_buf.append(call)
                cid = payload.get("call_id")
                if cid:
                    pending[cid] = call

            elif ptype in TOOL_OUTPUT_TYPES:
                cid = payload.get("call_id")
                out, removed = truncate_output(payload.get("output") or "")
                stats["chars_elided"] += removed
                call = pending.pop(cid, None)
                if call is not None:
                    call["output"] = out
                else:
                    # Output with no buffered call (truncated/resumed session).
                    assistant_buf.append({
                        "type": "tool_call", "name": None,
                        "input": None, "output": out, "status": None,
                    })

    flush_assistant()
    if len(turns) < 2:
        return None

    git = meta.get("git") or {}
    return {
        "session_id": meta.get("id") or path.stem,
        "source": "codex",
        "title": None,
        "model": meta.get("model"),
        "agent": meta.get("originator"),
        "directory": meta.get("cwd"),
        "git_commit": git.get("commit_hash"),
        "git_branch": git.get("branch"),
        "turns": turns,
    }


def main():
    ap = argparse.ArgumentParser(description="Extract Codex CLI rollout sessions")
    ap.add_argument("--sessions", default=str(SESSIONS_DIR))
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--keep-reasoning", action="store_true",
                    help="retain reasoning summaries (off by default: it is another model's CoT format)")
    args = ap.parse_args()

    files = sorted(Path(args.sessions).rglob("rollout-*.jsonl"))
    if args.limit:
        files = files[:args.limit]
    print(f"Found {len(files):,} Codex rollout files under {args.sessions}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = Path(args.out) if args.out else OUTPUT_DIR / "conversations.jsonl"

    stats = {"chars_elided": 0}
    written = skipped = total_turns = total_tools = 0

    with jsonlines.open(output_file, "w") as writer:
        for path in tqdm(files, desc="Sessions"):
            try:
                conv = parse_session(path, stats, args.keep_reasoning)
            except OSError:
                conv = None
            if conv is None:
                skipped += 1
                continue
            writer.write(conv)
            written += 1
            total_turns += len(conv["turns"])
            total_tools += sum(
                1 for t in conv["turns"]
                if isinstance(t.get("content"), list)
                for c in t["content"]
                if isinstance(c, dict) and c.get("type") == "tool_call"
            )

    print(f"\n{'='*60}\nCODEX EXTRACTION COMPLETE\n{'='*60}")
    print(f"Conversations: {written:,}   (skipped {skipped:,} short/empty)")
    print(f"Total turns:   {total_turns:,}")
    print(f"Tool calls:    {total_tools:,}")
    print(f"Chars elided:  {stats['chars_elided']:,}")
    print(f"Output:        {output_file} ({output_file.stat().st_size/1e6:.1f}MB)")


if __name__ == "__main__":
    main()
