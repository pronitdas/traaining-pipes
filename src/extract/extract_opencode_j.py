"""Extract conversations from the 76GB /mnt/j opencode.db.

Genuinely streaming: one session is read, converted, written, and freed before the next
begins. Both lookups are index-backed -- `message_session_time_created_id_idx` on
(session_id, time_created, id) and `part_session_idx` on (session_id) -- so per-session
queries never scan the 935K-row part table.

The previous version claimed to stream but called fetchall() over all 935K parts while
holding every session, message and part in one dict, and contained a dead first pass that
iterated all parts only to increment a counter before reconnecting and redoing the work.
On a 76GB database that is double I/O plus an OOM.

Usage:
    python src/extract/extract_opencode_j.py [--db PATH] [--limit N]
"""

import argparse
import json
import os
import sqlite3
from pathlib import Path

import jsonlines
from tqdm import tqdm

from common import truncate_json_like

DB_PATH = "/mnt/j/geospatial-gopel/opencode.db"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "raw" / "opencode_j"

# Read-only pragmas. immutable=1 tells SQLite the file cannot change, which skips locking.
PRAGMAS = [
    "PRAGMA query_only=ON",
    "PRAGMA mmap_size=1073741824",   # 1GB mmap
    "PRAGMA cache_size=-262144",     # 256MB page cache
    "PRAGMA temp_store=MEMORY",
]

# Parts carrying no training signal.
SKIP_PART_TYPES = {"step-start", "step-finish", "compaction", "snapshot"}


def build_turns(messages, parts_by_msg, stats):
    """Convert one session's rows into the shared conversation schema."""
    turns = []
    for mid, data in messages:
        role = data.get("role", "")
        parts = parts_by_msg.get(mid, [])

        if role == "user":
            # Join ALL text parts. The previous version took text_parts[0] and silently
            # dropped the rest, losing content from multi-part user messages.
            texts = [p["text"] for p in parts
                     if p.get("type") == "text" and p.get("text")]
            if texts:
                turns.append({"role": "user", "content": "\n".join(texts)})

        elif role == "assistant":
            content = []
            for p in parts:
                ptype = p.get("type")
                if ptype in SKIP_PART_TYPES:
                    continue
                if ptype == "text" and p.get("text"):
                    content.append({"type": "text", "text": p["text"]})
                elif ptype == "tool":
                    state = p.get("state") or {}
                    out, removed = truncate_json_like(state.get("output"))
                    inp, inp_removed = truncate_json_like(state.get("input"))
                    stats["chars_elided"] += removed + inp_removed
                    content.append({
                        "type": "tool_call",
                        "name": p.get("tool"),
                        "input": inp,
                        "output": out,
                        "status": state.get("status"),
                    })
                elif ptype == "patch":
                    # opencode stores edit patches inline -- keep them, they are direct
                    # next-edit-prediction supervision.
                    content.append({"type": "patch", "patch": p.get("hash") or p.get("files") or p})
            if content:
                turns.append({"role": "assistant", "content": content})
    return turns


def main():
    ap = argparse.ArgumentParser(description="Stream conversations out of the /mnt/j opencode.db")
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--limit", type=int, default=None, help="only process first N sessions")
    ap.add_argument("--out", default=None)
    ap.add_argument("--min-turns", type=int, default=2)
    args = ap.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = Path(args.out) if args.out else OUTPUT_DIR / "conversations.jsonl"

    size_gb = os.path.getsize(args.db) / 1e9
    print(f"Opening {args.db} ({size_gb:.1f}GB) read-only...")
    conn = sqlite3.connect(f"file:{args.db}?mode=ro&immutable=1", uri=True, timeout=300)
    for p in PRAGMAS:
        conn.execute(p)

    sessions = conn.execute(
        "SELECT id, title, model, agent, directory FROM session ORDER BY time_created"
    ).fetchall()
    if args.limit:
        sessions = sessions[:args.limit]
    print(f"  {len(sessions):,} sessions")

    stats = {"chars_elided": 0}
    written = skipped = total_turns = total_tools = 0

    with jsonlines.open(output_file, "w") as writer:
        for sid, title, model, agent, directory in tqdm(sessions, desc="Sessions"):
            # Index-backed: (session_id, time_created, id)
            msg_rows = conn.execute(
                "SELECT id, data FROM message WHERE session_id = ? ORDER BY time_created, id",
                (sid,),
            ).fetchall()
            if len(msg_rows) < args.min_turns:
                skipped += 1
                continue

            messages = []
            for mid, data_str in msg_rows:
                try:
                    messages.append((mid, json.loads(data_str)))
                except (json.JSONDecodeError, TypeError):
                    continue

            # Index-backed: part_session_idx (session_id)
            parts_by_msg = {}
            for mid, data_str in conn.execute(
                "SELECT message_id, data FROM part WHERE session_id = ? ORDER BY message_id, id",
                (sid,),
            ):
                try:
                    parts_by_msg.setdefault(mid, []).append(json.loads(data_str))
                except (json.JSONDecodeError, TypeError):
                    continue

            turns = build_turns(messages, parts_by_msg, stats)
            if len(turns) < args.min_turns:
                skipped += 1
            else:
                writer.write({
                    "session_id": sid,
                    "source": "opencode_j",
                    "title": title,
                    "model": model,
                    "agent": agent,
                    "directory": directory,
                    "turns": turns,
                })
                written += 1
                total_turns += len(turns)
                total_tools += sum(
                    1 for t in turns
                    if isinstance(t.get("content"), list)
                    for c in t["content"]
                    if isinstance(c, dict) and c.get("type") == "tool_call"
                )
            # Explicitly drop this session's rows before the next iteration.
            del messages, parts_by_msg, msg_rows

    conn.close()

    print(f"\n{'='*60}\nopencode.db (/mnt/j) EXTRACTION COMPLETE\n{'='*60}")
    print(f"Conversations: {written:,}   (skipped {skipped:,} short/empty)")
    print(f"Total turns:   {total_turns:,}")
    print(f"Tool calls:    {total_tools:,}")
    print(f"Chars elided:  {stats['chars_elided']:,} ({stats['chars_elided']/1e9:.2f}GB of tool output)")
    print(f"Output:        {output_file} ({output_file.stat().st_size/1e6:.1f}MB)")


if __name__ == "__main__":
    main()
