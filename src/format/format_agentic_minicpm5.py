"""Re-encode the agentic SFT data into MiniCPM5's native tool-call format.

The corpus written by format_sft.py carries tool calls as in-band text markers inside
assistant content -- [TOOL_CALL]name\\n{json}[/TOOL_CALL] -- which is a house convention, not
any model's syntax. Two things follow from that, and both are why this script exists:

  1. Nothing parses it. vLLM's tool parsers look for the model's native tokens, so a model
     trained on markers emits text that arrives at the client as prose. Zed's agent panel
     needs a real `tool_calls` field, which means the model has to speak <tool_call>.
  2. Markers tokenize badly. MiniCPM5 has <tool_call>, </tool_call>, <tool_response> and
     <think> as *single* tokens (ids 2, 3, 10, 8); "[TOOL_CALL]" shatters into a handful of
     subword pieces that the model must learn to emit in exact sequence. This is the same
     trap format_next_edit.py's docstring describes running into with fim pseudo-tokens.

So: markers in, native tokens out. The `messages` schema is preserved exactly, so
train_unsloth.py templates the result with no changes.

Two fixes ride along, both forced by what the data actually contains:

  double system message  99.6% of rows (15,904/15,963) open with two system messages -- the
                         base prompt and a bare "Working directory: ..." -- from a format_sft.py
                         version predating the fix its own comment at line 84 describes. They
                         are merged into one. MiniCPM5's template tolerates both, but one
                         system turn is what every later consumer expects.
  tool-result markers    [TOOL_RESULT]...[/TOOL_RESULT] on a role:"tool" turn is redundant
                         once the template wraps it in <tool_response>, so it is unwrapped.

Usage:
    python src/format/format_agentic_minicpm5.py
    python src/format/format_agentic_minicpm5.py --src data/formatted/agentic_sft.jsonl \\
        --out data/formatted/agentic_sft_minicpm5.jsonl
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
FORMATTED = ROOT / "data" / "formatted"

# format_sft.py emits "\n[TOOL_CALL]{name}\n{json}[/TOOL_CALL]\n" -- name on the marker line,
# arguments as indented JSON up to the closing marker. The name charset must admit "." and "-":
# MCP tools are named like "context7_query-docs" and "mcp__unreal-mcp__execute_python", and a
# \w+ name group silently leaves 1,442 of those spans unconverted as raw markers.
TOOL_CALL_RE = re.compile(r"\[TOOL_CALL\]([A-Za-z0-9_.\-]+)\n(.*?)\[/TOOL_CALL\]", re.S)
TOOL_RESULT_RE = re.compile(r"\[TOOL_RESULT\](.*?)\[/TOOL_RESULT\]", re.S)
THINKING_RE = re.compile(r"\[THINKING\](.*?)\[/THINKING\]", re.S)


class Unconvertible(Exception):
    """Raised when a marker cannot be parsed; the whole conversation is dropped."""


def convert_assistant(content, stats):
    """Rewrite one assistant turn's markers into native MiniCPM5 tokens."""

    def sub_thinking(m):
        stats["thinking_blocks"] += 1
        return f"<think>\n{m.group(1).strip()}\n</think>"

    def sub_tool_call(m):
        name, raw_args = m.group(1), m.group(2).strip()
        try:
            args = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError as exc:
            raise Unconvertible(f"tool {name}: {exc}") from exc
        if isinstance(args, str):
            # Some of the corpus double-encodes arguments: the JSON body is a *string* that
            # itself contains the real JSON object. agentic_sft.jsonl (and so the eval set
            # built from it) is almost entirely like this -- 293 of 300 eval targets -- while
            # agentic_sft_dedup.jsonl is not, which is a live train/eval mismatch. Decode the
            # inner layer so both corpora end up with real argument objects.
            try:
                inner = json.loads(args)
            except json.JSONDecodeError:
                inner = None
            if isinstance(inner, dict):
                stats["double_encoded_args"] += 1
                args = inner
        if not isinstance(args, dict):
            # Anything still not an object: keep the value rather than guess at a key, and
            # count it so it stays visible instead of silently becoming {}.
            stats["non_dict_args"] += 1
            args = {"input": args}
        stats["tool_calls"] += 1
        payload = json.dumps({"name": name, "arguments": args}, ensure_ascii=False)
        return f"<tool_call>\n{payload}\n</tool_call>"

    content = THINKING_RE.sub(sub_thinking, content)
    content = TOOL_CALL_RE.sub(sub_tool_call, content)
    return content.strip()


def first_call_name(text):
    """Name of the first native <tool_call> in an already-converted turn, if any."""
    i = text.find("<tool_call>")
    if i < 0:
        return None
    body = text[i + len("<tool_call>"):].split("</tool_call>", 1)[0].strip()
    try:
        return json.loads(body).get("name")
    except json.JSONDecodeError:
        return None


def convert_messages(messages, stats):
    """Merge leading system turns and re-encode every marker. Returns a new message list."""
    out = []
    system_parts = []

    for m in messages:
        role, content = m.get("role"), m.get("content") or ""

        if role == "system":
            system_parts.append(content.strip())
            continue

        if role == "assistant":
            content = convert_assistant(content, stats)
        elif role == "tool":
            unwrapped = TOOL_RESULT_RE.sub(lambda m: m.group(1), content)
            if unwrapped != content:
                stats["tool_results"] += 1
            content = unwrapped.strip()

        if content:
            out.append({"role": role, "content": content})

    if system_parts:
        if len(system_parts) > 1:
            stats["systems_merged"] += 1
        out.insert(0, {"role": "system", "content": "\n\n".join(p for p in system_parts if p)})
    return out


def convert_sft(src, out_path, stats):
    kept = dropped = 0
    with open(src) as fh, open(out_path, "w") as w:
        for line in fh:
            row = json.loads(line)
            try:
                messages = convert_messages(row["messages"], stats)
            except Unconvertible as exc:
                stats["drop_reasons"][str(exc)[:60]] += 1
                dropped += 1
                continue
            row["messages"] = messages
            w.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1
    return kept, dropped


def convert_eval(src, out_path, stats):
    """Eval rows carry a `messages` prefix plus a raw `target` assistant turn."""
    kept = dropped = 0
    with open(src) as fh, open(out_path, "w") as w:
        for line in fh:
            row = json.loads(line)
            try:
                row["messages"] = convert_messages(row["messages"], stats)
                row["target"] = convert_assistant(row["target"], stats)
            except Unconvertible as exc:
                stats["drop_reasons"][str(exc)[:60]] += 1
                dropped += 1
                continue
            # Re-derive `tool` from the parsed call. format_agentic_eval.py matches the name
            # with \w+, which truncates MCP tools at the first "-" ("context7_query-docs" is
            # stored as "context7_query"), so 3 of 300 rows carry a name the model can never
            # match. Take the name we just parsed instead of inheriting the truncation.
            name = first_call_name(row["target"])
            if name and name != row.get("tool"):
                stats["tool_field_corrected"] += 1
                row["tool"] = name
            w.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1
    return kept, dropped


def main():
    ap = argparse.ArgumentParser(description="Re-encode agentic data in MiniCPM5 native format")
    ap.add_argument("--src", default=str(FORMATTED / "agentic_sft_dedup.jsonl"))
    ap.add_argument("--out", default=str(FORMATTED / "agentic_sft_minicpm5.jsonl"))
    ap.add_argument("--eval-src", default=str(FORMATTED / "agentic_eval.jsonl"))
    ap.add_argument("--eval-out", default=str(FORMATTED / "agentic_eval_minicpm5.jsonl"))
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args()

    stats = Counter()
    stats["drop_reasons"] = Counter()

    kept, dropped = convert_sft(args.src, args.out, stats)
    print(f"wrote {kept:,} rows -> {args.out}   (dropped {dropped:,})")
    print(f"  tool calls converted:   {stats['tool_calls']:,}")
    print(f"  thinking blocks:        {stats['thinking_blocks']:,}")
    print(f"  tool results unwrapped: {stats['tool_results']:,}")
    print(f"  rows w/ systems merged: {stats['systems_merged']:,}")
    if stats["double_encoded_args"]:
        print(f"  double-encoded arguments decoded: {stats['double_encoded_args']:,}")
    if stats["non_dict_args"]:
        print(f"  non-dict arguments wrapped under 'input': {stats['non_dict_args']:,}")

    if not args.skip_eval:
        e_stats = Counter()
        e_stats["drop_reasons"] = Counter()
        e_kept, e_dropped = convert_eval(args.eval_src, args.eval_out, e_stats)
        print(f"\nwrote {e_kept:,} eval rows -> {args.eval_out}   (dropped {e_dropped:,})")
        print(f"  tool calls converted: {e_stats['tool_calls']:,}")
        if e_stats["double_encoded_args"]:
            print(f"  double-encoded arguments decoded: {e_stats['double_encoded_args']:,}")
        if e_stats["tool_field_corrected"]:
            print(f"  truncated `tool` names corrected: {e_stats['tool_field_corrected']:,}")
        stats["drop_reasons"].update(e_stats["drop_reasons"])

    if stats["drop_reasons"]:
        print("\ndrop reasons:")
        for reason, n in stats["drop_reasons"].most_common(10):
            print(f"  {n:>6,}  {reason}")


if __name__ == "__main__":
    main()
