"""Evaluate the agentic model on next-tool-call prediction.

Given a conversation truncated before an assistant turn that issued a tool call, the model
must produce that turn. We score:

  tool_match      picked the same tool the developer picked
  format_valid    emitted a well-formed tool-call block in the trained encoding
  called_any      called a tool at all rather than replying in prose
  args_overlap    token overlap of the JSON arguments, for the calls that matched
  llm_judge       is this a reasonable next action given the conversation?

Read tool_match against the majority baseline printed alongside it. `bash` alone is 28.5%
of all calls, so a model that always answers "bash" scores 28.5% while being useless.

llm_judge matters more here than in next-edit prediction: agentic work is genuinely
multi-path, and grepping where the developer happened to read a file is often equally
correct. tool_match is the strict lower bound, the judge is the fair one.

--format selects the tool-call encoding the model was trained on. The two corpora are the
same conversations in different clothes, so the scores are directly comparable.

Usage:
    python src/eval/eval_agentic.py --model unsloth/Qwen3.6-35B-A3B \
        --adapter output/lora_qwen36_agentic_q4/final --limit 300

    python src/eval/eval_agentic.py --model output/agentic_minicpm5_1b_q4/merged \
        --format minicpm5 --eval data/formatted/agentic_eval_minicpm5.jsonl
"""

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from eval_next_edit import judge_one          # shared gateway client (sends a User-Agent)

TOOL_RE = re.compile(r"\[TOOL_CALL\]([A-Za-z0-9_.\-]+)")
# The argument body is usually an object, but 293 of the 300 rows in agentic_eval.jsonl carry
# it double-encoded as a JSON *string* containing the real object. Matching only `{...}` scored
# format_valid at 7/300 no matter how good the model was, so accept both and unwrap below.
BLOCK_RE = re.compile(
    r"\[TOOL_CALL\]([A-Za-z0-9_.\-]+)\s*(\{.*?\}|\".*?\")?\s*\[/TOOL_CALL\]", re.S)

# MiniCPM5 emits its native single-token form instead, with the name inside the JSON:
#   <tool_call>\n{"name": ..., "arguments": {...}}\n</tool_call>
NATIVE_BLOCK_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)

JUDGE_SYSTEM_AGENTIC = (
    "You grade an AI coding agent's choice of next action. You are given the conversation "
    "so far, the action the developer actually took, and the action the model predicted.\n\n"
    "Say correct when the predicted action is a reasonable next step for the stated goal, "
    "even if it differs from the reference -- searching instead of reading, or a different "
    "but sensible command, can be equally valid. Say incorrect when it is off-task, repeats "
    "work already done, would error, or replies in prose where a tool call was needed.\n\n"
    'Reply with JSON only: {"correct": true|false, "reason": "<12 words max>"}'
)


def first_tool(text):
    m = TOOL_RE.search(text or "")
    return m.group(1) if m else None


def tool_args(text):
    m = BLOCK_RE.search(text or "")
    if not m or not m.group(2):
        return None
    try:
        parsed = json.loads(m.group(2))
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, str):          # double-encoded: the string *is* the object
        try:
            parsed = json.loads(parsed)
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


def _native_payload(text):
    m = NATIVE_BLOCK_RE.search(text or "")
    if not m:
        return None
    try:
        payload = json.loads(m.group(1))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def native_first_tool(text):
    p = _native_payload(text)
    return p.get("name") if p else None


def native_tool_args(text):
    p = _native_payload(text)
    args = p.get("arguments") if p else None
    return args if isinstance(args, dict) else None


# Everything format-dependent lives here; the scoring below is shared. `skip_special` is the
# subtle one: <tool_call> and </tool_call> are single *special* tokens in MiniCPM5's vocab, so
# decoding with skip_special_tokens=True deletes exactly the delimiters we parse and stop on.
FORMATS = {
    "markers": {
        "first_tool": first_tool,
        "tool_args": tool_args,
        "block_re": BLOCK_RE,
        "stop": "[/TOOL_CALL]",
        "skip_special": True,
    },
    "minicpm5": {
        "first_tool": native_first_tool,
        "tool_args": native_tool_args,
        "block_re": NATIVE_BLOCK_RE,
        "stop": "</tool_call>",
        "skip_special": False,
    },
}


def args_overlap(a, b):
    if not isinstance(a, dict) or not isinstance(b, dict):
        return 0.0
    ta = Counter(str(a.get(k))[:200] for k in a)
    tb = Counter(str(b.get(k))[:200] for k in b)
    if not ta or not tb:
        return 0.0
    inter = sum((ta & tb).values())
    return 2 * inter / (sum(ta.values()) + sum(tb.values()))


def main():
    ap = argparse.ArgumentParser(description="Evaluate agentic next-tool-call prediction")
    ap.add_argument("--model", required=True, help="base model or merged dir")
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir to apply to --model")
    ap.add_argument("--eval", default="data/formatted/agentic_eval.jsonl")
    ap.add_argument("--format", choices=sorted(FORMATS), default="markers",
                    help="tool-call encoding the model was trained on: 'markers' for the "
                         "in-band [TOOL_CALL] corpus, 'minicpm5' for native <tool_call>")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--max-seq-length", type=int, default=8192)
    ap.add_argument("--out", default=None)
    ap.add_argument("--judge-url", default=os.environ.get("JUDGE_URL"))
    ap.add_argument("--judge-model", default="auto/smart")
    ap.add_argument("--judge-key", default=os.environ.get("JUDGE_KEY"))
    ap.add_argument("--judge-workers", type=int, default=8)
    args = ap.parse_args()

    fmt_spec = FORMATS[args.format]
    parse_tool, parse_args = fmt_spec["first_tool"], fmt_spec["tool_args"]
    block_re, stop_str = fmt_spec["block_re"], fmt_spec["stop"]

    rows = [json.loads(l) for l in open(args.eval)][:args.limit]
    baseline = rows[0].get("majority_baseline", 0.0)
    maj_tool = rows[0].get("majority_tool", "?")
    print(f"Evaluating {len(rows):,} next-tool-call examples "
          f"(majority baseline: always '{maj_tool}' = {baseline:.1%})")
    print(f"  tool-call format: {args.format}  (stop on {stop_str!r})")
    if not any(block_re.search(r.get("target", "")) for r in rows):
        raise SystemExit(
            f"no {args.format}-format tool calls found in {args.eval} -- "
            f"--format and --eval disagree about the encoding")

    from unsloth import FastLanguageModel
    model, tok = FastLanguageModel.from_pretrained(
        args.model, max_seq_length=args.max_seq_length, load_in_4bit=True)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        print(f"  applied adapter: {args.adapter}")
    FastLanguageModel.for_inference(model)

    # Mid-conversation the model rarely emits EOS, so greedy decoding runs to the full
    # token cap on every example even though half the answers close within ~82 tokens.
    # Stopping at the closing marker cuts decode time roughly threefold.
    from transformers import StoppingCriteria, StoppingCriteriaList

    class StopOnToolClose(StoppingCriteria):
        def __init__(self, tokenizer, prompt_len):
            self.tok = tokenizer
            self.prompt_len = prompt_len

        def __call__(self, input_ids, scores, **kw):
            tail = self.tok.decode(input_ids[0][self.prompt_len:][-24:],
                                   skip_special_tokens=fmt_spec["skip_special"])
            return stop_str in tail

    agg = defaultdict(float)
    results = []
    confusion = Counter()

    for i, r in enumerate(rows, 1):
        prompt = tok.apply_chat_template(r["messages"], tokenize=False,
                                         add_generation_prompt=True)
        # text= is mandatory: Qwen3.6's tokenizer loads as a multimodal processor and a
        # positional argument is parsed as an image source.
        ids = tok(text=prompt, return_tensors="pt", truncation=True,
                  max_length=args.max_seq_length).to("cuda")
        plen = ids["input_ids"].shape[1]
        out = model.generate(**ids, max_new_tokens=args.max_new_tokens,
                             do_sample=False, pad_token_id=tok.eos_token_id,
                             stopping_criteria=StoppingCriteriaList(
                                 [StopOnToolClose(tok, plen)]))
        gen = tok.decode(out[0][plen:], skip_special_tokens=fmt_spec["skip_special"])

        pred_tool = parse_tool(gen)
        match = pred_tool == r["tool"]
        fmt = bool(block_re.search(gen))
        called = pred_tool is not None
        ov = args_overlap(parse_args(gen), parse_args(r["target"])) if match else 0.0

        agg["match"] += match; agg["fmt"] += fmt; agg["called"] += called
        agg["ov"] += ov
        if not match:
            confusion[f"{r['tool']} -> {pred_tool}"] += 1
        results.append({"gold_tool": r["tool"], "pred_tool": pred_tool,
                        "tool_match": match, "format_valid": fmt,
                        "called_any": called, "args_overlap": round(ov, 3),
                        "prediction": gen[:2000]})
        if i % 25 == 0:
            print(f"  {i}/{len(rows)}  tool={agg['match']/i:.1%}  fmt={agg['fmt']/i:.1%}  "
                  f"called={agg['called']/i:.1%}")

    n = len(rows)

    if args.judge_url and args.judge_key:
        print(f"\nJudging {n:,} predictions with {args.judge_model} ...")

        def _j(idx):
            r, res = rows[idx], results[idx]
            convo = "\n".join(f"[{m['role']}] {(m.get('content') or '')[:400]}"
                              for m in r["messages"][-6:])
            return judge_one(args.judge_url, args.judge_key, args.judge_model,
                             convo, r["target"][:1200], res["prediction"][:1200],
                             system=JUDGE_SYSTEM_AGENTIC,
                             labels=("Conversation so far", "Action the developer took",
                                     "Action the model predicted"))

        with ThreadPoolExecutor(max_workers=args.judge_workers) as ex:
            verdicts = list(ex.map(_j, range(n)))
        graded = [v for v, _ in verdicts if v is not None]
        for res, (v, why) in zip(results, verdicts):
            res["judge_correct"], res["judge_reason"] = v, why
        agg["judge"] = sum(graded)
        agg["judge_n"] = len(graded)
        if len(graded) < n:
            print(f"  warning: {n-len(graded)} judge calls failed and are excluded")

    print(f"\n{'='*60}\nAGENTIC NEXT-TOOL-CALL EVAL ({n:,} examples)\n{'='*60}")
    print(f"  tool_match     : {agg['match']/n:.1%}     (majority baseline '{maj_tool}': {baseline:.1%})")
    print(f"  format_valid   : {agg['fmt']/n:.1%}")
    print(f"  called_any     : {agg['called']/n:.1%}   (rest replied in prose)")
    if agg["match"]:
        print(f"  args_overlap   : {agg['ov']/agg['match']:.3f}   (among matched tools)")
    if agg.get("judge_n"):
        print(f"  llm_judge      : {agg['judge']/agg['judge_n']:.1%}   (n={int(agg['judge_n'])})")
    if confusion:
        print("\n  most common mispredictions (gold -> predicted):")
        for k, v in confusion.most_common(6):
            print(f"    {k:<34} {v}")

    if args.out:
        with open(args.out, "w") as fh:
            for r in results:
                fh.write(json.dumps(r) + "\n")
        print(f"\n  per-example results -> {args.out}")


if __name__ == "__main__":
    main()
