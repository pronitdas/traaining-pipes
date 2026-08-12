"""Evaluate a next-edit-prediction model on the held-out cluster split.

Metrics mirror arXiv 2508.10074 so the numbers are comparable to its published
Qwen2.5-Coder-3B baseline (33.3% exact match, 42.4% LLM-judge), minus the LLM judge --
nothing in this pipeline calls a model to grade another model.

  exact_match     normalised string equality (the strict bar)
  format_valid    output closes the SEARCH/REPLACE block. The SRI paper (ACL 2026) flags
                  structured output as where small models fail, so this is tracked first-class
  line_f1         F1 over the set of emitted lines -- credits "mostly right" edits that
                  exact match discards
  first_line_hit  did the model get the first changed line right (does it start correctly)
  no_op_rate      fraction where the model just echoed the CURRENT region unchanged; a model
                  that learns to predict "no change" can score deceptively well otherwise

IMPORTANT: only meaningful when the model was NOT trained on these clusters. Check that the
eval file came from `format_next_edit.py --holdout-frac` on the same run that produced the
training file, or the numbers measure memorisation.

Usage:
    python src/eval/eval_next_edit.py --model output/nextedit_qwen35_q4/merged \
        --eval data/formatted/next_edit_eval.jsonl --limit 300
"""

import argparse
import json
import os
import re
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

TERMINATOR = ">>>>>>> UPDATED"

JUDGE_SYSTEM = (
    "You grade code-edit predictions. Given the code region being edited, the reference "
    "edit a developer actually made, and a model's predicted edit, decide whether the "
    "prediction is an acceptable substitute for the reference.\n\n"
    "Judge INTENT, not characters. Say correct when the prediction makes the same "
    "functional change, even if identifiers, formatting, comments or ordering differ. "
    "Say incorrect when it makes a different change, changes nothing, is incomplete, or "
    "would not compile/run in context.\n\n"
    'Reply with JSON only: {"correct": true|false, "reason": "<12 words max>"}'
)


def judge_one(base_url, api_key, model, prompt, gold, pred, timeout=120,
              system=JUDGE_SYSTEM, labels=("Code region being edited", "Reference edit", "Predicted edit")):
    """Ask the judge whether pred is an acceptable substitute for gold.

    `system` and `labels` are overridable so the agentic eval can grade "was this a
    reasonable next action" rather than "is this the same edit".
    """
    user = (f"### {labels[0]}\n{prompt[-1500:]}\n\n"
            f"### {labels[1]}\n{gold[:1500]}\n\n"
            f"### {labels[2]}\n{pred[:1500]}")
    body = json.dumps({
        "model": model,
        "stream": False,          # the gateway defaults to SSE
        "temperature": 0,
        "max_tokens": 120,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
    }).encode()
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}",
                 # Cloudflare in front of the gateway 403s urllib's default User-Agent.
                 "User-Agent": "curl/8.5.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.load(r)
        txt = d["choices"][0]["message"]["content"]
        m = re.search(r"\{.*\}", txt, re.S)
        if not m:
            return None, "unparseable"
        v = json.loads(m.group(0))
        return bool(v.get("correct")), str(v.get("reason", ""))[:80]
    except Exception as e:                      # network/parse failure -> not a verdict
        return None, f"{type(e).__name__}"


def normalise(text):
    text = text.split(TERMINATOR)[0]
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def line_f1(pred, gold):
    p = Counter(l for l in normalise(pred).splitlines() if l.strip())
    g = Counter(l for l in normalise(gold).splitlines() if l.strip())
    if not p or not g:
        return 1.0 if not p and not g else 0.0
    overlap = sum((p & g).values())
    if overlap == 0:
        return 0.0
    prec, rec = overlap / sum(p.values()), overlap / sum(g.values())
    return 2 * prec * rec / (prec + rec)


def current_region(prompt):
    """Pull the CURRENT block out of a prompt, to detect no-op predictions."""
    m = re.search(r"<<<<<<< CURRENT\n(.*?)\n?=======\s*$", prompt, re.S)
    return normalise(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser(description="Evaluate next-edit prediction")
    ap.add_argument("--model", required=True)
    ap.add_argument("--eval", default="data/formatted/next_edit_eval.jsonl")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--no-sample", action="store_true",
                    help="take the first N rows instead of a seeded random sample. The file "
                         "is ordered by commit, so the first N collapse onto one or two "
                         "clusters -- a 300-row prefix drew 249 examples from a single one.")
    # Targets are long: p50 ~251 tokens, p90 ~541, max ~868. A 256 cap truncated 49% of
    # generations mid-edit, which showed up as 'format invalid' and destroyed exact match.
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--max-seq-length", type=int, default=4096)
    ap.add_argument("--out", default=None, help="write per-example results as JSONL")
    ap.add_argument("--judge-url", default=os.environ.get("JUDGE_URL"),
                    help="OpenAI-compatible base url, e.g. https://ai.tardis.digital/v1")
    ap.add_argument("--judge-model", default="auto/smart")
    ap.add_argument("--judge-key", default=os.environ.get("JUDGE_KEY"))
    ap.add_argument("--judge-workers", type=int, default=8)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.eval)]
    total = len(rows)
    if args.limit and args.limit < total:
        if args.no_sample:
            rows = rows[:args.limit]
        else:
            import random
            random.Random(1234).shuffle(rows)
            rows = rows[:args.limit]
    n_clusters = len({r.get("cluster_id") for r in rows})
    print(f"Evaluating {len(rows):,} of {total:,} held-out examples "
          f"across {n_clusters} clusters (max_new_tokens={args.max_new_tokens})")

    from unsloth import FastLanguageModel
    model, tok = FastLanguageModel.from_pretrained(
        args.model, max_seq_length=args.max_seq_length, load_in_4bit=True)
    FastLanguageModel.for_inference(model)

    agg = defaultdict(float)
    per_cluster = defaultdict(lambda: {"n": 0, "em": 0})
    results = []

    for i, r in enumerate(rows, 1):
        # text= is mandatory: this tokenizer loads as a multimodal processor, and a
        # positional argument is parsed as an image source.
        ids = tok(text=r["prompt"], return_tensors="pt", truncation=True,
                  max_length=args.max_seq_length).to("cuda")
        out = model.generate(**ids, max_new_tokens=args.max_new_tokens,
                             do_sample=False, pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)

        gold, pred = r["completion"], gen
        em = normalise(pred) == normalise(gold)
        fmt = TERMINATOR in gen
        f1 = line_f1(pred, gold)
        gl = normalise(gold).splitlines()
        pl = normalise(pred).splitlines()
        first = bool(gl) and bool(pl) and gl[0] == pl[0]
        cur = current_region(r["prompt"])
        noop = cur is not None and normalise(pred) == cur

        agg["em"] += em; agg["fmt"] += fmt; agg["f1"] += f1
        agg["first"] += first; agg["noop"] += noop
        pc = per_cluster[r.get("cluster_id", "?")]
        pc["n"] += 1; pc["em"] += em
        results.append({"sha": r.get("sha"), "path": r.get("path"),
                        "cluster_id": r.get("cluster_id"), "exact_match": em,
                        "format_valid": fmt, "line_f1": round(f1, 4),
                        "no_op": noop, "prediction": gen})
        if i % 25 == 0:
            print(f"  {i}/{len(rows)}  EM={agg['em']/i:.1%}  fmt={agg['fmt']/i:.1%}  F1={agg['f1']/i:.3f}")

    n = len(rows)

    # LLM-as-judge: the NEP paper's headline metric (42.4% for a fine-tuned 3B). Exact
    # match punishes edits that are functionally right but differ in naming or comments,
    # so this is the number that actually reflects usefulness.
    if args.judge_url and args.judge_key:
        print(f"\nJudging {n:,} predictions with {args.judge_model} ...")

        def _j(idx):
            r, res = rows[idx], results[idx]
            return judge_one(args.judge_url, args.judge_key, args.judge_model,
                             r["prompt"], r["completion"], res["prediction"])

        with ThreadPoolExecutor(max_workers=args.judge_workers) as ex:
            verdicts = list(ex.map(_j, range(n)))
        graded = [v for v, _ in verdicts if v is not None]
        for res, (v, why) in zip(results, verdicts):
            res["judge_correct"], res["judge_reason"] = v, why
        agg["judge"] = sum(graded)
        agg["judge_n"] = len(graded)
        if len(graded) < n:
            print(f"  warning: {n - len(graded)} judge calls failed and are excluded")

    print(f"\n{'='*60}\nNEXT-EDIT EVAL ({n:,} held-out examples)\n{'='*60}")
    print(f"  exact_match    : {agg['em']/n:.1%}     (NEP paper 3B baseline: 33.3%)")
    print(f"  format_valid   : {agg['fmt']/n:.1%}")
    print(f"  line_f1        : {agg['f1']/n:.3f}")
    print(f"  first_line_hit : {agg['first']/n:.1%}")
    print(f"  no_op_rate     : {agg['noop']/n:.1%}   (echoed CURRENT unchanged)")
    if agg.get("judge_n"):
        print(f"  llm_judge      : {agg['judge']/agg['judge_n']:.1%}     "
              f"(NEP paper 3B baseline: 42.4%; n={int(agg['judge_n'])})")

    worst = sorted((v["em"]/v["n"], k, v["n"]) for k, v in per_cluster.items() if v["n"] >= 5)
    if worst:
        print("\n  weakest clusters (EM, n>=5):")
        for score, cid, cnt in worst[:5]:
            print(f"    {cid:<28} {score:.0%}  (n={cnt})")

    if args.out:
        with open(args.out, "w") as fh:
            for r in results:
                fh.write(json.dumps(r) + "\n")
        print(f"\n  per-example results -> {args.out}")


if __name__ == "__main__":
    main()
