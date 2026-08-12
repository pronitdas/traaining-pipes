"""Build a next-edit-prediction dataset from mined commits + co-change clusters.

One example = "given the edits already made in this commit, produce the next one".
Hunks are ordered by (path, start line) -- git records no intra-commit chronology, so
this is an explicit approximation, not ground truth.

Coherence filter: every *clustered* file in the commit must belong to the same co-change
cluster. Files the clustering had too little evidence for are tolerated; a commit whose
files straddle two clusters is dropped. This is the mechanical stand-in for the
GPT-4o-mini pass that arXiv 2508.10074 used to discard 72.8% of incoherent edit sequences.

Output format is deliberately plain ASCII -- `### headers` and git merge markers, no
pseudo-tokens. A previous run fed `<|fim_prefix|>` to a model whose vocabulary had no such
token, and every marker shattered into 7-8 subword tokens. Merge markers tokenize
correctly in any vocabulary, so that failure cannot recur.

Usage:
    python src/format/format_next_edit.py [--context 10] [--max-region-lines 80]
"""

import argparse
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).parent.parent.parent
COMMITS = ROOT / "data" / "mined" / "commits.jsonl"
CLUSTERS = ROOT / "data" / "mined" / "clusters.json"
OUTPUT = ROOT / "data" / "formatted" / "next_edit.jsonl"

HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def load_cluster_map(path):
    cl = json.load(open(path))
    f2c = {}
    members = defaultdict(list)
    for repo, d in cl["repos"].items():
        for c in d["clusters"]:
            for f in c["files"]:
                f2c[(repo, f)] = c["cluster_id"]
                members[c["cluster_id"]].append(f)
    return f2c, members


def coherent(commit, f2c):
    """Return the cluster id if all clustered files agree, else None."""
    known = {f2c.get((commit["repo"], f)) for f in commit["files"]} - {None}
    return next(iter(known)) if len(known) == 1 else None


def git_diff(repo_path, sha, path, context):
    try:
        return subprocess.run(
            ["git", "show", f"--unified={context}", "--format=", "--no-renames", sha, "--", path],
            cwd=repo_path, capture_output=True, text=True, errors="replace", timeout=60,
        ).stdout
    except (subprocess.TimeoutExpired, OSError):
        return ""


def parse_hunks(diff_text):
    """Split a unified diff into hunks of (start_line, pre_lines, post_lines, raw)."""
    hunks = []
    cur = None
    for line in diff_text.splitlines():
        m = HUNK_RE.match(line)
        if m:
            if cur:
                hunks.append(cur)
            cur = {"start": int(m.group(1)), "pre": [], "post": [], "raw": [line]}
            continue
        if cur is None:
            continue
        if line.startswith("\\"):          # "\ No newline at end of file"
            continue
        cur["raw"].append(line)
        if line.startswith("-"):
            cur["pre"].append(line[1:])
        elif line.startswith("+"):
            cur["post"].append(line[1:])
        elif line.startswith(" "):
            cur["pre"].append(line[1:])
            cur["post"].append(line[1:])
    if cur:
        hunks.append(cur)
    return [h for h in hunks if h["pre"] != h["post"]]


def build_prompt(prior, path, pre_lines, siblings):
    parts = []
    if prior:
        parts.append("### Edits already made")
        parts.extend(prior)
        parts.append("")
    if siblings:
        parts.append("### Related files (same feature cluster)")
        parts.extend(siblings)
        parts.append("")
    parts.append(f"### Current file: {path}")
    parts.append("<<<<<<< CURRENT")
    parts.append("\n".join(pre_lines))
    parts.append("=======")
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser(description="Build next-edit-prediction examples from git")
    ap.add_argument("--commits", default=str(COMMITS))
    ap.add_argument("--clusters", default=str(CLUSTERS))
    ap.add_argument("--out", default=str(OUTPUT))
    ap.add_argument("--context", type=int, default=10, help="diff context lines")
    ap.add_argument("--max-region-lines", type=int, default=80,
                    help="skip hunks whose region exceeds this (whole-file rewrites)")
    ap.add_argument("--max-siblings", type=int, default=6)
    ap.add_argument("--max-prior-hunks", type=int, default=4,
                    help="most recent N hunks kept as edit history")
    ap.add_argument("--max-prior-lines", type=int, default=40,
                    help="per-hunk line cap inside the edit history")
    ap.add_argument("--max-prompt-chars", type=int, default=11000,
                    help="~3k tokens; oldest history is dropped until the prompt fits")
    ap.add_argument("--holdout-frac", type=float, default=0.0,
                    help="reserve this fraction of CLUSTERS (not examples) as an eval split, "
                         "written alongside --out as *_eval.jsonl. Splitting by cluster rather "
                         "than by example is the point: examples from one cluster share files "
                         "and edit history, so an example-level split leaks and measures "
                         "memorisation instead of generalisation.")
    args = ap.parse_args()

    f2c, members = load_cluster_map(args.clusters)
    commits = [json.loads(l) for l in open(args.commits)]
    commits = [c for c in commits if len(c["files"]) >= 2]
    print(f"{len(commits):,} multi-file commits; {len(f2c):,} clustered files")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Choose held-out clusters up front, deterministically, so the split is reproducible
    # and identical across reruns.
    eval_clusters = set()
    if args.holdout_frac > 0:
        import random
        all_clusters = sorted(members)
        random.Random(42).shuffle(all_clusters)
        n_hold = max(1, int(len(all_clusters) * args.holdout_frac))
        eval_clusters = set(all_clusters[:n_hold])
        print(f"Holding out {len(eval_clusters)} of {len(all_clusters)} clusters for eval")

    # A commit may touch unclustered files alongside its cluster's files, so the same
    # file can be reached from a train commit and an eval commit. Splitting on clusters
    # alone therefore still leaks at file level (37 files did). Collect every file an
    # eval commit touches and bar those from the train side. Uses the file lists already
    # in commits.jsonl -- no git calls.
    eval_files = set()
    if eval_clusters:
        for c in commits:
            if coherent(c, f2c) in eval_clusters:
                eval_files.update((c["repo"], f) for f in c["files"])
        print(f"  {len(eval_files):,} files reserved to the eval side")

    eval_path = out_path.with_name(out_path.stem + "_eval" + out_path.suffix)
    n_written = n_commits = n_eval = 0
    skipped_incoherent = skipped_nohunks = skipped_big = skipped_leak = 0

    eval_fh = open(eval_path, "w") if eval_clusters else None
    with open(out_path, "w") as fh:
        for c in tqdm(commits, desc="Commits"):
            cid = coherent(c, f2c)
            if cid is None:
                skipped_incoherent += 1
                continue

            # Collect hunks per file, ordered by (path, first hunk line).
            per_file = []
            for path in sorted(c["files"]):
                hunks = parse_hunks(git_diff(c["repo_path"], c["sha"], path, args.context))
                for h in hunks:
                    per_file.append((path, h))
            per_file.sort(key=lambda x: (x[0], x[1]["start"]))
            if len(per_file) < 2:
                skipped_nohunks += 1
                continue

            touched = {p for p, _ in per_file}
            siblings = [f for f in members.get(cid, ()) if f not in touched][:args.max_siblings]

            wrote_any = False
            for i in range(1, len(per_file)):
                path, h = per_file[i]
                if len(h["pre"]) > args.max_region_lines or len(h["post"]) > args.max_region_lines:
                    skipped_big += 1
                    continue
                # Keep only the most recent hunks, each line-capped. Unbounded history
                # produced prompts up to 16M chars; the target is what matters, and
                # recent edits carry nearly all the predictive signal.
                recent = per_file[max(0, i - args.max_prior_hunks):i]
                prior = []
                for p_path, p_h in recent:
                    body = p_h["raw"][1:]
                    if len(body) > args.max_prior_lines:
                        keep = args.max_prior_lines // 2
                        body = body[:keep] + [f"... [{len(body) - 2 * keep} lines elided] ..."] + body[-keep:]
                    prior.append(f"--- {p_path}")
                    prior.extend(body)

                prompt = build_prompt(prior, path, h["pre"], siblings)
                # Drop oldest history until the prompt fits the budget.
                while len(prompt) > args.max_prompt_chars and prior:
                    cut = next((k for k, ln in enumerate(prior[1:], 1) if ln.startswith("--- ")), len(prior))
                    prior = prior[cut:]
                    prompt = build_prompt(prior, path, h["pre"], siblings)
                if len(prompt) > args.max_prompt_chars:
                    skipped_big += 1
                    continue
                target = "\n".join(h["post"]) + "\n>>>>>>> UPDATED"
                row = json.dumps({
                    "text": prompt + "\n" + target,
                    "prompt": prompt,
                    "completion": target,
                    "repo": c["repo"], "sha": c["sha"], "cluster_id": cid, "path": path,
                }) + "\n"
                if cid in eval_clusters:
                    eval_fh.write(row)
                    n_eval += 1
                elif (c["repo"], path) in eval_files:
                    skipped_leak += 1          # file also edited on the eval side
                else:
                    fh.write(row)
                    n_written += 1
                    wrote_any = True
            n_commits += wrote_any
    if eval_fh:
        eval_fh.close()

    size = out_path.stat().st_size / 1e6
    print(f"\n{'='*60}\nNEXT-EDIT DATASET COMPLETE\n{'='*60}")
    print(f"Examples: {n_written:,} from {n_commits:,} commits")
    if eval_clusters:
        print(f"Eval split: {n_eval:,} examples from {len(eval_clusters)} held-out clusters "
              f"-> {eval_path}")
    print(f"Dropped -> incoherent(cross-cluster):{skipped_incoherent:,}  "
          f"no-hunks:{skipped_nohunks:,}  oversized-region:{skipped_big:,}  eval-file-leak:{skipped_leak:,}")
    print(f"Output: {out_path} ({size:.1f}MB)")


if __name__ == "__main__":
    main()
