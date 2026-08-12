"""Mine git history into filtered commit records for co-change analysis.

Mechanical only -- no model is involved in deciding what is signal. The filters exist
because raw history is dominated by things that destroy co-change statistics:
formatting sweeps, lockfile churn, dependency bumps and merge commits all make unrelated
files appear to change together.

Only files reported by `git ls-files` are considered, which by construction excludes
node_modules, dist/build output and anything gitignored.

Usage:
    python src/mine/mine_commits.py --roots ~/workspace/tardis ~/workspace/invarya
"""

import argparse
import json
import re
import subprocess
from pathlib import Path

from tqdm import tqdm

OUTPUT = Path(__file__).parent.parent.parent / "data" / "mined" / "commits.jsonl"

# Conventional-commit types that carry no feature-coherence signal.
SKIP_TYPES = {"chore", "ci", "build", "style", "docs", "release", "revert", "test"}
CONVENTIONAL = re.compile(r"^(?P<type>[a-z]+)(?:\((?P<scope>[^)]*)\))?!?:")

# A commit touching more than this many files is a sweep, not a feature.
MAX_FILES = 25

SKIP_MESSAGE = re.compile(r"\[skip ci\]|^Merge |^Revert |bump version", re.I)

# Never train on these. Secrets are memorised permanently into weights.
SECRET_PATHS = re.compile(r"(^|/)\.env($|\.)|\.pem$|\.key$|\.p12$|id_rsa|credentials?\.json$", re.I)

SKIP_PATH = re.compile(
    r"(^|/)(node_modules|dist|build|out|target|vendor|\.next|\.nuxt|__pycache__|"
    r"\.venv|venv|coverage|\.git)/|"
    r"\.(lock|min\.js|min\.css|snap|map|png|jpe?g|gif|svg|ico|webp|pdf|zip|gz|tar|"
    r"woff2?|ttf|eot|mp4|mp3|wav|bin|so|dylib|dll|exe|pyc|class|jar|wasm)$|"
    r"(^|/)(package-lock\.json|pnpm-lock\.yaml|yarn\.lock|poetry\.lock|Cargo\.lock|"
    r"go\.sum|composer\.lock|Gemfile\.lock|uv\.lock)$",
    re.I,
)

CODE_EXT = re.compile(
    r"\.(py|js|jsx|ts|tsx|rs|go|java|kt|swift|c|h|cc|cpp|hpp|cs|rb|php|scala|sh|bash|"
    r"zsh|sql|gd|vue|svelte|lua|ex|exs|erl|hs|ml|r|jl|dart|toml|yaml|yml|json|md)$",
    re.I,
)


def run(cmd, cwd):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                          errors="replace", timeout=600).stdout


def find_repos(roots, max_depth=3):
    """Locate git repos under the given roots, without descending into node_modules."""
    found = []
    for root in roots:
        root = Path(root).expanduser()
        if not root.exists():
            continue
        if (root / ".git").is_dir():
            found.append(root)
        for depth in range(1, max_depth + 1):
            pattern = "/".join(["*"] * depth) + "/.git"
            for g in root.glob(pattern):
                if "node_modules" in g.parts or not g.is_dir():
                    continue
                found.append(g.parent)
    return sorted(set(found))


def tracked_files(repo):
    out = run(["git", "ls-files"], repo)
    return {line for line in out.splitlines() if line}


def keep_path(path, tracked):
    if path not in tracked:
        return False
    if SECRET_PATHS.search(path) or SKIP_PATH.search(path):
        return False
    return bool(CODE_EXT.search(path))


def parse_log(repo, tracked, stats):
    """Yield filtered commit records for one repo."""
    sep = "\x1f"   # unit separator; \x00 cannot be passed through argv
    fmt = sep.join(["%H", "%ct", "%s"])
    out = run(
        ["git", "log", "--no-merges", "--numstat", f"--format=__C__{fmt}", "--no-renames"],
        repo,
    )
    sha = ts = subject = None
    files = []

    def flush():
        nonlocal sha, files
        if sha is None:
            return None
        rec = None
        if files:
            m = CONVENTIONAL.match(subject or "")
            ctype = m.group("type") if m else None
            scope = m.group("scope") if m else None
            if ctype in SKIP_TYPES:
                stats["skip_type"] += 1
            elif SKIP_MESSAGE.search(subject or ""):
                stats["skip_msg"] += 1
            elif len(files) > MAX_FILES:
                stats["skip_large"] += 1
            else:
                rec = {
                    "repo": repo.name, "repo_path": str(repo), "sha": sha,
                    "ts": int(ts), "subject": subject,
                    "type": ctype, "scope": scope, "files": sorted(files),
                }
        else:
            stats["skip_nofiles"] += 1
        sha, files = None, []
        return rec

    for line in out.splitlines():
        if line.startswith("__C__"):
            rec = flush()
            if rec:
                yield rec
            sha, ts, subject = (line[5:].split(sep) + ["", "", ""])[:3]
            files = []
        elif line.strip() and sha is not None:
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            added, deleted, path = parts
            if added == "-" and deleted == "-":
                continue  # binary
            if keep_path(path, tracked):
                files.append(path)
    rec = flush()
    if rec:
        yield rec


def main():
    global MAX_FILES
    ap = argparse.ArgumentParser(description="Mine git history into filtered commit records")
    ap.add_argument("--roots", nargs="+",
                    default=[str(Path.home() / "workspace")])
    ap.add_argument("--out", default=str(OUTPUT))
    ap.add_argument("--max-files", type=int, default=MAX_FILES)
    args = ap.parse_args()
    MAX_FILES = args.max_files

    repos = find_repos(args.roots)
    print(f"Found {len(repos)} git repos")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {"skip_type": 0, "skip_msg": 0, "skip_large": 0, "skip_nofiles": 0,
             "skip_dup_checkout": 0}
    kept = 0
    per_repo = {}
    # Deduplicate on the sha alone, not (repo, sha). A git sha hashes content plus
    # history, so an identical sha IS an identical commit -- it cannot collide across
    # genuinely different projects. The same work shows up on disk under several names
    # (a copy nested in a tooling directory, a renamed clone), and mining each one
    # double-counts every commit: it inflates co-change pair counts and puts byte-identical
    # examples on both sides of a train/eval split.
    seen_sha = set()

    with open(out_path, "w") as fh:
        for repo in tqdm(repos, desc="Repos"):
            try:
                tracked = tracked_files(repo)
            except (subprocess.TimeoutExpired, OSError):
                continue
            if not tracked:
                continue
            n = 0
            try:
                for rec in parse_log(repo, tracked, stats):
                    key = rec["sha"]
                    if key in seen_sha:
                        stats["skip_dup_checkout"] += 1
                        continue
                    seen_sha.add(key)
                    fh.write(json.dumps(rec) + "\n")
                    kept += 1
                    n += 1
            except (subprocess.TimeoutExpired, OSError):
                continue
            if n:
                per_repo[repo.name] = n

    print(f"\n{'='*60}\nCOMMIT MINING COMPLETE\n{'='*60}")
    print(f"Kept commits: {kept:,}")
    print(f"Dropped -> type:{stats['skip_type']:,}  message:{stats['skip_msg']:,}  "
          f">{MAX_FILES}files:{stats['skip_large']:,}  no-code-files:{stats['skip_nofiles']:,}  "
          f"dup-checkout:{stats['skip_dup_checkout']:,}")
    print(f"Top repos: {sorted(per_repo.items(), key=lambda x: -x[1])[:10]}")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
