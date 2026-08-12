"""Build a markdown corpus from the workspace for the agentic training mix.

Raw counts are misleading: of 18,615 .md files on disk, 60% are exact duplicates and
~12,000 are third-party boilerplate (the BMAD framework installed into many repos, plus
agent skill/plugin definitions). Training on those teaches someone else's templates.
`~/obsidian-vault` is excluded entirely -- it is a symlink farm over the same repos, so
including it would double-count.

Secret handling is the reason this file is careful. The satsure epic BRIEF.md documents
embed an SSH key path, a production server IP and JIRA_API_TOKEN usage in *prose*, which
filename-based .env filtering does not catch. Anything reaching training data is memorised
into the weights permanently and cannot be removed without retraining, so:

  - files containing a private key block are DROPPED outright
  - inline credentials are REDACTED in place
  - the run hard-fails if anything still matches after redaction

Usage:
    python src/format/format_markdown_corpus.py [--min-chars 200] [--max-chars 60000]
"""

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).parent.parent.parent
OUTPUT = ROOT / "data" / "formatted" / "markdown_corpus.jsonl"

SEARCH_ROOTS = [Path.home() / "workspace", Path("/mnt/j")]

# Build artefacts, vendored code, VCS internals.
SKIP_PATH = re.compile(
    r"(^|/)(node_modules|\.git|dist|build|out|target|vendor|\.next|\.nuxt|__pycache__|"
    r"\.venv|venv|coverage|site-packages|\.pnpm|\.cache|\.turbo)/", re.I)

# Third-party content that happens to live in the tree. Not the user's writing.
SKIP_BOILERPLATE = re.compile(
    r"(^|/)(_bmad|\.bmad)(/|$)|"                      # BMAD framework (installed)
    r"(^|/)\.(claude|opencode|codex)/(skills|plugins|agents|commands)/|"
    r"(^|/)node_modules/", re.I)

# Whole file is unsafe -- do not attempt to salvage.
FATAL_SECRET = re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")

# Redact in place. Each pattern keeps a short prefix so the text still reads sensibly.
REDACTIONS = [
    (re.compile(r"\b(sk-[A-Za-z0-9]{16,})"), "sk-REDACTED"),
    (re.compile(r"\b(ghp_[A-Za-z0-9]{20,})"), "ghp_REDACTED"),
    (re.compile(r"\b(gho_[A-Za-z0-9]{20,})"), "gho_REDACTED"),
    (re.compile(r"\b(glpat-[A-Za-z0-9_-]{16,})"), "glpat-REDACTED"),
    (re.compile(r"\b(rpa_[A-Za-z0-9]{20,})"), "rpa_REDACTED"),
    (re.compile(r"\b(AKIA[0-9A-Z]{16})"), "AKIA_REDACTED"),
    (re.compile(r"\b(AIza[0-9A-Za-z_-]{35})"), "AIza_REDACTED"),
    (re.compile(r"\b(xox[baprs]-[A-Za-z0-9-]{10,})"), "xoxb-REDACTED"),
    (re.compile(r"\bey[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"), "JWT_REDACTED"),
    # KEY=value / "token": "value" style assignments
    (re.compile(r"(?i)\b([A-Z0-9_]*(?:TOKEN|SECRET|PASSWORD|PASSWD|APIKEY|API_KEY)[A-Z0-9_]*)\s*[=:]\s*['\"]?([^\s'\"&#]{8,})"),
     r"\1=REDACTED"),
    # credentials embedded in a URL
    (re.compile(r"(https?://)([^/\s:@]+):([^/\s@]+)@"), r"\1REDACTED:REDACTED@"),
    # private key material referenced by path
    (re.compile(r"[\w./~-]+\.pem\b"), "REDACTED.pem"),
]

# Anything still matching after redaction fails the build.
VERIFY = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----|\bsk-[A-Za-z0-9]{16,}|\bghp_[A-Za-z0-9]{20,}|"
    r"\bAKIA[0-9A-Z]{16}|\bAIza[0-9A-Za-z_-]{35}|\bglpat-[A-Za-z0-9_-]{16,}|"
    r"\brpa_[A-Za-z0-9]{20,}|\bxox[baprs]-[A-Za-z0-9-]{10,}")


def scrub(text):
    """Redact inline credentials. Returns (text, n_redactions)."""
    n = 0
    for pat, repl in REDACTIONS:
        text, k = pat.subn(repl, text)
        n += k
    return text, n


def main():
    ap = argparse.ArgumentParser(description="Build markdown corpus for the agentic mix")
    ap.add_argument("--out", default=str(OUTPUT))
    ap.add_argument("--roots", nargs="+", default=[str(p) for p in SEARCH_ROOTS])
    ap.add_argument("--min-chars", type=int, default=200,
                    help="skip stubs; 388 files on disk are under 200B")
    ap.add_argument("--max-chars", type=int, default=60000,
                    help="split longer documents into chunks of this size")
    ap.add_argument("--max-tokens", type=float, default=None,
                    help="cap the corpus at roughly this many million tokens. Documents are "
                         "sampled whole (seeded) so continuation chunks are never orphaned. "
                         "Raw documents much above ~20%% of an SFT mix start to erode "
                         "instruction-following and tool-call formatting.")
    args = ap.parse_args()

    files = []
    for root in args.roots:
        root = Path(root)
        if not root.exists():
            continue
        for p in root.rglob("*.md"):
            sp = str(p)
            if SKIP_PATH.search(sp) or SKIP_BOILERPLATE.search(sp):
                continue
            if p.is_symlink():          # obsidian-vault mirrors the same content
                continue
            files.append(p)
    print(f"Candidate markdown files: {len(files):,}")

    seen = set()
    kept = []
    stats = Counter()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_chars = 0
    leaked = []

    with open(out_path, "w") as fh:
        for p in tqdm(files, desc="Markdown"):
            try:
                raw = p.read_text(errors="replace")
            except OSError:
                stats["unreadable"] += 1
                continue
            if len(raw) < args.min_chars:
                stats["too_short"] += 1
                continue
            if FATAL_SECRET.search(raw):
                stats["dropped_private_key"] += 1
                continue

            text, n_red = scrub(raw)
            if n_red:
                stats["files_redacted"] += 1
                stats["redactions"] += n_red
            if VERIFY.search(text):
                leaked.append(str(p))
                continue

            h = hashlib.md5(text.encode()).hexdigest()
            if h in seen:
                stats["duplicate"] += 1
                continue
            seen.add(h)
            kept.append((str(p), text))

        # Sample whole documents when a budget is set, so a document's continuation
        # chunks are never separated from its head.
        if args.max_tokens:
            import random
            random.Random(42).shuffle(kept)
            budget = args.max_tokens * 1e6 * 3.6      # ~3.6 chars/token
            chosen, used = [], 0.0
            for rel, text in kept:
                if used + len(text) > budget:
                    continue
                chosen.append((rel, text))
                used += len(text)
            stats["dropped_over_budget"] = len(kept) - len(chosen)
            kept = chosen

        for rel, text in kept:
            for i in range(0, len(text), args.max_chars):
                chunk = text[i:i + args.max_chars]
                if len(chunk) < args.min_chars:
                    continue
                header = f"# Document: {rel}\n\n" if i == 0 else f"# Document (cont.): {rel}\n\n"
                fh.write(json.dumps({"text": header + chunk, "source_path": rel}) + "\n")
                stats["chunks"] += 1
                total_chars += len(chunk)
            stats["files_kept"] += 1

    print(f"\n{'='*60}\nMARKDOWN CORPUS COMPLETE\n{'='*60}")
    print(f"Files kept   : {stats['files_kept']:,}   chunks: {stats['chunks']:,}")
    print(f"Dropped      -> duplicate:{stats['duplicate']:,}  too_short:{stats['too_short']:,}  "
          f"private_key:{stats['dropped_private_key']:,}  unreadable:{stats['unreadable']:,}")
    print(f"Redacted     -> {stats['files_redacted']:,} files, {stats['redactions']:,} secrets")
    print(f"Size         : {total_chars/1e6:.1f}MB text (~{total_chars/3.6/1e6:.1f}M tokens)")
    print(f"Output       : {out_path}")

    if leaked:
        print(f"\n!!! BUILD FAILED: {len(leaked)} files still matched a secret after redaction:")
        for f in leaked[:10]:
            print(f"    {f}")
        raise SystemExit(1)
    print("\nSecret verification: CLEAN")


if __name__ == "__main__":
    main()
