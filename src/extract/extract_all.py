"""Extract all conversations from all sources and merge into a unified dataset."""

import json
import os
from pathlib import Path
import jsonlines
from tqdm import tqdm

# Import extractors
import sys
sys.path.insert(0, str(Path(__file__).parent))
from extract_opencode import main as extract_opencode
from extract_claude import main as extract_claude

DATA_DIR = Path(__file__).parent.parent.parent / "data"
MERGED_OUTPUT = DATA_DIR / "raw" / "all_conversations.jsonl"


def merge_sources():
    """Merge all extracted conversations into one file.

    Streams source->sink. The /mnt/j corpus alone is ~627MB of JSONL, which becomes
    several GB once parsed into Python objects, so conversations are never accumulated
    in a list.
    """
    sources = [
        DATA_DIR / "raw" / "opencode_j" / "conversations.jsonl",   # 76GB /mnt/j DB
        DATA_DIR / "raw" / "opencode" / "conversations.jsonl",
        DATA_DIR / "raw" / "claude" / "conversations.jsonl",
        DATA_DIR / "raw" / "codex" / "conversations.jsonl",
    ]

    from collections import Counter
    source_counts = Counter()
    total = 0
    total_turns = 0

    MERGED_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(MERGED_OUTPUT, "w") as writer:
        for src in sources:
            if not src.exists():
                print(f"  [skip] {src} (not extracted yet)")
                continue
            n = 0
            with jsonlines.open(src) as reader:
                for conv in tqdm(reader, desc=f"Merging {src.parent.name}"):
                    writer.write(conv)
                    source_counts[conv.get("source", conv.get("format", src.parent.name))] += 1
                    total_turns += len(conv.get("turns", ()))
                    n += 1
            total += n
            print(f"  {src.parent.name}: {n:,} conversations")

    print(f"\n=== MERGED DATASET ===")
    print(f"Total conversations: {total:,}")
    print(f"By source: {dict(source_counts)}")
    print(f"Total turns: {total_turns:,}")
    print(f"Output: {MERGED_OUTPUT} ({MERGED_OUTPUT.stat().st_size/1e6:.1f}MB)")


def main():
    print("=" * 60)
    print("EXTRACTING FROM OPENCODE.DB")
    print("=" * 60)
    extract_opencode()
    
    print()
    print("=" * 60)
    print("EXTRACTING FROM CLAUDE")
    print("=" * 60)
    extract_claude()
    
    print()
    print("=" * 60)
    print("MERGING ALL SOURCES")
    print("=" * 60)
    merge_sources()


if __name__ == "__main__":
    main()
