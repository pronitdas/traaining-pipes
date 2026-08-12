"""Extract Fill-in-the-Middle (FIM) training data from code files
referenced in conversations + your actual codebase.

FIM format for Qwen/Qwen2.5-Coder style autocomplete:
  <fim_prefix>code before cursor</fim_prefix>
  <fim_suffix>code after cursor</fim_suffix>
  <fim_middle>code at cursor</fim_middle>

Also produces StarCoder-style FIM format for broader compatibility.
"""

import json
import os
import random
from pathlib import Path
from tqdm import tqdm
import jsonlines
from collections import defaultdict

DATA_DIR = Path(__file__).parent.parent.parent.parent.parent / "data"
OUTPUT_DIR = DATA_DIR / "formatted"

# Source codebases to mine for FIM data
CODEBASE_PATHS = [
    "/home/pronit/workspace/tardis",
    "/home/pronit/workspace/invarya",
    "/home/pronit/workspace/satsure",
]

# File extensions to include
INCLUDE_EXTENSIONS = {
    ".ts", ".tsx", ".js", ".jsx", ".svelte", ".vue",
    ".py", ".rs", ".go", ".gd", ".zig",
    ".css", ".scss", ".html",
    ".json", ".yaml", ".yml", ".toml", ".sql",
    ".md",  # markdown for doc autocomplete
}

EXCLUDE_DIRS = {
    "node_modules", ".git", "dist", "build", "__pycache__",
    ".next", ".nuxt", "target", ".cache", ".venv", "venv",
    "*.lock", "vendor", ".turbo", ".codegraph", "coverage",
    ".svelte-kit", ".output",
}

EXCLUDE_PATTERNS = {
    ".min.js", ".min.css", ".map", ".lock",
    "package-lock", "bun.lock", "go.sum",
}

MAX_FILE_LINES = 500
MIN_FILE_LINES = 5
MAX_FIM_EXAMPLES_PER_FILE = 5
MAX_LINE_LENGTH = 200


def should_include_file(filepath):
    ext = Path(filepath).suffix.lower()
    if ext not in INCLUDE_EXTENSIONS:
        return False
    
    parts = filepath.parts
    for part in parts:
        if any(exc in part for exc in EXCLUDE_DIRS):
            return False
    
    for pattern in EXCLUDE_PATTERNS:
        if pattern in filepath.name:
            return False
    
    return True


def find_code_files():
    """Find all code files in configured codebases."""
    all_files = []
    for base_path in CODEBASE_PATHS:
        base = Path(base_path)
        if not base.exists():
            continue
        for filepath in base.rglob("*"):
            if filepath.is_file() and should_include_file(filepath):
                all_files.append(filepath)
    return all_files


def create_fim_examples(filepath, max_examples=MAX_FIM_EXAMPLES_PER_FILE):
    """Create FIM examples from a single file by splitting at random points."""
    examples = []
    
    try:
        content = filepath.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    
    lines = content.split("\n")
    if len(lines) < MIN_FILE_LINES or len(lines) > MAX_FILE_LINES:
        return []
    
    # Skip files with very long lines (probably minified)
    if any(len(line) > MAX_LINE_LENGTH for line in lines):
        return []
    
    # Generate FIM examples at different split points
    available = len(lines) - 6
    if available <= 0:
        return []
    num_splits = min(max_examples, max(1, available))
    split_points = sorted(random.sample(range(5, len(lines) - 1), min(num_splits, available)))
    
    for split_point in split_points:
        prefix = "\n".join(lines[:split_point])
        middle = lines[split_point] if split_point < len(lines) else ""
        suffix = "\n".join(lines[split_point + 1:]) if split_point + 1 < len(lines) else ""
        
        # Skip if middle is empty or just whitespace
        if not middle.strip():
            continue
        
        # Skip very short middles
        if len(middle.strip()) < 3:
            continue
        
        examples.append({
            "prefix": prefix,
            "middle": middle,
            "suffix": suffix,
            "language": filepath.suffix.lstrip("."),
            "filepath": str(filepath),
        })
    
    return examples


def format_fim_qwen(example):
    """Format as Qwen2.5-Coder FIM format."""
    prefix = example["prefix"]
    middle = example["middle"]
    suffix = example["suffix"]
    return {
        "text": f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>{middle}<|fim_hole|><|endoftext|>",
        "language": example["language"],
    }


def format_fim_starcoder(example):
    """Format as StarCoder FIM format."""
    prefix = example["prefix"]
    middle = example["middle"]
    suffix = example["suffix"]
    return {
        "text": f"<fim_prefix>{prefix}<fim_suffix>{suffix}<fim_middle>{middle}",
        "language": example["language"],
    }


def main():
    random.seed(42)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Finding code files...")
    code_files = find_code_files()
    print(f"Found {len(code_files):,} code files")
    
    # Language distribution
    lang_counts = defaultdict(int)
    for f in code_files:
        lang_counts[f.suffix.lstrip(".")] += 1
    print("\nLanguage distribution:")
    for lang, count in sorted(lang_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"  {count:6d}  .{lang}")
    
    print("\nExtracting FIM examples...")
    all_examples = []
    for filepath in tqdm(code_files, desc="FIM extraction"):
        examples = create_fim_examples(filepath)
        all_examples.extend(examples)
    
    print(f"\nTotal FIM examples: {len(all_examples):,}")
    
    # Format in both Qwen and StarCoder formats
    random.shuffle(all_examples)
    
    # Qwen format
    qwen_examples = [format_fim_qwen(ex) for ex in tqdm(all_examples, desc="Formatting Qwen")]
    qwen_file = OUTPUT_DIR / "fim_qwen.jsonl"
    with jsonlines.open(qwen_file, "w") as writer:
        for ex in qwen_examples:
            writer.write(ex)
    print(f"Qwen FIM: {len(qwen_examples):,} -> {qwen_file}")
    
    # StarCoder format
    starcoder_examples = [format_fim_starcoder(ex) for ex in all_examples]
    starcoder_file = OUTPUT_DIR / "fim_starcoder.jsonl"
    with jsonlines.open(starcoder_file, "w") as writer:
        for ex in starcoder_examples:
            writer.write(ex)
    print(f"StarCoder FIM: {len(starcoder_examples):,} -> {starcoder_file}")
    
    # Also create a combined dataset with code from conversations (tool outputs that contain code)
    print("\nExtracting code from conversation tool results...")
    conv_file = DATA_DIR / "raw" / "all_conversations.jsonl"
    conv_code_examples = []
    
    if conv_file.exists():
        with jsonlines.open(conv_file) as reader:
            for conv in tqdm(reader, desc="Conversations"):
                for turn in conv.get("turns", []):
                    if turn["role"] == "tool_result":
                        content = turn.get("content", "")
                        if isinstance(content, str) and len(content) > 50:
                            # Check if it looks like code output
                            lines = content.split("\n")
                            if len(lines) > 3:
                                # Create FIM from tool output
                                split = len(lines) // 2
                                prefix = "\n".join(lines[:split])
                                middle = lines[split] if split < len(lines) else ""
                                suffix = "\n".join(lines[split+1:]) if split + 1 < len(lines) else ""
                                if middle.strip():
                                    conv_code_examples.append({
                                        "prefix": prefix,
                                        "middle": middle,
                                        "suffix": suffix,
                                        "language": "mixed",
                                        "filepath": "conversation",
                                    })
    
    print(f"Code from conversations: {len(conv_code_examples):,}")
    
    # Add conversation code to Qwen FIM dataset
    conv_qwen = [format_fim_qwen(ex) for ex in conv_code_examples]
    combined_file = OUTPUT_DIR / "fim_combined.jsonl"
    with jsonlines.open(combined_file, "w") as writer:
        for ex in qwen_examples + conv_qwen:
            writer.write(ex)
    print(f"Combined FIM: {len(qwen_examples) + len(conv_qwen):,} -> {combined_file}")


if __name__ == "__main__":
    main()
