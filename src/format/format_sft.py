"""Format extracted conversations into SFT training data for Qwen3-36B-A3B.

Produces two datasets:
1. Agentic SFT -- tool-call conversations in Qwen chat format with tool schema
2. Instruction SFT -- plain text reasoning/planning/explanation turns (no tools)
"""

import json
import os
import re
import random
from pathlib import Path
from tqdm import tqdm
import jsonlines
from collections import defaultdict

DATA_DIR = Path(__file__).parent.parent.parent / "data"
INPUT_FILE = DATA_DIR / "raw" / "all_conversations.jsonl"
OUTPUT_DIR = DATA_DIR / "formatted"

SYSTEM_PROMPT_AGENTIC = "You are an expert coding assistant with access to tools. You help with software engineering tasks by reading code, running commands, editing files, and making decisions. You are precise, thorough, and follow existing code conventions. You use tools efficiently and explain your reasoning concisely."

SYSTEM_PROMPT_INSTRUCTION = "You are an expert software engineer and technical advisor. You provide clear, accurate, and practical guidance on architecture, code quality, debugging, and development workflows."

MAX_TOOL_OUTPUT = 2000
# Windows start at a user turn (see chunk_conversation), so this is really "how much of
# one request->action cycle do we keep". Measured over the corpus: cycle length is p50=2,
# p90=21, p99=93 turns. 20 truncated 10% of cycles -- and the long ones are precisely the
# long tool-call chains worth learning. 40 covers 95.8% of cycles.
MAX_CONTEXT_TURNS = 40
MIN_TURNS = 3
MAX_TURNS = 50

TOOL_OPEN = "\n[TOOL_CALL]"
TOOL_CLOSE = "[/TOOL_CALL]\n"


def truncate_output(text, max_chars=MAX_TOOL_OUTPUT):
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... [truncated, {len(text)} total chars]"


def format_tool_call(name, input_data):
    input_str = json.dumps(input_data, indent=2) if input_data else "{}"
    return f"{TOOL_OPEN}{name}\n{input_str}{TOOL_CLOSE}"


def format_tool_result(output, status="completed"):
    output = truncate_output(output)
    return f"[TOOL_RESULT]{output}[/TOOL_RESULT]"


def extract_turn_text(turn):
    content = turn.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict):
                if p.get("type") == "text":
                    parts.append(p["text"])
                elif p.get("type") == "thinking":
                    parts.append(f"[THINKING]{p.get('text','')}[/THINKING]")
        return "\n".join(parts)
    return ""


def has_tool_calls(turn):
    content = turn.get("content", [])
    if isinstance(content, list):
        return any(isinstance(p, dict) and p.get("type") == "tool_call" for p in content)
    return False


def has_tool_calls_in_str(text):
    return TOOL_OPEN in text


def build_agentic_example(turns, session_meta=None):
    # One system message only. Emitting a second one for the working directory left
    # 99.7% of examples with two, which some chat templates reject outright
    # ("System message must be at the beginning") and which no template expects.
    system = SYSTEM_PROMPT_AGENTIC
    if session_meta and session_meta.get("directory"):
        system += f"\n\nWorking directory: {session_meta['directory']}"
    messages = [{"role": "system", "content": system}]


    for turn in turns:
        role = turn["role"]
        content = turn.get("content", "")
        
        if role == "user":
            text = content if isinstance(content, str) else extract_turn_text(turn)
            if text.strip():
                messages.append({"role": "user", "content": text})
        
        elif role == "assistant":
            if isinstance(content, list):
                msg_parts = []
                for p in content:
                    if not isinstance(p, dict):
                        continue
                    ptype = p.get("type")
                    if ptype == "thinking":
                        thinking = p.get("text", "").strip()
                        if thinking:
                            msg_parts.append(f"[THINKING]{thinking}[/THINKING]")
                    elif ptype == "text":
                        msg_parts.append(p["text"])
                    elif ptype == "tool_call":
                        tool_str = format_tool_call(p.get("name"), p.get("input"))
                        msg_parts.append(tool_str)
                
                if msg_parts:
                    messages.append({
                        "role": "assistant",
                        "content": "\n".join(msg_parts)
                    })
            elif isinstance(content, str) and content.strip():
                messages.append({"role": "assistant", "content": content})
        
        elif role == "tool_result":
            output = content if isinstance(content, str) else str(content)
            messages.append({
                "role": "tool",
                "content": format_tool_result(output)
            })
    
    has_tools = any(
        m["role"] == "assistant" and TOOL_OPEN in m.get("content", "")
        for m in messages
    )
    
    if not has_tools or len(messages) < 4:
        return None
    
    return {"messages": messages}


def build_instruction_example(turns):
    messages = [{"role": "system", "content": SYSTEM_PROMPT_INSTRUCTION}]
    
    for turn in turns:
        role = turn["role"]
        content = turn.get("content", "")
        
        if role in ("user", "assistant"):
            if role == "assistant" and isinstance(content, list) and has_tool_calls(turn):
                continue
            
            text = extract_turn_text(turn)
            if text.strip() and TOOL_OPEN not in text:
                messages.append({"role": role, "content": text})
        elif role == "tool_result":
            continue
    
    if len(messages) < 4:
        return None
    
    has_user = any(m["role"] == "user" for m in messages[1:])
    has_assistant = any(m["role"] == "assistant" for m in messages[1:])
    
    if not has_user or not has_assistant:
        return None
    
    return {"messages": messages}


def chunk_conversation(turns, max_turns=MAX_CONTEXT_TURNS):
    """Split a long conversation into windows that each begin with a user turn.

    Slicing on a fixed stride made 53% of examples open with an assistant turn --
    the window landed mid-exchange. With a system prompt and no request in front of
    it, that teaches the model to start acting before being asked. Snapping each
    window back to the nearest preceding user turn keeps every example shaped like
    "request -> actions", which is what we want the model to learn.
    """
    user_idx = [i for i, t in enumerate(turns) if t.get("role") == "user"]
    if not user_idx:
        return []
    if len(turns) <= max_turns and user_idx[0] == 0:
        return [turns]

    # One window per request: from a user turn up to the next user turn, capped at
    # max_turns. A fixed stride was wrong twice over -- it opened 53% of windows
    # mid-exchange, and because the stride scales with the window size, widening the
    # window actually *reduced* coverage. Cycle boundaries give full coverage of the
    # conversation with no overlap and no duplicated content.
    chunks = []
    bounds = user_idx + [len(turns)]
    for start, nxt in zip(user_idx, bounds[1:]):
        chunk = turns[start:min(nxt, start + max_turns)]
        if len(chunk) >= MIN_TURNS:
            chunks.append(chunk)
    return chunks


def estimate_tokens(char_count):
    return char_count // 4


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    conversations = []
    with jsonlines.open(INPUT_FILE) as reader:
        for conv in reader:
            conversations.append(conv)
    
    print(f"Loaded {len(conversations)} conversations")
    
    agentic_examples = []
    instruction_examples = []
    
    for conv in tqdm(conversations, desc="Formatting"):
        turns = conv.get("turns", [])
        if len(turns) < MIN_TURNS:
            continue
        
        chunks = chunk_conversation(turns)
        
        for chunk in chunks:
            agentic = build_agentic_example(chunk, conv)
            if agentic:
                total_chars = sum(len(m["content"]) for m in agentic["messages"])
                agentic["token_estimate"] = estimate_tokens(total_chars)
                if agentic["token_estimate"] <= 8000:
                    agentic_examples.append(agentic)
            
            instruction = build_instruction_example(chunk)
            if instruction:
                total_chars = sum(len(m["content"]) for m in instruction["messages"])
                instruction["token_estimate"] = estimate_tokens(total_chars)
                if instruction["token_estimate"] <= 8000:
                    instruction_examples.append(instruction)
    
    random.seed(42)
    random.shuffle(agentic_examples)
    random.shuffle(instruction_examples)
    
    agentic_file = OUTPUT_DIR / "agentic_sft.jsonl"
    with jsonlines.open(agentic_file, "w") as writer:
        for ex in tqdm(agentic_examples, desc="Writing agentic"):
            writer.write(ex)
    
    instruction_file = OUTPUT_DIR / "instruction_sft.jsonl"
    with jsonlines.open(instruction_file, "w") as writer:
        for ex in tqdm(instruction_examples, desc="Writing instruction"):
            writer.write(ex)
    
    print(f"\n{'='*60}")
    print(f"FORMATTED DATASETS")
    print(f"{'='*60}")
    print(f"Agentic SFT:   {len(agentic_examples)} examples -> {agentic_file}")
    print(f"Instruction:    {len(instruction_examples)} examples -> {instruction_file}")
    
    agentic_tokens = sum(e["token_estimate"] for e in agentic_examples)
    instruction_tokens = sum(e["token_estimate"] for e in instruction_examples)
    print(f"\nEstimated tokens:")
    print(f"  Agentic:      {agentic_tokens:,} (~{agentic_tokens/1e6:.1f}M)")
    print(f"  Instruction:   {instruction_tokens:,} (~{instruction_tokens/1e6:.1f}M)")
    print(f"  Total:         {(agentic_tokens + instruction_tokens):,} (~{(agentic_tokens + instruction_tokens)/1e6:.1f}M)")
    
    tool_counts = defaultdict(int)
    tool_pattern = re.compile(re.escape(TOOL_OPEN) + r"(\w+)")
    for ex in agentic_examples:
        for m in ex["messages"]:
            if m["role"] == "assistant" and TOOL_OPEN in m.get("content", ""):
                for match in tool_pattern.finditer(m["content"]):
                    tool_counts[match.group(1)] += 1
    
    print(f"\nTool call distribution:")
    for tool, count in sorted(tool_counts.items(), key=lambda x: -x[1]):
        print(f"  {count:5d}  {tool}")
    
    avg_agentic = sum(len(e["messages"]) for e in agentic_examples) / len(agentic_examples) if agentic_examples else 0
    avg_instruction = sum(len(e["messages"]) for e in instruction_examples) / len(instruction_examples) if instruction_examples else 0
    print(f"\nAvg messages per example:")
    print(f"  Agentic:      {avg_agentic:.1f}")
    print(f"  Instruction:  {avg_instruction:.1f}")


if __name__ == "__main__":
    main()
