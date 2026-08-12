"""Extract conversations from Claude Code transcripts and project files."""

import json
import glob
import os
from pathlib import Path
from tqdm import tqdm
import jsonlines
from collections import defaultdict

CLAUDE_TRANSCRIPTS_DIR = os.path.expanduser("~/.claude/transcripts")
CLAUDE_PROJECTS_DIR = os.path.expanduser("~/.claude/projects")
OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "raw" / "claude"

# Tool name mapping: Claude Code tool names → unified names
TOOL_NAME_MAP = {
    "Bash": "bash",
    "Read": "read",
    "Write": "write",
    "Edit": "edit",
    "Grep": "grep",
    "Glob": "glob",
    "Skill": "skill",
    "WebFetch": "webfetch",
    "WebSearch": "websearch",
    "TodoWrite": "todowrite",
    "Task": "task",
    "MultiEdit": "multi_edit",
    "NotebookEdit": "notebook_edit",
}


def extract_from_transcript(filepath):
    """Extract conversation from a single Claude transcript JSONL file."""
    turns = []
    with open(filepath) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            msg_type = d.get("type")
            content = d.get("content")
            
            if msg_type == "user":
                if isinstance(content, str):
                    turns.append({"role": "user", "content": content})
                elif isinstance(content, list):
                    text_parts = []
                    for c in content:
                        if isinstance(c, dict) and c.get("type") == "text":
                            text_parts.append(c["text"])
                        elif isinstance(c, dict) and c.get("type") == "tool_result":
                            tool_result_text = ""
                            rc = c.get("content", "")
                            if isinstance(rc, str):
                                tool_result_text = rc
                            elif isinstance(rc, list):
                                tool_result_text = "\n".join(
                                    r.get("text", "") for r in rc if isinstance(r, dict)
                                )
                            turns.append({
                                "role": "tool_result",
                                "tool_use_id": c.get("tool_use_id", ""),
                                "content": tool_result_text
                            })
                    if text_parts:
                        turns.append({"role": "user", "content": "\n".join(text_parts)})
            elif msg_type == "assistant":
                if isinstance(content, list):
                    content_parts = []
                    for c in content:
                        if not isinstance(c, dict):
                            continue
                        ct = c.get("type")
                        if ct == "text":
                            content_parts.append({"type": "text", "text": c["text"]})
                        elif ct == "thinking":
                            content_parts.append({
                                "type": "thinking",
                                "text": c.get("thinking", "")
                            })
                        elif ct == "tool_use":
                            tool_name = TOOL_NAME_MAP.get(c.get("name", ""), c.get("name", "").lower())
                            content_parts.append({
                                "type": "tool_call",
                                "name": tool_name,
                                "input": c.get("input", {}),
                                "tool_use_id": c.get("id", "")
                            })
                    if content_parts:
                        turns.append({"role": "assistant", "content": content_parts})
    
    return turns


def extract_from_project_file(filepath):
    """Extract conversation from a Claude project JSONL file (richer format)."""
    turns = []
    cwd = None
    session_id = None
    
    with open(filepath) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            if d.get("type") == "user":
                msg = d.get("message", {})
                content = msg.get("content", "")
                cwd = d.get("cwd", cwd)
                session_id = d.get("sessionId", session_id)
                
                if isinstance(content, str):
                    turns.append({"role": "user", "content": content})
                elif isinstance(content, list):
                    text_parts = []
                    for c in content:
                        if isinstance(c, dict):
                            if c.get("type") == "text":
                                text_parts.append(c["text"])
                            elif c.get("type") == "tool_result":
                                tool_result_text = ""
                                rc = c.get("content", "")
                                if isinstance(rc, str):
                                    tool_result_text = rc
                                elif isinstance(rc, list):
                                    tool_result_text = "\n".join(
                                        r.get("text", "") for r in rc if isinstance(r, dict)
                                    )
                                turns.append({
                                    "role": "tool_result",
                                    "tool_use_id": c.get("tool_use_id", ""),
                                    "content": tool_result_text
                                })
                    if text_parts:
                        turns.append({"role": "user", "content": "\n".join(text_parts)})
            
            elif d.get("type") == "assistant":
                msg = d.get("message", {})
                content = msg.get("content", [])
                model = msg.get("model", "")
                
                if isinstance(content, list):
                    content_parts = []
                    for c in content:
                        if not isinstance(c, dict):
                            continue
                        ct = c.get("type")
                        if ct == "text":
                            content_parts.append({"type": "text", "text": c["text"]})
                        elif ct == "thinking":
                            content_parts.append({
                                "type": "thinking",
                                "text": c.get("thinking", "")
                            })
                        elif ct == "tool_use":
                            tool_name = TOOL_NAME_MAP.get(c.get("name", ""), c.get("name", "").lower())
                            content_parts.append({
                                "type": "tool_call",
                                "name": tool_name,
                                "input": c.get("input", {}),
                                "tool_use_id": c.get("id", "")
                            })
                    if content_parts:
                        turn = {"role": "assistant", "content": content_parts}
                        if model:
                            turn["model"] = model
                        turns.append(turn)
    
    return turns, cwd, session_id


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Extract from project files (richer, has thinking blocks)
    print("Extracting from Claude project files...")
    project_files = sorted(glob.glob(f"{CLAUDE_PROJECTS_DIR}/**/*.jsonl", recursive=True))
    conversations = []
    
    for fpath in tqdm(project_files, desc="Project files"):
        try:
            turns, cwd, sid = extract_from_project_file(fpath)
            if len(turns) >= 4:
                conversations.append({
                    "session_id": sid or os.path.basename(fpath).replace(".jsonl", ""),
                    "source": "claude_project",
                    "directory": cwd,
                    "turns": turns
                })
        except Exception as e:
            tqdm.write(f"Error in {fpath}: {e}")
    
    print(f"  {len(conversations)} conversations from project files")
    
    # Extract from transcript files (lighter format, more numerous)
    print("Extracting from Claude transcript files...")
    transcript_files = sorted(glob.glob(f"{CLAUDE_TRANSCRIPTS_DIR}/*.jsonl"))
    transcript_conversations = []
    
    for fpath in tqdm(transcript_files, desc="Transcripts"):
        try:
            turns = extract_from_transcript(fpath)
            if len(turns) >= 4:
                transcript_conversations.append({
                    "session_id": os.path.basename(fpath).replace(".jsonl", ""),
                    "source": "claude_transcript",
                    "turns": turns
                })
        except Exception as e:
            tqdm.write(f"Error in {fpath}: {e}")
    
    print(f"  {len(transcript_conversations)} conversations from transcripts")
    
    # Deduplicate: some transcripts overlap with project files
    existing_sids = {c["session_id"] for c in conversations}
    deduped_transcripts = [
        c for c in transcript_conversations
        if c["session_id"] not in existing_sids
    ]
    
    all_conversations = conversations + deduped_transcripts
    
    output_file = OUTPUT_DIR / "conversations.jsonl"
    with jsonlines.open(output_file, "w") as writer:
        for conv in tqdm(all_conversations, desc="Writing"):
            writer.write(conv)
    
    print(f"\nTotal: {len(all_conversations)} conversations")
    print(f"  Project files: {len(conversations)}")
    print(f"  Transcripts (deduped): {len(deduped_transcripts)}")
    print(f"Output: {output_file}")
    
    total_turns = sum(len(c["turns"]) for c in all_conversations)
    total_tool_calls = sum(
        1 for c in all_conversations
        for t in c["turns"]
        if t["role"] == "assistant" and isinstance(t["content"], list)
        for p in t["content"]
        if isinstance(p, dict) and p.get("type") == "tool_call"
    )
    print(f"Total turns: {total_turns:,}")
    print(f"Total tool calls: {total_tool_calls:,}")
    
    total_thinking = sum(
        1 for c in all_conversations
        for t in c["turns"]
        if t["role"] == "assistant" and isinstance(t["content"], list)
        for p in t["content"]
        if isinstance(p, dict) and p.get("type") == "thinking"
    )
    print(f"Total thinking blocks: {total_thinking:,}")


if __name__ == "__main__":
    main()
