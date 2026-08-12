"""Extract conversations from opencode.db (both legacy and v2 schemas)."""

import sqlite3
import json
import os
from pathlib import Path
from tqdm import tqdm
import jsonlines

DB_PATH = os.path.expanduser("~/.local/share/opencode/opencode.db")
OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "raw" / "opencode"


def extract_legacy_messages(conn):
    """Extract from legacy message + part tables (older opencode format)."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT m.session_id, m.id, m.time_created, m.data,
               p.id, p.data, p.time_created
        FROM message m
        LEFT JOIN part p ON p.message_id = m.id
        ORDER BY m.session_id, m.time_created, p.time_created
    """)
    
    sessions = {}
    for row in cursor.fetchall():
        session_id = row[0]
        if session_id not in sessions:
            sessions[session_id] = {"messages": []}
        
        msg_data = json.loads(row[3])
        if row[5]:
            part_data = json.loads(row[5])
            msg_data.setdefault("parts", []).append(part_data)
        
        if not sessions[session_id]["messages"] or sessions[session_id]["messages"][-1].get("id") != row[1]:
            sessions[session_id]["messages"].append({"id": row[1], "data": msg_data})
        else:
            sessions[session_id]["messages"][-1]["data"].setdefault("parts", []).append(part_data)
    
    return sessions


def extract_v2_messages(conn):
    """Extract from session_v2 + session_message tables (newer opencode format)."""
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, title, model, agent, project_id, directory FROM session_v2")
    sessions_meta = {}
    for row in cursor.fetchall():
        sessions_meta[row[0]] = {
            "title": row[1],
            "model": row[2],
            "agent": row[3],
            "project_id": row[4],
            "directory": row[5],
            "messages": []
        }
    
    cursor.execute("""
        SELECT session_id, id, type, seq, data, time_created
        FROM session_message
        ORDER BY session_id, seq
    """)
    
    for row in cursor.fetchall():
        sid = row[0]
        if sid not in sessions_meta:
            continue
        data = json.loads(row[4])
        sessions_meta[sid]["messages"].append({
            "id": row[1],
            "type": row[2],
            "seq": row[3],
            "data": data,
            "time_created": row[5]
        })
    
    return sessions_meta


def normalize_to_conversations(sessions):
    """Convert extracted sessions to a unified conversation format."""
    conversations = []
    
    for sid, session in sessions.items():
        if not session.get("messages"):
            continue
        
        turns = []
        for msg in session["messages"]:
            data = msg.get("data", {})
            msg_type = msg.get("type", data.get("role", ""))
            
            if msg_type == "user":
                text = data.get("text", "")
                if text:
                    turns.append({"role": "user", "content": text})
            elif msg_type == "assistant":
                content_parts = []
                for c in data.get("content", []):
                    if c.get("type") == "text":
                        content_parts.append({"type": "text", "text": c["text"]})
                    elif c.get("type") == "tool":
                        state = c.get("state", {})
                        tool_call = {
                            "type": "tool_call",
                            "name": c.get("name"),
                            "input": state.get("input", {}),
                            "output": None,
                            "status": state.get("status")
                        }
                        if state.get("content"):
                            tool_output_parts = []
                            for tc in state["content"]:
                                if isinstance(tc, dict) and tc.get("type") == "text":
                                    tool_output_parts.append(tc["text"])
                            tool_call["output"] = "\n".join(tool_output_parts)
                        content_parts.append(tool_call)
                if content_parts:
                    turns.append({"role": "assistant", "content": content_parts})
            elif msg_type == "summary":
                turns.append({"role": "summary", "content": data})
        
        if len(turns) >= 2:
            conversations.append({
                "session_id": sid,
                "title": session.get("title", ""),
                "model": session.get("model"),
                "agent": session.get("agent"),
                "directory": session.get("directory"),
                "turns": turns
            })
    
    return conversations


def extract_legacy_normalized(conn):
    """Extract and normalize legacy format messages."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT m.session_id, m.id, m.time_created, m.data
        FROM message m
        ORDER BY m.session_id, m.time_created
    """)
    
    sessions = {}
    for row in cursor.fetchall():
        sid = row[0]
        if sid not in sessions:
            sessions[sid] = []
        data = json.loads(row[3])
        sessions[sid].append({"id": row[1], "data": data, "time_created": row[2]})
    
    cursor.execute("""
        SELECT p.session_id, p.message_id, p.data, p.time_created
        FROM part p
        ORDER BY p.message_id, p.time_created
    """)
    
    parts_by_msg = {}
    for row in cursor.fetchall():
        mid = row[1]
        parts_by_msg.setdefault(mid, []).append(json.loads(row[2]))
    
    conversations = []
    for sid, msgs in sessions.items():
        turns = []
        for msg in msgs:
            data = msg["data"]
            role = data.get("role", "")
            msg_parts = parts_by_msg.get(msg["id"], [])
            
            if role == "user":
                text_parts = [p for p in msg_parts if p.get("type") == "text"]
                text = text_parts[0]["text"] if text_parts else ""
                if text:
                    turns.append({"role": "user", "content": text})
            elif role == "assistant":
                content_parts = []
                for p in msg_parts:
                    if p.get("type") == "text":
                        content_parts.append({"type": "text", "text": p["text"]})
                    elif p.get("type") == "tool":
                        state = p.get("state", {})
                        content_parts.append({
                            "type": "tool_call",
                            "name": p.get("tool"),
                            "input": state.get("input", {}),
                            "output": state.get("output", ""),
                            "status": state.get("status")
                        })
                if content_parts:
                    turns.append({"role": "assistant", "content": content_parts})
        
        if len(turns) >= 2:
            conversations.append({
                "session_id": sid,
                "turns": turns,
                "format": "legacy"
            })
    
    return conversations


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    
    print("Extracting v2 sessions...")
    v2_sessions = extract_v2_messages(conn)
    v2_conversations = normalize_to_conversations(v2_sessions)
    print(f"  {len(v2_conversations)} conversations from v2 schema")
    
    print("Extracting legacy sessions...")
    legacy_conversations = extract_legacy_normalized(conn)
    print(f"  {len(legacy_conversations)} conversations from legacy schema")
    
    all_conversations = v2_conversations + legacy_conversations
    
    output_file = OUTPUT_DIR / "conversations.jsonl"
    with jsonlines.open(output_file, "w") as writer:
        for conv in tqdm(all_conversations, desc="Writing"):
            writer.write(conv)
    
    print(f"\nTotal: {len(all_conversations)} conversations")
    print(f"Output: {output_file}")
    
    total_turns = sum(len(c["turns"]) for c in all_conversations)
    total_tool_calls = sum(
        1 for c in all_conversations
        for t in c["turns"]
        if t["role"] == "assistant" and isinstance(t["content"], list)
        for p in t["content"]
        if isinstance(p, dict) and p.get("type") == "tool_call"
    )
    print(f"Total turns: {total_turns}")
    print(f"Total tool calls: {total_tool_calls}")
    
    conn.close()


if __name__ == "__main__":
    main()
