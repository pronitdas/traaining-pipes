"""Report how a model actually writes tool calls, by rendering its own chat template.

Adding a base model to the matrix means knowing three things that differ per family and are
not guessable: what delimits a tool call, how a tool RESULT is carried, and whether those
delimiters are single special tokens (which decides skip_special_tokens at eval time).

Rather than trust a model card, this feeds a known probe conversation through the model's real
chat_template.jinja and prints what comes out, plus the token ids of each delimiter.

    python src/format/inspect_template.py openbmb/MiniCPM5-1B
    python src/format/inspect_template.py llmfan46/gemma-4-E4B-it-uncensored-heretic

Read the rendered output, then add the family to tool_formats.py.
"""

import argparse
import json

PROBE_TOOLS = [{
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a file",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
    },
}]

PROBE = [
    {"role": "system", "content": "SYSTEM_MARKER"},
    {"role": "user", "content": "USER_MARKER"},
    {"role": "assistant", "content": "ASSISTANT_MARKER"},
    {"role": "tool", "content": "TOOL_RESULT_MARKER"},
    {"role": "assistant", "content": "FINAL_MARKER"},
]

CANDIDATE_DELIMS = [
    "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>",
    "<think>", "</think>", "<|tool_call_start|>", "<|tool_call_end|>",
    "<|tool_response_start|>", "```tool_code", "```tool_outputs", "<start_of_turn>",
    "<|im_start|>", "<|im_end|>", "[TOOL_CALL]",
]


def main():
    ap = argparse.ArgumentParser(description="Show a model's real tool-call convention")
    ap.add_argument("model")
    ap.add_argument("--trust-remote-code", action="store_true")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)

    print(f"=== {args.model} ===")
    print(f"vocab: {len(tok):,}")

    for label, kwargs in [("without tools", {}), ("with tools", {"tools": PROBE_TOOLS})]:
        try:
            out = tok.apply_chat_template(PROBE, tokenize=False,
                                          add_generation_prompt=False, **kwargs)
        except Exception as e:
            print(f"\n--- {label}: FAILED: {type(e).__name__}: {e}")
            continue
        print(f"\n--- rendered, {label} ---")
        print(out)

    # A tool turn has to go somewhere. Which role the template assigns it decides whether the
    # formatter can emit role:"tool" at all, or must fold results into a user turn.
    try:
        rendered = tok.apply_chat_template(PROBE, tokenize=False, add_generation_prompt=False)
        idx = rendered.find("TOOL_RESULT_MARKER")
        if idx >= 0:
            print("\n--- context around the tool result ---")
            print(repr(rendered[max(0, idx - 120):idx + 60]))
    except Exception:
        pass

    print("\n--- delimiter tokenization (1 token = native special token) ---")
    for d in CANDIDATE_DELIMS:
        ids = tok.encode(d, add_special_tokens=False)
        if len(ids) == 1:
            print(f"  {d:<26} -> {ids}   SPECIAL")
        elif len(ids) <= 3:
            print(f"  {d:<26} -> {ids}")
    print("\nOnly delimiters marked SPECIAL require skip_special_tokens=False at eval time.")

    # enable_thinking is a per-family kwarg; whether it defaults on changes the training text.
    for flag in (True, False):
        try:
            out = tok.apply_chat_template(PROBE, tokenize=False, add_generation_prompt=False,
                                          enable_thinking=flag)
            print(f"enable_thinking={flag}: accepted, <think> present={'<think>' in out}")
        except Exception as e:
            print(f"enable_thinking={flag}: rejected ({type(e).__name__})")


if __name__ == "__main__":
    main()
