"""One definition per model family of how a tool call is written on the wire.

Why this exists: format_agentic_minicpm5.py encoded MiniCPM5's <tool_call> syntax, and
eval_agentic.py separately encoded how to parse it back. Two places, one fact. At eight base
models across three task heads that becomes sixteen places to disagree, and a mismatch is
invisible -- it shows up as a model that "scores 0%" when really the parser was looking for
the wrong delimiter.

So: every family is declared once here, and both the formatter and the evaluator import it.

The corpus produced by format_sft.py is the CANONICAL form -- role/content messages whose
assistant turns carry [TOOL_CALL]name\\n{json}[/TOOL_CALL]. That is a house convention, not
any model's syntax. A ToolFormat renders canonical -> target, and parses target -> (name, args)
for scoring. `markers` is itself a ToolFormat so the house format stays a comparable baseline.

Adding a family: run `python src/format/inspect_template.py <hf-model-id>` and read the tool
convention out of the model's own chat_template.jinja. Do not write one from memory -- the
delimiters differ per family and a wrong guess is silently scored as total failure.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class ToolFormat:
    name: str

    # canonical -> target
    encode_call: Callable[[str, dict], str]

    # target -> (tool_name, args); returns (None, None) when no call is present
    block_re: re.Pattern
    _parse: Callable[[str], tuple]

    # How a tool RESULT turn is carried. "tool_role" means emit {"role":"tool"} and let the
    # chat template wrap it; "user" means the family has no tool role and it must be folded
    # into a user turn.
    result_role: str = "tool"

    think_open: str = "<think>"
    think_close: str = "</think>"

    # Generation stop marker, and whether the decoder must KEEP special tokens to see it.
    # This bit is load-bearing: in MiniCPM5 </tool_call> is a single special token, so
    # decoding with skip_special_tokens=True deletes the very delimiter we stop and parse on.
    stop: str = "</tool_call>"
    skip_special: bool = False

    notes: str = ""

    def first_tool(self, text):
        return self._parse(text or "")[0]

    def tool_args(self, text):
        return self._parse(text or "")[1]


# ---------------------------------------------------------------- house marker format

MARKER_TOOL_RE = re.compile(r"\[TOOL_CALL\]([A-Za-z0-9_.\-]+)")
# Arguments arrive as an object, or -- in agentic_sft.jsonl and the eval set built from it --
# double-encoded as a JSON *string* containing the object. Both must match.
MARKER_BLOCK_RE = re.compile(
    r"\[TOOL_CALL\]([A-Za-z0-9_.\-]+)\s*(\{.*?\}|\".*?\")?\s*\[/TOOL_CALL\]", re.S)


def _decode_args(raw):
    """Parse an argument blob, unwrapping one layer of double-encoding if present."""
    if raw is None:
        return None
    try:
        v = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError:
        return None
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            return None
    return v if isinstance(v, dict) else None


def _parse_markers(text):
    m = MARKER_BLOCK_RE.search(text)
    if not m:
        hit = MARKER_TOOL_RE.search(text)
        return (hit.group(1) if hit else None), None
    return m.group(1), _decode_args(m.group(2))


MARKERS = ToolFormat(
    name="markers",
    encode_call=lambda n, a: f"\n[TOOL_CALL]{n}\n{json.dumps(a, indent=2)}[/TOOL_CALL]\n",
    block_re=MARKER_BLOCK_RE,
    _parse=_parse_markers,
    result_role="tool",
    think_open="[THINKING]", think_close="[/THINKING]",
    stop="[/TOOL_CALL]",
    skip_special=True,          # markers are ordinary text, not special tokens
    notes="House format from format_sft.py. Baseline only -- no runtime parses it.",
)


# ------------------------------------------------------- XML <tool_call> families
# MiniCPM5 and the Qwen line (Qwen2/Qwen3/Qwen3.5/Qwen3.6, and therefore Plano-Orchestrator-4B,
# Arch-Agent-7B and Ornith-1.0-9B, which are all Qwen-derived) share this convention: the call
# is a JSON object {"name","arguments"} between <tool_call> and </tool_call>, and results come
# back on a tool role that the template renders inside <tool_response>.
# Verified against MiniCPM5-1B's chat_template.jinja (tokens 2/3/10 respectively).

XML_BLOCK_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def _parse_xml(text):
    m = XML_BLOCK_RE.search(text)
    if not m:
        return None, None
    try:
        payload = json.loads(m.group(1))
    except json.JSONDecodeError:
        return None, None
    if not isinstance(payload, dict):
        return None, None
    args = payload.get("arguments")
    return payload.get("name"), (args if isinstance(args, dict) else None)


def _encode_xml(name, args):
    payload = json.dumps({"name": name, "arguments": args}, ensure_ascii=False)
    return f"<tool_call>\n{payload}\n</tool_call>"


XML_TOOL_CALL = ToolFormat(
    name="xml_tool_call",
    encode_call=_encode_xml,
    block_re=XML_BLOCK_RE,
    _parse=_parse_xml,
    result_role="tool",
    stop="</tool_call>",
    skip_special=False,         # delimiters are single special tokens; must not be stripped
    notes="MiniCPM5 + all Qwen-derived families. vLLM parsers: minicpm5 / hermes / qwen3_xml.",
)


# ---------------------------------------------------------------------- registry
#
# Maps a base model to the wire format it was post-trained on. Keys are the --target names.
# UNVERIFIED entries are deliberately absent rather than guessed: gemma4 and lfm2 use their own
# delimiters, and inspect_template.py must confirm them before either is added. A wrong guess
# here trains a model to emit syntax nothing can parse.

# ------------------------------------------------------------------ LFM2 (Liquid)
# Verified against Dingdust/LFM2.5-2.6B-heretic: ChatML-ish, but with a genuine `tool` role
# (<|im_start|>tool) and its own call delimiters <|tool_call_start|> / <|tool_call_end|>
# (tokens 124905/124906). <think>/</think> are also native (124901/124902).
#
# The payload shape inside those delimiters is a JSON list here rather than Liquid's Pythonic
# call form. That is a deliberate choice: we are fine-tuning, so the model learns whatever we
# emit, and a JSON list parses unambiguously. The cost is that it partially overwrites the
# pretrained tool behaviour -- acceptable, since these delimiters are still the native ones.

LFM2_BLOCK_RE = re.compile(r"<\|tool_call_start\|>\s*(\[.*?\]|\{.*?\})\s*<\|tool_call_end\|>", re.S)


def _parse_lfm2(text):
    m = LFM2_BLOCK_RE.search(text)
    if not m:
        return None, None
    try:
        payload = json.loads(m.group(1))
    except json.JSONDecodeError:
        return None, None
    if isinstance(payload, list):
        payload = payload[0] if payload else None
    if not isinstance(payload, dict):
        return None, None
    args = payload.get("arguments")
    return payload.get("name"), (args if isinstance(args, dict) else None)


LFM2 = ToolFormat(
    name="lfm2",
    encode_call=lambda n, a: "<|tool_call_start|>" + json.dumps(
        [{"name": n, "arguments": a}], ensure_ascii=False) + "<|tool_call_end|>",
    block_re=LFM2_BLOCK_RE,
    _parse=_parse_lfm2,
    result_role="tool",
    stop="<|tool_call_end|>",
    skip_special=False,
    notes="Liquid LFM2.5. Native tool role. Delimiters are special tokens.",
)


FORMATS = {
    "markers": MARKERS,
    "minicpm5": XML_TOOL_CALL,
    "qwen": XML_TOOL_CALL,
    "lfm2": LFM2,
}

# Families deliberately NOT registered, with the evidence:
#
# gemma4  Gemma-4's chat template SILENTLY DROPS role:"tool" turns -- a probe conversation
#         renders with the tool result missing entirely, so every tool result in the corpus
#         would vanish during templating. It also emits assistant turns as <|turn>model,
#         declares tools in a bespoke non-JSON DSL (<|tool>declaration:name{...}<tool|>), and
#         <think> is 3 ordinary tokens rather than a special one.
#
#         This is FAMILY-WIDE, not a bad repack. Verified across three variants -- E4B
#         (llmfan46/gemma-4-E4B-it-uncensored-heretic), 12B agentic
#         (huihui-ai/Huihui-gemma-4-12B-agentic-fable5-abliterated) and 31B
#         (wangzhang/gemma-4-31B-it-abliterated) -- which render byte-for-byte identically and
#         all drop the tool turn, including the one that advertises itself as agentic.
#
#         Excluded from the agentic head. Re-including it means authoring a corrected template,
#         which would then be required at serving time too (stock vLLM would not match).
#         Reproduce with: python src/format/inspect_template.py <any gemma-4 model>


# base model -> format key. Verified via HF api + chat_template inspection.
MODEL_FORMATS = {
    "openbmb/MiniCPM5-1B": "minicpm5",
    "katanemo/Plano-Orchestrator-4B": "qwen",        # Qwen3ForCausalLM
    "katanemo/Arch-Agent-7B": "qwen",                # Qwen2ForCausalLM
    "ornith-ai/Ornith-1.0-9B": "qwen",               # Qwen3_5ForConditionalGeneration
    "Dingdust/LFM2.5-2.6B-heretic": "lfm2",          # Lfm2ForCausalLM
    "Youssofal/Qwen3.6-35B-A3B-Abliterated-Heretic-BF16": "qwen",
    # Pending template inspection:
    #   unsloth/Muse-Glimmer-30B  (MuseGlimmerForConditionalGeneration)
}

# Training ladder, cheapest first. A 30B costs ~30x a 1B per epoch, so the large models are
# only worth paying for once eval shows the corpus is teaching the task on the small ones.
LADDER = [
    "openbmb/MiniCPM5-1B",
    "Dingdust/LFM2.5-2.6B-heretic",
    "katanemo/Plano-Orchestrator-4B",
    "katanemo/Arch-Agent-7B",
    "ornith-ai/Ornith-1.0-9B",
    "unsloth/Muse-Glimmer-30B",
    "Youssofal/Qwen3.6-35B-A3B-Abliterated-Heretic-BF16",
]


def get(name):
    if name not in FORMATS:
        raise KeyError(f"unknown tool format {name!r}; known: {sorted(FORMATS)}")
    return FORMATS[name]


def for_model(model_id):
    """Format a base model expects, or None if it has not been verified yet."""
    key = MODEL_FORMATS.get(model_id)
    return FORMATS[key] if key else None
