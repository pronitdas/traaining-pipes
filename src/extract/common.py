"""Shared helpers for session extractors.

Tool output dominates the raw corpora: the 76GB /mnt/j opencode.db averages ~81KB per
part while the median part is only ~2.8KB, i.e. the bulk is a long tail of enormous
file reads and command dumps. Training on those teaches the model to regurgitate file
contents, so every extractor truncates tool output through the same helper.
"""

DEFAULT_HEAD = 2000
DEFAULT_TAIL = 1000


def truncate_output(text, head=DEFAULT_HEAD, tail=DEFAULT_TAIL):
    """Cap a tool result to head+tail characters.

    Returns (text, chars_removed). Keeps both ends because the head carries the shape
    of the result and the tail usually carries the outcome (errors, totals, exit lines).
    """
    if not isinstance(text, str):
        return text, 0
    if len(text) <= head + tail + 64:
        return text, 0
    removed = len(text) - head - tail
    return f"{text[:head]}\n... [{removed:,} chars elided] ...\n{text[-tail:]}", removed


def truncate_json_like(value, head=DEFAULT_HEAD, tail=DEFAULT_TAIL):
    """Truncate a tool payload that may be a str, dict, or list.

    Non-string payloads are serialised first so a huge structured result is capped too.
    """
    if isinstance(value, str):
        return truncate_output(value, head, tail)
    if value is None:
        return value, 0
    import json
    try:
        s = json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        s = str(value)
    return truncate_output(s, head, tail)
