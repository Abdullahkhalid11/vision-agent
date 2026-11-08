"""Input guard for questions (PRD V1 Section 7, item 3).

Flags suspicious questions. Never rewrites them, and never silently drops them.
Flagged questions are recorded as status=rejected and never reach the model.
"""
import re
from dataclasses import dataclass

MAX_QUESTION_CHARS = 500

FLAG_PATTERNS = {
    "ignore_instructions": r"ignore\s+(all\s+|any\s+|the\s+|your\s+)?(previous|prior|above|earlier)\b",
    "system_prompt": r"system\s+prompt",
    "reveal": r"\breveal\b",
    "api_key": r"api[\s_-]*key",
    "role_override": r"\byou\s+are\s+now\b",
    "developer_mode": r"developer\s+mode",
}
_COMPILED = {name: re.compile(p, re.IGNORECASE) for name, p in FLAG_PATTERNS.items()}


@dataclass(frozen=True)
class Verdict:
    flags: tuple

    @property
    def allowed(self):
        return not self.flags


def check_question(text):
    flags = []
    if not isinstance(text, str) or not text.strip():
        flags.append("empty")
    else:
        if len(text) > MAX_QUESTION_CHARS:
            flags.append("too_long")
        for name, rx in _COMPILED.items():
            if rx.search(text):
                flags.append(name)
    return Verdict(tuple(flags))
