"""Shared types and text helpers for the solvers."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

SOLVED = "solved"          # answer is known
INFEASIBLE = "infeasible"  # proven unsolvable -> no answer, no tokens
UNKNOWN = "unknown"        # not understood by the free solver -> candidate for the LLM


@dataclass
class Outcome:
    status: str
    answer: Optional[str] = None
    reason: str = ""
    tier: str = "deterministic"
    info: dict = field(default_factory=dict)

    @staticmethod
    def solved(answer, reason: str = "", tier: str = "deterministic", **info) -> "Outcome":
        return Outcome(SOLVED, str(answer), reason, tier, info)

    @staticmethod
    def infeasible(reason: str, tier: str = "deterministic", **info) -> "Outcome":
        return Outcome(INFEASIBLE, None, reason, tier, info)

    @staticmethod
    def unknown(reason: str, **info) -> "Outcome":
        return Outcome(UNKNOWN, None, reason, "deterministic", info)


# Sentences of the instruction that only describe the output protocol.
_PROTOCOL = re.compile(r"answer\.txt|current directory|file tools|when you are done|write only|work only", re.I)
# A sentence that merely names a tool ("Compute it with the bash tool (python3 -c).")
# is boilerplate too - but only if it carries no quantity: "Using Python, it ships
# 5 boxes per day for 3 days." is part of the problem and must be kept.
_TOOL_WORD = re.compile(r"\bbash\b|\bpython3?\b|\bgrep\b", re.I)
_QUANTITY = re.compile(
    r"(?<![\w.])\d[\d,.]*(?![\w])|\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"half|twice|double[sd]?|triple[sd]?|dozen)\b", re.I)

_SENT_SPLIT = re.compile(r"(?<=[.?!])\s+(?=[A-Z(\"'`])")


def _is_boilerplate(sentence: str) -> bool:
    if _PROTOCOL.search(sentence):
        return True
    return bool(_TOOL_WORD.search(sentence)) and not _QUANTITY.search(sentence)


def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return []
    parts = _SENT_SPLIT.split(text)
    return [p.strip() for p in parts if p.strip()]


def content_sentences(instruction: str) -> List[str]:
    """Instruction sentences that carry the actual problem statement."""
    return [s for s in split_sentences(instruction) if not _is_boilerplate(s)]


def problem_text(instruction: str) -> str:
    """The instruction without the tooling boilerplate (used in LLM prompts)."""
    return " ".join(content_sentences(instruction)) or (instruction or "").strip()


_NUM_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "a": 1, "an": 1,
}
NUM = r"(\d[\d,]*|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"


def parse_int(tok: str) -> int:
    tok = tok.strip().lower()
    if tok in _NUM_WORDS:
        return _NUM_WORDS[tok]
    return int(tok.replace(",", ""))
