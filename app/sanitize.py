"""Input sanitisation: normalise text, reject malformed input, flag injection attempts.

Design: we *never* silently rewrite meaning. We normalise whitespace/unicode,
enforce a length budget, and classify the input into one of three outcomes:

  ok       -> pass to the model
  flagged  -> pass to the model, but tell it (via system prompt) the text matched
              an injection-like pattern and must be treated as data only
  blocked  -> do not call the model; return a canned response

Blocking is reserved for strong, unambiguous patterns so that legitimate
questions containing words like "system" or "ignore" are not rejected.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MULTI_WS = re.compile(r"[ \t]+")
_MULTI_NL = re.compile(r"\n{3,}")

# Strong patterns: clear attempts to override instructions or impersonate roles.
_BLOCK_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"ignore\s+(all\s+|any\s+)?(previous|prior|above|earlier)\s+(instructions|prompts?|rules)",
        r"disregard\s+(all\s+|any\s+)?(previous|prior|above|earlier)\s+(instructions|prompts?|rules)",
        r"(reveal|print|show|repeat|output|leak)\s+(me\s+)?(your|the)\s+(system|hidden|secret)\s+(prompt|instructions)",
        r"you\s+are\s+now\s+(in\s+)?(dan|developer\s+mode|jailbreak)",
        r"</?\s*(system|assistant|human|developer)\s*>",
        r"\[\s*(system|inst|/inst)\s*\]",
        r"<\|?(im_start|im_end|endoftext)\|?>",
    )
]
# Weak patterns: suspicious but plausible in a legitimate question.
_FLAG_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(system prompt|developer message|jailbreak)\b",
        r"\bpretend\s+(to\s+be|you\s+are)\b",
        r"\bact\s+as\s+(an?\s+)?(unrestricted|unfiltered)",
        r"\bnew\s+instructions?\b",
        r"[A-Za-z0-9+/]{120,}={0,2}",  # long base64-looking blob
    )
]


class Verdict(str, Enum):
    OK = "ok"
    FLAGGED = "flagged"
    BLOCKED = "blocked"


@dataclass
class SanitizedInput:
    text: str
    verdict: Verdict
    reasons: list[str] = field(default_factory=list)
    original_length: int = 0

    @property
    def allowed(self) -> bool:
        return self.verdict is not Verdict.BLOCKED


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = _CONTROL.sub("", text).replace("\r\n", "\n").replace("\r", "\n")
    text = _MULTI_WS.sub(" ", text)
    text = _MULTI_NL.sub("\n\n", text)
    return text.strip()


def _looks_meaningless(text: str) -> bool:
    letters = sum(ch.isalnum() for ch in text)
    if letters == 0:
        return True
    # mostly punctuation / symbols, or one repeated character ("aaaaaaa")
    if letters / max(len(text), 1) < 0.3 and len(text) > 3:
        return True
    return len(set(text.lower())) <= 2 and len(text) > 4


def sanitize(raw: str, max_chars: int = 2000) -> SanitizedInput:
    original_length = len(raw or "")
    text = normalize(raw)
    reasons: list[str] = []

    if not text:
        return SanitizedInput("", Verdict.BLOCKED, ["empty_input"], original_length)
    if len(text) > max_chars:
        return SanitizedInput(text[:max_chars], Verdict.BLOCKED, [f"too_long>{max_chars}"], original_length)
    if _looks_meaningless(text):
        return SanitizedInput(text, Verdict.BLOCKED, ["malformed_input"], original_length)

    for pat in _BLOCK_PATTERNS:
        if pat.search(text):
            reasons.append(f"injection:{pat.pattern[:40]}")
    if reasons:
        return SanitizedInput(text, Verdict.BLOCKED, reasons, original_length)

    for pat in _FLAG_PATTERNS:
        if pat.search(text):
            reasons.append(f"suspicious:{pat.pattern[:40]}")
    verdict = Verdict.FLAGGED if reasons else Verdict.OK
    return SanitizedInput(text, verdict, reasons, original_length)


BLOCKED_MESSAGES = {
    "empty_input": "I didn't receive a question. Ask me anything about Indian unicorn startups — for example, \"What does Razorpay do?\"",
    "malformed_input": "I couldn't make sense of that input. Try a question like \"List fintech unicorns in Bengaluru\".",
    "too_long": "That message is too long for me to process. Please keep questions under {max_chars} characters.",
    "injection": "I can only help with questions about the Indian unicorn startups in my dataset, and I can't change how I operate. What would you like to know about these companies?",
}


def blocked_reply(result: SanitizedInput, max_chars: int) -> str:
    for reason in result.reasons:
        key = reason.split(":")[0].split(">")[0]
        if key in BLOCKED_MESSAGES:
            return BLOCKED_MESSAGES[key].format(max_chars=max_chars)
    return BLOCKED_MESSAGES["malformed_input"]
