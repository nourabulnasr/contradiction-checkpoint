"""Judges decide ALLOW / HALT for a proposed action given a plan + recent history.

Two implementations ship here:

- HeuristicJudge: zero dependencies, zero API keys, deterministic. Looks for
  sentences in the plan/recent turns that carry a negation or reversal signal
  ("don't", "actually", "hold off", "wait", ...) and checks whether the
  proposed action shares meaningful keywords with that sentence. This is the
  default judge and what the demo runs on.

- AnthropicJudge: what you'd actually deploy. Sends {plan, last N turns,
  proposed action} to a small Claude model and asks it to judge. Only usable
  if ANTHROPIC_API_KEY is set — this repo does not require it to run.

Be honest about what the heuristic judge is: a keyword/negation matcher, not
NLU. It catches the class of "explicit reversal on a topic the action also
touches" contradictions used in the demo scenarios. It will miss subtle or
implicitly-phrased contradictions — the same class of judgment the AgentAbstain
paper (arXiv:2607.10059) found frontier models get right only 59.5% of the
time. This tool does not claim to solve that; it claims to catch the obvious,
literal cases cheaply and without an API call.
"""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Sequence


class Verdict(str, Enum):
    ALLOW = "ALLOW"
    HALT = "HALT"


@dataclass
class CheckResult:
    verdict: Verdict
    reasoning: str
    confidence: float
    evidence: str | None = None  # the turn/sentence that triggered a HALT, if any


STOPWORDS = {
    "the", "a", "an", "to", "of", "in", "on", "for", "and", "or", "is", "are",
    "be", "this", "that", "it", "with", "we", "you", "i", "please", "just",
    "only", "should", "would", "will", "can", "need", "needs", "lets", "let",
    "go", "ahead", "make", "sure", "at", "as", "by", "from", "so", "but",
    "was", "were", "been", "have", "has", "had", "do", "does", "did", "not",
    "now", "then", "also", "get", "got", "into", "out", "up", "down", "over",
}

# Substrings that mark a sentence as carrying a negation / plan-reversal signal.
REVERSAL_MARKERS = [
    "don't", "do not", "never", "stop ", "avoid", "no longer", "hold off",
    "wait,", "wait ", "actually,", "actually ", "instead", "changed my mind",
    "scratch that", "cancel", "keep ", "preserve", "before you", "until we",
    "make sure", "double-check", "double check", "get sign-off",
    "get sign off", "need approval", "ask first", "hold on", "on second thought",
    "not yet", "don't touch", "leave alone", "back off",
]

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def tokenize(text: str) -> set[str]:
    """Lowercase, split into alnum tokens, drop stopwords and short tokens."""
    tokens = _TOKEN_RE.findall(text.lower())
    return {t for t in tokens if len(t) > 2 and t not in STOPWORDS}


def _sentences(text: str) -> list[str]:
    # Good enough for demo-scale turns; not a real sentence splitter.
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


class BaseJudge(ABC):
    @abstractmethod
    def check(self, plan: str, history: Sequence[tuple[str, str]], action: str) -> CheckResult:
        """history is a sequence of (role, text) turns, most recent last."""
        raise NotImplementedError


class HeuristicJudge(BaseJudge):
    """Deterministic, dependency-free contradiction matcher. Default judge."""

    def check(self, plan: str, history: Sequence[tuple[str, str]], action: str) -> CheckResult:
        action_tokens = tokenize(action)

        # Candidate constraint sentences: the plan itself, plus every turn in
        # the provided history window, split into sentences.
        candidates: list[str] = _sentences(plan)
        for _role, text in history:
            candidates.extend(_sentences(text))

        best_overlap: set[str] = set()
        best_sentence: str | None = None

        for sentence in candidates:
            low = sentence.lower()
            if not any(marker in low for marker in REVERSAL_MARKERS):
                continue
            overlap = tokenize(sentence) & action_tokens
            if len(overlap) > len(best_overlap):
                best_overlap = overlap
                best_sentence = sentence

        if best_sentence and best_overlap:
            confidence = min(1.0, 0.4 + 0.2 * len(best_overlap))
            return CheckResult(
                verdict=Verdict.HALT,
                reasoning=(
                    f"Found a reversal/negation signal that shares keywords "
                    f"{sorted(best_overlap)!r} with the proposed action: "
                    f'"{best_sentence}"'
                ),
                confidence=confidence,
                evidence=best_sentence,
            )

        return CheckResult(
            verdict=Verdict.ALLOW,
            reasoning="No recent turn carries a negation/reversal signal that shares keywords with this action.",
            confidence=0.6,
            evidence=None,
        )


class AnthropicJudge(BaseJudge):
    """Real LLM judge. Requires ANTHROPIC_API_KEY. Not used by the demo by default."""

    def __init__(self, model: str = "claude-haiku-4-5-20251001", api_key: str | None = None):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "AnthropicJudge requires ANTHROPIC_API_KEY (set it in .env — see .env.example). "
                "Use HeuristicJudge if you don't want to configure a key."
            )

    def check(self, plan: str, history: Sequence[tuple[str, str]], action: str) -> CheckResult:
        try:
            import anthropic  # imported lazily so the package has zero hard deps
        except ImportError as e:
            raise RuntimeError(
                "AnthropicJudge needs the 'anthropic' package: pip install anthropic"
            ) from e

        client = anthropic.Anthropic(api_key=self.api_key)
        history_text = "\n".join(f"[{role}] {text}" for role, text in history)
        prompt = (
            "You are a contradiction checkpoint for an autonomous agent. "
            "You are shown the agent's original plan, the last few turns of "
            "conversation since that plan was formed, and an action the agent "
            "is about to take that CANNOT be undone. Your only job: does "
            "anything said since the plan was formed contradict this specific "
            "action? Do not evaluate whether the action is a good idea in "
            "general, only whether it contradicts something said.\n\n"
            f"PLAN:\n{plan}\n\nRECENT TURNS:\n{history_text}\n\n"
            f"PROPOSED IRREVERSIBLE ACTION:\n{action}\n\n"
            'Respond with strict JSON only: {"verdict": "ALLOW"|"HALT", '
            '"reasoning": "<one sentence>", "confidence": <0-1 float>}'
        )
        response = client.messages.create(
            model=self.model,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Model didn't return clean JSON — fail safe to HALT rather than
            # silently allowing an irreversible action on a parse error.
            return CheckResult(
                verdict=Verdict.HALT,
                reasoning=f"Judge returned non-JSON output, failing safe: {raw[:200]!r}",
                confidence=0.0,
            )
        return CheckResult(
            verdict=Verdict(data.get("verdict", "HALT")),
            reasoning=data.get("reasoning", ""),
            confidence=float(data.get("confidence", 0.5)),
        )
