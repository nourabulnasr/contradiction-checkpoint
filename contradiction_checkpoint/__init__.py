"""Contradiction Checkpoint — re-check the plan for drift right before an irreversible action.

Core idea (see README for the paper this is built against): most agent guardrails
gate on budget, secrets, or prompt injection. None of them re-read the last few
turns of a running session immediately before an irreversible tool call and ask
"does anything said since the plan was formed contradict this specific action?"
This package is a small, honest attempt at exactly that narrow check.
"""

from .core import CheckpointSession, ContradictionHalt, CheckResult, Verdict
from .judges import BaseJudge, HeuristicJudge, AnthropicJudge

__all__ = [
    "CheckpointSession",
    "ContradictionHalt",
    "CheckResult",
    "Verdict",
    "BaseJudge",
    "HeuristicJudge",
    "AnthropicJudge",
]
