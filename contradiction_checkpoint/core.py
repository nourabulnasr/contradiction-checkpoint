"""Core session object: holds a running plan + conversation history, and wraps
irreversible actions with a contradiction check right before they execute.
"""

from __future__ import annotations

import functools
import inspect
from typing import Callable, TypeVar

from .judges import BaseJudge, CheckResult, HeuristicJudge, Verdict

F = TypeVar("F", bound=Callable)


class ContradictionHalt(Exception):
    """Raised instead of executing an irreversible action the checkpoint flagged."""

    def __init__(self, result: CheckResult, action_description: str):
        self.result = result
        self.action_description = action_description
        super().__init__(
            f"HALTED before executing: {action_description}\nReason: {result.reasoning}"
        )


class CheckpointSession:
    """Represents one running agent session: a plan plus a rolling window of turns.

    Usage:
        session = CheckpointSession(plan="Refactor auth. Do not touch tests/.")
        session.add_turn("user", "actually keep the old test fixtures as reference")

        @session.guard
        def delete_file(path):
            os.remove(path)

        delete_file("tests/fixtures/old.py")  # raises ContradictionHalt
    """

    def __init__(self, plan: str, judge: BaseJudge | None = None, window: int = 10):
        self.plan = plan
        self.judge = judge or HeuristicJudge()
        self.window = window
        self._history: list[tuple[str, str]] = []
        self.log: list[dict] = []  # every check made in this session, for the demo transcript

    def add_turn(self, role: str, text: str) -> None:
        self._history.append((role, text))

    def recent_history(self) -> list[tuple[str, str]]:
        return self._history[-self.window:]

    def check_action(self, action_description: str) -> CheckResult:
        result = self.judge.check(self.plan, self.recent_history(), action_description)
        if not isinstance(result, CheckResult) or (
            result.verdict is not Verdict.ALLOW and result.verdict is not Verdict.HALT
        ):
            result = CheckResult(
                verdict=Verdict.HALT,
                reasoning="Judge returned an invalid CheckResult or verdict; execution halted.",
                confidence=0.0,
            )
        self.log.append({"action": action_description, "result": result})
        return result

    def guard(self, func: F) -> F:
        """Decorator: wraps an irreversible callable with a contradiction check.

        On ALLOW, calls func normally and returns its result.
        On HALT, raises ContradictionHalt instead of calling func.
        """

        def check_call(args, kwargs):
            arg_desc = ", ".join(
                [repr(a) for a in args] + [f"{k}={v!r}" for k, v in kwargs.items()]
            )
            action_description = f"{func.__name__}({arg_desc})"
            result = self.check_action(action_description)
            if result.verdict is not Verdict.ALLOW:
                raise ContradictionHalt(result, action_description)

        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                # A coroutine can be created long before it starts. Check the
                # current history here, not when the coroutine object is made.
                check_call(args, kwargs)
                return await func(*args, **kwargs)

            async_wrapper.__wrapped_unguarded__ = func
            return async_wrapper

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            check_call(args, kwargs)
            return func(*args, **kwargs)

        wrapper.__wrapped_unguarded__ = func  # escape hatch for tests
        return wrapper
