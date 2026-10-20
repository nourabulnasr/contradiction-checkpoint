"""Execution depends on a current check and an explicit, valid ALLOW."""

import asyncio
import unittest

from contradiction_checkpoint import CheckpointSession, ContradictionHalt
from contradiction_checkpoint.judges import BaseJudge, CheckResult, Verdict


class StaticJudge(BaseJudge):
    def __init__(self, result):
        self.result = result

    def check(self, plan, history, action):
        return self.result


class TestMExplicitDecision(unittest.TestCase):
    def test_non_enum_verdicts_never_execute(self):
        for verdict in ["HALT", "ALLOW", "UNKNOWN", None, False]:
            with self.subTest(verdict=verdict):
                result = CheckResult(verdict=verdict, reasoning="test response", confidence=1)
                session = CheckpointSession("Send report", judge=StaticJudge(result))
                calls = []

                @session.guard
                def send_report():
                    calls.append("sent")

                with self.assertRaises(ContradictionHalt):
                    send_report()
                self.assertEqual(calls, [])
                self.assertIs(session.log[-1]["result"].verdict, Verdict.HALT)

    def test_non_result_objects_halt_with_a_record(self):
        for result in [None, {}, "ALLOW"]:
            with self.subTest(result=result):
                session = CheckpointSession("Send report", judge=StaticJudge(result))

                @session.guard
                def send_report():
                    self.fail("invalid judge response reached execution")

                with self.assertRaises(ContradictionHalt):
                    send_report()
                self.assertIs(session.log[-1]["result"].verdict, Verdict.HALT)


class TestMAsyncExecutionBoundary(unittest.IsolatedAsyncioTestCase):
    async def test_later_reversal_blocks_previously_created_coroutine(self):
        session = CheckpointSession("Deploy the service.")
        calls = []

        @session.guard
        async def deploy(name):
            calls.append(name)

        pending = deploy("billing-service")
        session.add_turn("user", "Actually, don't deploy the billing service yet.")
        with self.assertRaises(ContradictionHalt):
            await pending
        self.assertEqual(calls, [])

    async def test_creation_is_not_execution_or_an_authorization_check(self):
        session = CheckpointSession("Send report.")
        calls = []

        @session.guard
        async def send_report():
            calls.append("sent")
            return "ok"

        pending = send_report()
        try:
            self.assertEqual(session.log, [])
            self.assertEqual(calls, [])
            self.assertTrue(asyncio.iscoroutinefunction(send_report))
            self.assertEqual(await pending, "ok")
            self.assertEqual(calls, ["sent"])
            self.assertEqual(len(session.log), 1)
        finally:
            pending.close()

    async def test_invalid_async_decision_never_executes(self):
        session = CheckpointSession(
            "Send report", judge=StaticJudge(CheckResult("HALT", "test", 1))
        )

        @session.guard
        async def send_report():
            self.fail("invalid decision reached the async action")

        with self.assertRaises(ContradictionHalt):
            await send_report()

    async def test_valid_async_error_keeps_its_original_type(self):
        session = CheckpointSession("Send report.")

        @session.guard
        async def send_report():
            raise LookupError("backend failed")

        with self.assertRaisesRegex(LookupError, "backend failed"):
            await send_report()


if __name__ == "__main__":
    unittest.main()
