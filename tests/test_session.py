import unittest

from contradiction_checkpoint import CheckpointSession, ContradictionHalt


class TestCheckpointSession(unittest.TestCase):
    def test_guard_allows_and_executes(self):
        session = CheckpointSession(plan="Send the report to finance.")

        @session.guard
        def send_report(to):
            return f"sent to {to}"

        result = send_report("finance@co.com")
        self.assertEqual(result, "sent to finance@co.com")
        self.assertEqual(len(session.log), 1)

    def test_guard_halts_and_does_not_execute(self):
        session = CheckpointSession(plan="Clean up tests/.")
        session.add_turn("user", "Actually, don't delete the tests fixtures, keep them.")

        calls = []

        @session.guard
        def delete_file(path):
            calls.append(path)  # should never run
            return "deleted"

        with self.assertRaises(ContradictionHalt) as ctx:
            delete_file("tests/fixtures.py")

        self.assertEqual(calls, [])  # underlying function never executed
        self.assertIn("delete_file", ctx.exception.action_description)

    def test_window_limits_history_considered(self):
        session = CheckpointSession(plan="Deploy the service.", window=1)
        # Old turn (outside window) carries the contradiction; should NOT be seen.
        session.add_turn("user", "Actually, don't deploy the service yet.")
        session.add_turn("user", "totally unrelated remark")

        @session.guard
        def deploy(name):
            return f"deployed {name}"

        result = deploy("billing-service")
        self.assertEqual(result, "deployed billing-service")


if __name__ == "__main__":
    unittest.main()
