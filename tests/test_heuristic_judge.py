import unittest

from contradiction_checkpoint.judges import HeuristicJudge, Verdict, tokenize


class TestTokenize(unittest.TestCase):
    def test_drops_stopwords_and_short_tokens(self):
        tokens = tokenize("Do not delete the tests folder, ok?")
        self.assertIn("delete", tokens)
        self.assertIn("tests", tokens)
        self.assertIn("folder", tokens)
        self.assertNotIn("the", tokens)
        self.assertNotIn("not", tokens)
        self.assertNotIn("ok", tokens)  # len <= 2


class TestHeuristicJudge(unittest.TestCase):
    def setUp(self):
        self.judge = HeuristicJudge()

    def test_allow_when_no_conflict(self):
        result = self.judge.check(
            plan="Email the Q3 report to finance.",
            history=[("user", "sounds good"), ("assistant", "will do")],
            action="send_report('finance@co.com', 'q3.pdf')",
        )
        self.assertEqual(result.verdict, Verdict.ALLOW)

    def test_halt_on_explicit_reversal_with_shared_keywords(self):
        result = self.judge.check(
            plan="Clean up unused test fixtures in tests/.",
            history=[
                ("user", "Actually, keep the old auth test fixtures as a reference, don't delete them yet."),
            ],
            action="delete_file('tests/fixtures/legacy_auth_backup.py')",
        )
        self.assertEqual(result.verdict, Verdict.HALT)
        self.assertIn("fixtures", result.evidence.lower())

    def test_allow_when_reversal_signal_present_but_unrelated_topic(self):
        result = self.judge.check(
            plan="Send the newsletter to subscribers.",
            history=[
                ("user", "Actually, hold off on the marketing budget spreadsheet for now."),
            ],
            action="send_newsletter('weekly-digest')",
        )
        self.assertEqual(result.verdict, Verdict.ALLOW)

    def test_halt_picks_the_strongest_matching_sentence(self):
        result = self.judge.check(
            plan="Ship the billing-service hotfix once tests pass.",
            history=[
                ("assistant", "Tests are passing, billing-service is ready to deploy."),
                ("user", "Wait, hold off deploying billing-service until we get sign-off from finance."),
            ],
            action="deploy_to_production('billing-service')",
        )
        self.assertEqual(result.verdict, Verdict.HALT)
        self.assertIn("sign-off", result.evidence)

    def test_window_default_confidence_is_between_0_and_1(self):
        result = self.judge.check(plan="", history=[], action="noop()")
        self.assertTrue(0.0 <= result.confidence <= 1.0)


if __name__ == "__main__":
    unittest.main()
