"""Tests for deterministic evaluation checks and dependency-blocked reporting."""

import argparse
import unittest

from scripts.evaluate_poc import (
    CASES,
    UNKNOWN_ANSWER,
    _blocked_report,
    _matches_case,
)


class EvaluationCheckTests(unittest.TestCase):
    def test_numeric_answer_ignores_commas_and_spaces(self) -> None:
        self.assertTrue(_matches_case("Rs 1,000 per month", [2], CASES[2]))

    def test_expected_citation_page_is_required(self) -> None:
        self.assertFalse(_matches_case("24 days", [2], CASES[0]))

    def test_unsupported_question_requires_exact_fallback(self) -> None:
        self.assertTrue(_matches_case(UNKNOWN_ANSWER, [], CASES[-1]))
        self.assertFalse(
            _matches_case("I don't know.", [], CASES[-1])
        )

    def test_dependency_blocker_report_does_not_claim_measured_results(self) -> None:
        args = argparse.Namespace(
            power_watts=100,
            electricity_rate_inr_per_kwh=8,
        )
        report = _blocked_report(ImportError("missing package"), args)
        self.assertIn("Live evaluation not run", report)
        self.assertIn("NOT RUN", report)
        self.assertIn("No application latency or energy measurements", report)


if __name__ == "__main__":
    unittest.main()
