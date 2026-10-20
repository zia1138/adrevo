import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from adrevo.cli import app
from adrevo.plan2build import eval_plan_prompt, research_plan_prompt


class PlanTests(unittest.TestCase):
    def test_research_prompt_defines_measurements_and_eval_prompt_defines_score(self):
        self.assertIn("## What must be measured", research_plan_prompt)
        self.assertNotIn("`combined_score`", research_plan_prompt)
        self.assertIn("## Scoring system", eval_plan_prompt)
        self.assertIn("`combined_score`", eval_plan_prompt)
        self.assertIn("measurements\nspecified in `RESEARCH_PLAN.md`", eval_plan_prompt)

    def run_plan(
        self,
        command,
        plan_filename,
        create_plan=True,
        session_id="test-session-id",
        include_research_plan=False,
    ):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "project"
            project.mkdir()
            (project / "BACKGROUND.md").write_text("Related work")
            if include_research_plan:
                (project / "RESEARCH_PLAN.md").write_text("Research thesis")
            results = Path(directory) / "plan results"

            def respond(prompt):
                self.assertIn(f"`{plan_filename}`", prompt)
                if create_plan:
                    (results / plan_filename).write_text("Plan")

            with patch("adrevo.cli.Codex") as codex:
                codex.return_value.session_id = session_id
                codex.return_value.initial_msg.side_effect = respond
                result = CliRunner().invoke(app, [command, str(project), str(results)])
                codex.return_value.initial_msg.assert_called_once()
                codex.return_value.send_msg.assert_not_called()
                files = sorted(p.name for p in results.iterdir())
                return result, files, results

    def test_research_plan_displays_session_and_resume_command(self):
        result, files, results = self.run_plan("research-plan", "RESEARCH_PLAN.md")
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(files, ["BACKGROUND.md", "RESEARCH_PLAN.md"])
        self.assertIn("Created research plan:", result.output)
        self.assertIn("Codex session ID: test-session-id", result.output)
        self.assertIn(f"cd '{results.resolve()}' && codex resume test-session-id", result.output)

    def test_eval_plan_consumes_research_plan_and_creates_build_plan(self):
        result, files, _ = self.run_plan(
            "eval-plan", "EVAL_PLAN.md", include_research_plan=True
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(files, ["BACKGROUND.md", "EVAL_PLAN.md", "RESEARCH_PLAN.md"])
        self.assertIn("Created evaluation plan:", result.output)

    def test_missing_plan_reports_error_and_session(self):
        result, files, _ = self.run_plan(
            "research-plan", "RESEARCH_PLAN.md", create_plan=False
        )
        self.assertEqual(result.exit_code, 1, result.output)
        self.assertEqual(files, ["BACKGROUND.md"])
        self.assertIn("Expected plan file was not created:", result.output)
        self.assertIn("codex resume test-session-id", result.output)

    def test_missing_session_reports_error(self):
        result, _, _ = self.run_plan(
            "research-plan", "RESEARCH_PLAN.md", session_id=None
        )
        self.assertEqual(result.exit_code, 1, result.output)
        self.assertIn("Codex did not return a session ID.", result.output)

    def test_eval_plan_requires_research_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "project"
            project.mkdir()
            results = Path(directory) / "results"
            result = CliRunner().invoke(app, ["eval-plan", str(project), str(results)])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("RESEARCH_PLAN.md is required", result.output)


class BuildTests(unittest.TestCase):
    def test_build_copies_the_evaluation_plan_and_required_background(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "project"
            project.mkdir()
            (project / "EVAL_PLAN.md").write_text("Evaluation plan")
            (project / "BACKGROUND.md").write_text("background")
            (project / "unrelated.txt").write_text("do not copy")
            results = Path(directory) / "results"

            result = CliRunner().invoke(app, ["build", str(project), str(results)])

            self.assertEqual(result.exit_code, 0, result.output)
            self.assertEqual(
                sorted(path.name for path in results.iterdir()),
                ["BACKGROUND.md", "BUILD.md", "EVAL_PLAN.md"],
            )
            self.assertIn("`EVAL_PLAN.md`", (results / "BUILD.md").read_text())
            self.assertIn("Build workspace is ready.", result.output)
            self.assertIn("codex 'Read BUILD.md", result.output)
            self.assertIn("start Codex interactively", result.output)
            self.assertIn("Then paste this prompt:", result.output)

    def test_build_requires_evaluation_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "project"
            project.mkdir()
            (project / "BACKGROUND.md").write_text("Related work")
            results = Path(directory) / "results"

            result = CliRunner().invoke(app, ["build", str(project), str(results)])

            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("EVAL_PLAN.md is required", result.output)
