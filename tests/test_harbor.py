import tempfile
import tomllib
import unittest
from pathlib import Path

from typer.testing import CliRunner

from adrevo.cli import app


class HarborExportTests(unittest.TestCase):
    def test_cli_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / "source"
            output = root / "harbor-task"
            source_files = {
                "config.py": '''from adrevo.config import AdrevoConfig, ModelSpec
from pydantic_ai.models.test import TestModel
def models():
    return [ModelSpec(model_id="test", model=TestModel(), settings={})]
def get_adrevo_config():
    return AdrevoConfig(build_evo_models=models, task_sys_msg="Multiply matrices.")
''',
                "config_openai.py": "API_KEY = 'not-exported'\n",
                "pyproject.toml": '[project]\nname = "fixture"\nversion = "0.1.0"\n',
                "evaluate_local.py": "# Development evaluator\n",
                "evaluate_docker.py": "# Final evaluator\n",
                "baseline/main.py": "# Trusted baseline\n",
                "datasets/example.txt": "input data\n",
                "evo/main.py": "print('candidate')\n",
                "evo/helpers.py": "VALUE = 42\n",
                "evo/pyproject.toml": '[project]\nname = "candidate"\nversion = "0.1.0"\n',
                ".env": "SECRET=not-exported\n",
                "evo/.venv/marker": "not-exported",
                "__pycache__/marker": "not-exported",
                "results_previous/result": "not-exported",
                "results.json": '{"correct": true, "combined_score": 1.0}\n',
            }
            for relative, contents in source_files.items():
                path = project / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(contents)

            arguments = [
                "harbor", str(project), "--output", str(output),
                "--config", "config.py",
                "--evaluate", "evaluate_local.py",
                "--agent-timeout-sec", "900",
            ]
            runner = CliRunner()
            result = runner.invoke(app, arguments)
            self.assertEqual(result.exit_code, 0, result.output)
            task = tomllib.loads((output / "task.toml").read_text())
            self.assertEqual(task["agent"]["timeout_sec"], 900)
            self.assertEqual(task["artifacts"], [
                {"source": "/app/evo", "destination": "evo"},
            ])
            instructions = (output / "instruction.md").read_text()
            self.assertIn("Multiply matrices.", instructions)
            self.assertIn("uv run -qq --project . python evaluate_local.py", instructions)
            self.assertIn("Continue until you believe you have achieved a good solution", instructions)
            self.assertNotIn("Harbor", instructions)
            self.assertIn("--disable-verification", result.output)
            self.assertEqual({p.name for p in output.iterdir()}, {
                "instruction.md", "task.toml", "environment",
            })
            snapshot = output / "environment" / "project"
            for relative, contents in source_files.items():
                self.assertEqual((project / relative).read_text(), contents)
                if (
                    contents == "not-exported"
                    or relative == ".env"
                    or relative.startswith("config")
                    or relative == "evaluate_docker.py"
                    or relative == "results.json"
                ):
                    self.assertFalse((snapshot / relative).exists(), relative)
                else:
                    self.assertEqual((snapshot / relative).read_text(), contents)
            self.assertIn("WORKDIR /app", (output / "environment" / "Dockerfile").read_text())

            self.assertNotEqual(runner.invoke(app, arguments).exit_code, 0)
            arguments[3] = str(project / "nested-output")
            self.assertNotEqual(runner.invoke(app, arguments).exit_code, 0)
            self.assertFalse((project / "nested-output").exists())


if __name__ == "__main__":
    unittest.main()
