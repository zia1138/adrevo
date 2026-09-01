import json
import tempfile
import unittest
from pathlib import Path

from pydantic_ai.models.test import TestModel
from pydantic_ai.settings import ModelSettings

from adrevo.agents import AdrevoState
from adrevo.config import AdrevoConfig, ModelSpec
from adrevo.database import Program, ProgramDatabase


def make_config(
    model_id: str = "test-model",
    input_token_cost: float = 1.25,
    output_token_cost: float = 2.5,
) -> AdrevoConfig:
    return AdrevoConfig(
        build_evo_models=lambda: [
            ModelSpec(
                model_id=model_id,
                model=TestModel(),
                settings=ModelSettings(),
                input_token_cost=input_token_cost,
                output_token_cost=output_token_cost,
            )
        ]
    )


class ResumeStateTests(unittest.TestCase):
    def test_database_checkpoint_round_trip(self):
        database_class = ProgramDatabase.__ray_metadata__.modified_class
        database = database_class(make_config())
        try:
            initial = Program(
                id="initial",
                files={"evo/main.py": "print('initial')"},
                model_id="initial",
                correct=True,
                combined_score=1.0,
            )
            child = Program(
                id="child",
                files={"evo/main.py": "print('child')"},
                model_id="test-model",
                parent_id=initial.id,
                generation=1,
                correct=True,
                combined_score=2.0,
            )
            database.add_initial(initial, b"initial-zip")
            database.add(child, b"child-zip")
            saved_epoch = database.search_epoch

            with tempfile.TemporaryDirectory() as temp_dir_name:
                resume_dir = Path(temp_dir_name)
                (resume_dir / "database_state.json").write_text(
                    json.dumps(database.get_database_state()),
                    encoding="utf-8",
                )
                (resume_dir / "initial.zip").write_bytes(b"initial-zip")
                (resume_dir / "child.zip").write_bytes(b"child-zip")

                restored = database_class(
                    make_config(model_id="replacement-model"),
                    resume_dir=str(resume_dir),
                )
                try:
                    self.assertEqual(restored.global_best_id, "child")
                    self.assertEqual(restored.search_focus_id, "child")
                    self.assertEqual(restored.search_epoch, saved_epoch + 1)
                    self.assertEqual(restored.programs["initial"].children_ids, ["child"])
                    self.assertEqual(restored.get_zip_bytes("child"), b"child-zip")
                    self.assertFalse(restored.cancellation_requested)
                    self.assertEqual(restored.active_claims, {})
                finally:
                    restored.close()
        finally:
            database.close()

    def test_agent_state_checkpoint_allows_model_changes(self):
        state_class = AdrevoState.__ray_metadata__.modified_class
        state = state_class(make_config())
        state.generation = 17
        state.input_tokens["test-model"] = 123
        state.output_tokens["test-model"] = 45

        checkpoint = json.loads(json.dumps(state.snapshot()))
        replacement_config = make_config(
            model_id="replacement-model",
            input_token_cost=3.0,
            output_token_cost=4.0,
        )
        restored = state_class(replacement_config, checkpoint=checkpoint)

        self.assertEqual(restored.generation, 17)
        self.assertEqual(restored.input_tokens, {"replacement-model": 0})
        self.assertEqual(restored.output_tokens, {"replacement-model": 0})
        self.assertAlmostEqual(
            restored.compute_cost(),
            (123 * 1.25 + 45 * 2.5) / 1e6,
        )

        restored.input_tokens["replacement-model"] = 10
        restored.output_tokens["replacement-model"] = 5
        expected_cost = (
            (123 * 1.25 + 45 * 2.5) / 1e6
            + (10 * 3.0 + 5 * 4.0) / 1e6
        )
        self.assertAlmostEqual(restored.compute_cost(), expected_cost)

        # A second resume rolls the replacement model's usage into the carried
        # historical cost without requiring that model to remain configured.
        second_checkpoint = json.loads(json.dumps(restored.snapshot()))
        second_restored = state_class(
            make_config(model_id="third-model"),
            checkpoint=second_checkpoint,
        )
        self.assertAlmostEqual(second_restored.compute_cost(), expected_cost)


if __name__ == "__main__":
    unittest.main()
