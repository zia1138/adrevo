"""Adrevo configuration; the model can read the harness but only edit selection.cpp."""

from adrevo.config import AdrevoConfig, AdrevoFile, ModelSpec
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from pydantic_ai.providers.openai import OpenAIProvider


SYSTEM_MSG = """Search for local k-mer selection methods that improve conserved-anchor
window coverage at the fixed density budget specified in evaluate.py.
Replace only evo/selection.cpp. It is separately compiled and linked to the fixed
evaluate.cpp harness. Include ../selection_api.hpp and implement its exact API.
The candidate returns one local selection decision, never scores or seed keys.
Use only the supplied local context and parameters. Do not use persistent state,
I/O, clocks, process inspection, benchmark seed reconstruction, or undefined
behavior. Do not define main or replace harness symbols. Helper functions and
types should live in an anonymous namespace. Arbitrary local selection predicates
are allowed; you need not preserve the initial minimizer algorithm or a window
coverage guarantee. The harness checks repeatability and invalid target handling.
The benchmark currently tests substitutions, not indels or mapping specificity.
"""


def build_evo_models() -> list[ModelSpec]:
    # Same model setup as examples/circle_packing/config_openai.py.
    return [ModelSpec(
        model_id="gpt-5.4-mini",
        model=OpenAIResponsesModel("gpt-5.4-mini", provider=OpenAIProvider()),
        settings=OpenAIResponsesModelSettings(
            openai_reasoning_effort="high", openai_reasoning_summary="detailed",
            openai_service_tier="flex",
        ),
        input_token_cost=0.75,
        output_token_cost=4.50,
    )]


def get_adrevo_config() -> AdrevoConfig:
    return AdrevoConfig(
        task_sys_msg=SYSTEM_MSG,
        build_evo_models=build_evo_models,
        evolvable_files=(AdrevoFile("evo/selection.cpp", "cpp"),),
        fixed_files=(
            AdrevoFile("selection_api.hpp", "cpp"),
            AdrevoFile("evaluate.cpp", "cpp"),
        ),
        evaluate_file="evaluate.py",
        maximize_combined_score=True,
        max_cost=5.0,
        evaluator_timeout_sec=300,
    )
