import textwrap

from adrevo.config import AdrevoConfig, EvolvableFile, ModelSpec
from pydantic_ai.models.cerebras import CerebrasModel, CerebrasModelSettings
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from pydantic_ai.providers.cerebras import CerebrasProvider
from pydantic_ai.providers.openai import OpenAIProvider


SYSTEM_MSG = textwrap.dedent("""\
    Pack exactly 26 non-overlapping circles in a unit square and maximize the
    sum of their radii. Replace complete candidate files; there are no protected
    EVOLVE-BLOCK regions.

    Running `evo/main.py` must write `circle_packing.json` in its working
    directory. The JSON object must contain `centers`, an array with shape
    (26, 2), and `radii`, an array with shape (26,). The trusted evaluator
    independently checks finite values, containment, non-overlap, and computes
    the score. Do not report or hard-code a score.

    The original SimpleTES evaluation limit is 530 seconds. Use deterministic
    seeds for stochastic algorithms. Candidate dependencies belong in
    `evo/pyproject.toml`.
""")

STRATEGIES = (
    "Combine global stochastic exploration with deterministic local refinement and multiple restarts.",
    "Use nonlinear or constrained optimization from scipy or cvxpy.",
    "Use a substantially different geometric construction from the parent.",
    "Make the smallest evidence-driven change likely to improve the score.",
)


def build_evo_models() -> list[ModelSpec]:
    cerebras_provider = CerebrasProvider()
    openai_provider = OpenAIProvider()
    return [
        ModelSpec(
            model_id="gpt-oss-120b-medium",
            model=CerebrasModel("gpt-oss-120b", provider=cerebras_provider),
            settings=CerebrasModelSettings(openai_reasoning_effort="medium"),
            input_token_cost=0.35,
            output_token_cost=0.75,
            max_model_turns=10,
        ),
        ModelSpec(
            model_id="gpt-5.6-terra",
            model=OpenAIResponsesModel("gpt-5.6-terra", provider=openai_provider),
            settings=OpenAIResponsesModelSettings(
                openai_reasoning_effort="medium",
                openai_service_tier="flex",
            ),
            input_token_cost=1.0,
            output_token_cost=6.0,
            max_concurrent_leases=1,
            max_model_turns=4,
        ),
    ]


def get_adrevo_config() -> AdrevoConfig:
    return AdrevoConfig(
        num_agent_workers=5,
        task_sys_msg=SYSTEM_MSG,
        build_evo_models=build_evo_models,
        evolvable_files=(
            EvolvableFile("evo/main.py", "python"),
            EvolvableFile("evo/pyproject.toml", "toml"),
        ),
        strategies=STRATEGIES,
        pr_no_strategy=0.2,
        pr_strategies=(0.4, 0.2, 0.1, 0.1),
        max_cost=5.0,
        evaluator_timeout_sec=560,
    )
