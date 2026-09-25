import textwrap

from adrevo.config import AdrevoConfig, EvolvableFile, ModelSpec
from pydantic_ai.models.cerebras import CerebrasModel, CerebrasModelSettings
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from pydantic_ai.providers.cerebras import CerebrasProvider
from pydantic_ai.providers.openai import OpenAIProvider


SYSTEM_MSG = textwrap.dedent("""\
    Construct a 29 by 29 matrix whose entries are exactly +1 or -1 and maximize
    the absolute value of its determinant.

    Running `evo/main.py` must write `hadamard.json` in its working directory
    with the JSON shape `{"matrix": [[...], ...]}`. The trusted evaluator checks
    the shape and entries, computes the determinant exactly with the Bareiss
    algorithm, and divides its absolute value by the same reference constant
    used by SimpleTES. Higher `combined_score` is better. Do not report or
    hard-code a score.

    Replace complete candidate files; there are no protected EVOLVE-BLOCK
    regions. The evaluation limit is 350 seconds. Use deterministic random
    seeds. Candidate dependencies belong in `evo/pyproject.toml`.
""")

STRATEGIES = (
    "Start from a structured combinatorial design and improve it with determinant-aware local search.",
    "Use fast floating-point screening followed by exact evaluation of promising matrices.",
    "Use a substantially different construction or optimization method from the parent.",
    "Make the smallest evidence-driven change likely to increase the score.",
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
        evaluator_timeout_sec=380,
    )
