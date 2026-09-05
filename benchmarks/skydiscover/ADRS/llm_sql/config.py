import textwrap

from adrevo.config import AdrevoConfig, ModelSpec
from pydantic_ai.models.cerebras import CerebrasModel, CerebrasModelSettings
from pydantic_ai.providers.cerebras import CerebrasProvider
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from pydantic_ai.providers.openai import OpenAIProvider

SYSTEM_MSG = textwrap.dedent("""\
    You are an expert in data optimization and LLM prompt caching. Your task is to evolve
    the existing Evolved class to maximize prefix hit count (PHC) for efficient LLM prompt caching.

    Problem Context:
    - You are given a pandas DataFrame `df` with text data in rows and columns
    - The goal is to reorder columns to maximize prefix reuse when processing rows sequentially
    - Prefix reuse occurs when consecutive rows have matching values in the same column positions
    - This reduces LLM computation costs by reusing cached prefixes

    Objective:
    - Dual objective: (1) maximize prefix reuse across consecutive rows and (2) minimize
      end-to-end runtime of the algorithm.
    - Combined score: combined_score = 0.95 * average_hit_rate + 0.05 * (12 - min(12, runtime)) / 12

    Required API (DO NOT CHANGE):
    - You must keep the existing Evolved class structure and the reorder method signature:
      ```python
      class Evolved(Algorithm):
          def reorder(
              self,
              df: pd.DataFrame,
              early_stop: int = 0,
              row_stop: int = None,
              col_stop: int = None,
              col_merge: List[List[str]] = [],
              one_way_dep: List[Tuple[str, str]] = [],
              distinct_value_threshold: float = 0.8,
              parallel: bool = True,
          ) -> Tuple[pd.DataFrame, List[List[str]]]:
      ```

    The main entry point of the code is:
    `Evolved().reorder(df, early_stop, row_stop, col_stop, col_merge, ...)`
""")


#strategies = ("S1",
#              "S2")
#pr_no_strategy = 0.2
#pr_strategies = (0.5, 0.15, 0.05, 0.05, 0.05)


def build_evo_models() -> list[ModelSpec]:
    cerebras_provider = CerebrasProvider()
    gpt_oss_120b_medium = ModelSpec(
        model_id="gpt-oss-120b-medium",
        model=CerebrasModel("gpt-oss-120b", provider=cerebras_provider),
        settings=CerebrasModelSettings(openai_reasoning_effort="medium"),
        input_token_cost=0.35,
        output_token_cost=0.75,            
        max_concurrent_leases=1000,
        max_model_turns=10,        
    )
    openai_provider = OpenAIProvider()
    openai_gpt_5_6_terra = ModelSpec(
        model_id="gpt-5.6-terra",
        model=OpenAIResponsesModel("gpt-5.6-terra", provider=openai_provider),
        settings=OpenAIResponsesModelSettings(openai_reasoning_effort="medium", openai_service_tier="flex"),
        input_token_cost=1, # per 1M tokens
        output_token_cost=6, # per 1M tokens
        max_concurrent_leases=1,
        max_model_turns=4,
    )    
    return [
        gpt_oss_120b_medium,
        openai_gpt_5_6_terra,
    ]



def get_adrevo_config() -> AdrevoConfig:
    return AdrevoConfig(
        num_agent_workers=5,
        task_sys_msg=SYSTEM_MSG,
        build_evo_models=build_evo_models,
        model_wait_poll_sec=2.0,
        #strategies=strategies,
        #pr_no_strategy=pr_no_strategy,
        #pr_strategies=pr_strategies,
        max_cost=5.0,
        evaluator_timeout_sec=240,
    )
