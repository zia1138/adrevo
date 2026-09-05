import textwrap

from adrevo.config import AdrevoConfig, ModelSpec
from pydantic_ai.models.cerebras import CerebrasModel, CerebrasModelSettings
from pydantic_ai.providers.cerebras import CerebrasProvider
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from pydantic_ai.providers.openai import OpenAIProvider

SYSTEM_MSG = textwrap.dedent("""\
    You are an expert in database transaction optimization.
    Your task is to improve a scheduling function to find better schedules for
    transactional workloads made up of read and write operations to data items.
    There are conflicts between these transactions on items and reducing the delay
    of these conflicts will lead to schedules with lower makespan.

    **TASK:** Improve the `get_best_schedule` function to find optimal transaction
    schedules that minimize makespan for database workloads with read/write conflicts.

    **PROBLEM SPECIFICS:**
    - Input: JSON workload with transactions like "txn0":"w-17 r-5 w-3 r-4 ..."
    - Operations: Each transaction is a sequence of read (r-{key}) and write (w-{key})
      operations on data items
    - Conflicts: Read-write and write-write conflicts on the same key create dependencies
    - Goal: Find transaction ordering that minimizes total makespan

    **SEARCH SUGGESTIONS:**
    - Greedy: Try a greedy algorithm to iteratively pick the transaction that increases
      makespan the least.
    - Avoid only using heuristics like transaction length, number of writes, etc. because
      these do not correspond to the actual makespan of the schedule.

    The main entry point of the code is: `get_best_schedule(workload, num_seqs)`
    The top-level function is: `get_schedules()` which returns schedules for all workloads.
""")


strategies = ("You must propose an algorithmic approach that combines global stochastic exploration (for example, simulated annealing, evolutionary search, or basin hopping) with local refinement across multiple random restarts. Do not give only a theoretical bound or a purely exact-optimization method. Include: initialization, objective function, move/proposal mechanism, acceptance rule, restart strategy, refinement step, and stopping criteria.",
              "Use a different underlying algorithm from the parent program.",
              "Make your solution verbose adding comments and rationale for each step.",              
              "Make as minimal a change as possible to improve the parent program.")

pr_no_strategy = 0.35
pr_strategies = (0.4, 0.15, 0.05, 0.05)


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
        strategies=strategies,
        pr_no_strategy=pr_no_strategy,
        pr_strategies=pr_strategies,
        max_cost=5.0,
        evaluator_timeout_sec=240,
    )
