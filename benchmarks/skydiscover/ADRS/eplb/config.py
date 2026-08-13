import textwrap

from adrevo.config import AdrevoConfig, BackendConfig, ModelSpec
from pydantic_ai.models.cerebras import CerebrasModel, CerebrasModelSettings
from pydantic_ai.providers.cerebras import CerebrasProvider
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from pydantic_ai.providers.openai import OpenAIProvider

SYSTEM_MSG = textwrap.dedent("""\
    You are an expert programmer specializing in optimization algorithms. Your task
    is to improve the Mixture-of-Expert models Expert Parallelism Load Balancer
    (MoE EPLB) expert rearrangement algorithm.

    This algorithm will take the load metrics recorded by the vLLM server, and
    rearrange the experts to balance the load. It can make replicas of some experts
    to achieve better load balancing.

    Your goal will be two-fold:
    1. Improve the algorithm to achieve better load balancing; while
    2. Improve the algorithm to be more efficient, i.e. reduce the execution time
       of the algorithm itself, since perfect load balancing is NP-hard.

    The current algorithm is implemented in the `rebalance_experts` function.

    The main entry point of the code is:
    `rebalance_experts(weight, num_replicas, num_groups, num_nodes, num_gpus)`
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
    )


def get_backend_config() -> BackendConfig:
    return BackendConfig(timeout_sec=120, 
                         data_dirs = ("data",))
