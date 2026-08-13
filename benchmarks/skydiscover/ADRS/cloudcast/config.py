import textwrap

from adrevo.config import AdrevoConfig, BackendConfig, ModelSpec
from pydantic_ai.models.cerebras import CerebrasModel, CerebrasModelSettings
from pydantic_ai.providers.cerebras import CerebrasProvider
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from pydantic_ai.providers.openai import OpenAIProvider

SYSTEM_MSG = textwrap.dedent("""\
    You are an expert in cloud infrastructure optimization. Your task is to evolve the
    search_algorithm(src, dsts, G, num_partitions) function to minimize overall
    data transfer cost across multiple clouds.
    Focus on efficiently broadcasting input data to multiple destination nodes by leveraging
    parallel paths and overlapping transfers across networks. Use the provided graph
    and the BroadCastTopology class to identify low-cost routes.
    Prioritize strategies that reduce redundant transfers, balance load across networks,
    and exploit multi-network topologies to minimize cost.

    The main entry point of the code is: `search_algorithm(src, dsts, G, num_partitions)`

    NOTE: search_algorithm() must return a BroadCastTopology instance.
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
                         data_dirs=("examples", "profiles"))
