# adrevo

adrevo is an experimental system for agentic, AI-driven directed evolution of algorithms.
It is built on [ray](https://www.ray.io/), 
[pydantic-ai](https://github.com/pydantic/pydantic-ai), and [logfire](https://github.com/pydantic/logfire).

<span style="color:red">NOTE: This project is experimental and under active development.</span>.

**📖 [Documentation](https://zia1138.github.io/adrevo)**

# Installation

Clone the repo and setup a `uv` environment as follows:
```bash
git clone git@github.com:zia1138/adrevo.git
cd adrevo
git worktree add -b gh-pages site origin/gh-pages
uv sync
source .venv/bin/activate
```

Create an account and project called `adrevo` on [logfire](https://logfire.pydantic.dev/login). 
You will need to authenticate to logfire and select the `adrevo` project to log your runs. You can do this as follows:
```bash
logfire auth
logfire projects use adrevo
```
This will create a file `.logfire/logfire_credentials.json` with your logfire credentials and configuration. If you use a ray cluster, you need the logfire credentials on every node (copy `.logfire/` to all nodes or
or set `LOGFIRE_TOKEN` on all nodes).

# Quickstart Instructions

You can run the circle packing example using the config files in `examples/circle_packing`:
```bash
adrevo run examples/circle_packing
```

Runs periodically save a resumable checkpoint in their results directory. Resume
an interrupted run by supplying the same project and results base:

```bash
adrevo run examples/circle_packing --results-dir ./results
adrevo resume examples/circle_packing --results-dir ./results
```

`resume` restores completed programs, the search position, generation count, and
token-cost accounting. Work that was in flight at the checkpoint is discarded;
workers and model leases start fresh. The resumed config may use different models
or model costs; historical spend is carried forward while new token usage is
accounted for with the new configuration. Other configuration values may also be
changed when resuming. Total `max_generations` and `max_cost` limits include work
from before the resume.

The checkpoint files are:

```text
database_state.json       # program metadata and search state
evogen_state.json         # generation and cost accounting
<program_id>.zip          # executable project state for each program
```

Use `adrevo --help` to get all of the command line parameters. 

We use a config-as-code system where you initialize data classes in project config files to modify parameters.
Projects can keep one or more root-level config files named `config.py` or `config_*.py`.
If multiple config files are present, `adrevo run` prompts you to choose one unless you pass `--config`.
For folder runs, `--config` applies the same config filename inside each project directory.

Example root-level configs:

- `config.py`
- `config_fast.py`
- `config_cerebras.py`
- `config_gemini.py`

See [src/adrevo/config.py](src/adrevo/config.py) for the configuration dataclasses and validation helpers.


# Related Open Source Projects

Related projects use search, like adrevo. Some also incorporate reinforcement learning (RL).

## No RL 

* [AlphaEvolve](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/)
* [ShinkaEvolve](https://github.com/SakanaAI/ShinkaEvolve)
* [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve)
* [LLM4AD](https://github.com/Optima-CityU/llm4ad)
* [skydiscover](https://github.com/skydiscover-ai/skydiscover)
* [GigaEvo](https://github.com/AIRI-Institute/gigaevo-platform/tree/main)
* [station](https://github.com/dualverse-ai/station)
* [autoresearch](https://github.com/karpathy/autoresearch)
* [GEPA](https://github.com/gepa-ai/gepa)
* [CodeEvolve](https://github.com/inter-co/science-codeevolve)

## RL Incorporated

* [ThetaEvolve](https://github.com/ypwang61/ThetaEvolve)
* [TTT-Discover](https://github.com/test-time-training/discover)
