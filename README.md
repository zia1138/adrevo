# adrevo <a href="https://zia1138.github.io/adrevo"><img src="docs/adrevo.png" align="right" height="209" alt="adrevo docs" /></a>

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

# Benchmarks

Below are benchmarks using results from [AdaEvolve: Adaptive LLM Driven Zeroth-Order Optimization](https://arxiv.org/abs/2602.20133"). For OpenEvolve, GEPA, ShinkaEvolve, and AdaEvolve the GPT-5 backbone results are shown from the AdaEvolve paper.

## Mathematical Optimization Results

| Strategy | Circle Packing ↑ | Circle Packing (Rect) ↑ | Heilbronn (Convex) ↑ | Heilbronn (Triangles) ↑ | MinMax Distance ↑ | Signal Processing ↑ |
|----------|-----------------:|------------------------:|---------------------:|------------------------:|------------------:|--------------------:|
| Human / SOTA | 2.634 | 2.364 | 0.0306 | 0.0360 | 0.2399 | – |
| AlphaEvolve | 2.635 | <span style="color:red"><strong>2.3658</strong></span> | 0.0309 | <span style="color:red"><strong>0.0365</strong></span> | 0.2398 | – |
| OpenEvolve | 2.541 | 2.276 | 0.027 | 0.028 | 0.2243 | 0.622 |
| GEPA | 2.628 | 2.354 | 0.027 | 0.032 | 0.2392 | 0.705 |
| ShinkaEvolve | 2.541 | 2.358 | 0.026 | 0.034 | 0.2398 | 0.533 |
| AdaEvolve | 2.63598308 | 2.361 | 0.029 | 0.036 | <span style="color:red"><strong>0.2404</strong></span> | 0.718 |
| **adrevo** | **2.63598308499572** | 2.3621 | <span style="color:red"><strong>0.03092</strong></span> | <span style="color:red"><strong>0.0365</strong></span> | 0.2401 | <span style="color:red"><strong>0.956</strong></span> |
| **Cost (adrevo)** | **$1.27** | **$1.81** | **$3.16** | **$4.61** | **$1.62** | **$2.61** |

## ADRS Benchmark Results 

| Strategy | Cloudcast ↓ (Best) | EPLB ↑ (Best) | Prism ↑ (Best) | LLM-SQL ↑ (Best) | TXN ↑ (Best) |
|----------|-------------------:|--------------:|---------------:|-----------------:|-------------:|
| Human / SOTA | 626.2 | 0.1265 | 21.89 | 0.692 | 2,725 |
| OpenEvolve | 729.8 | 0.1272 | 26.23 | 0.716 | 4,329 |
| GEPA | 645.7 | 0.1445 | 26.23 | 0.713 | 3,984 |
| Shinka | 812.7 | 0.1272 | 26.26 | 0.713 | 4,329 |
| AdaEvolve | 640.5 | 0.1453 | <span style="color:red"><strong>26.37</strong></span> | 0.775 | <span style="color:red"><strong>4,348</strong></span> |
| **adrevo** | <span style="color:red"><strong>618.07</strong></span> | <span style="color:red"><strong>0.1516</strong></span> | 25.26 | <span style="color:red"><strong>0.9893</strong></span> | 4,273.50 |
| **Cost (adrevo)** | **$3.20** | **$1.99** | **$0.22** | **$1.11** | **$1.46** |



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
