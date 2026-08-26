# adrevo

![adrevo](adrevo.png){align=right}

For the quickstart, custom-project overview, adaptive backtracking, and benchmark results, see the [README](https://github.com/zia1138/adrevo#readme). This page is the operational reference.

## Project contract

An adrevo project contains `main.py`, `evaluate.py`, and `config.py` (or a `config_*.py` variant).

- `main.py`: your initial algorithm; adrevo replaces this file.
- `evaluate.py`: validates `main.py` and writes `results.json`.
- `config.py`: defines `get_adrevo_config()` and `get_backend_config()`.

`results.json` must contain `correct` (boolean), `error` (string or `null`), and `combined_score` (number). Higher scores are better. The current execution backend invokes `evaluate.py` by name.

## Configuration

Configuration is Python. Construct the dataclasses and override only the defaults you need.

| Object | Important fields |
|---|---|
| `ModelSpec` | `model_id`, Pydantic AI `model`, `settings`, token costs, concurrency leases, model turns |
| `AdrevoConfig` | model builder, task prompt, workers, generations, cost limit, backtracking, strategies |
| `BackendConfig` | timeout, `uv`/`pixi`, immutable data directories |

`build_evo_models()` returns one or more ordered `ModelSpec`s. Adrevo tries the next model after a failed attempt; after the final model fails, it may backtrack. Use [Pydantic AI's model API](https://pydantic.dev/docs/ai/models/overview). See the [example config](https://github.com/zia1138/adrevo/blob/main/examples/circle_packing/config_openai.py) and [all config fields](https://github.com/zia1138/adrevo/blob/main/src/adrevo/config.py).

## Commands

| Command | Purpose | Key options |
|---|---|---|
| `adrevo run PROJECT` | Start one project | `--config`, `--results-dir`, `--verbose`, `--ray-address` |
| `adrevo resume PROJECT` | Continue a checkpointed project | required `--results-dir`; optional `--config` |
| `adrevo run-folder PROJECT...` | Run several projects | `--config`, `--results-dir`, `--max-concurrent` |

Use `adrevo COMMAND --help` for the complete option list.

## Operations

Resume with the same project and results base:

```bash
adrevo run my-project --results-dir ./results
adrevo resume my-project --results-dir ./results
```

The resumable directory is `./results/my-project/`; it contains `database_state.json`, `evogen_state.json`, and one `<program-id>.zip` per saved program. In-flight work is not resumed.

For several projects:

```bash
adrevo run-folder project-a project-b --results-dir ./results --max-concurrent 2
```

Project directory names must be distinct. To connect to Ray Client, pass `--ray-address ray://HOST:10001` (or `--ray-ip HOST --ray-port 10001`). Use `--verbose` for Logfire logs and Pydantic AI traces; on a cluster, provide `.logfire/` credentials or `LOGFIRE_TOKEN` on every node.

## Common failures

| Symptom | Check |
|---|---|
| No config found | Add `config.py` or `config_*.py`; use `--config` for several variants. |
| Initial run fails | Run `evaluate.py` locally; it must write a valid initial score. |
| Run ends early | Check generation/cost limits, evaluator timeout, and model credentials. |
| Results directory exists | Choose a new `--results-dir`; use `resume` only for a checkpointed run. |
