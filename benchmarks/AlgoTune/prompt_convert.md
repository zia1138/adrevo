Using the provided task folder.

Convert this task into a minimal Adrevo project. Make the changes directly, preserving unrelated files and user edits.

Read `README.md`, `examples/circle_packing/`, and the task's source and `description.txt`. Follow the repository's current configuration and evaluator contracts.

Create this structure:

```text
<task>/
├── evaluate.py
├── config_cerebras.py
├── pyproject.toml
├── baseline/
│   ├── main.py
│   └── pyproject.toml
└── evo/
    ├── main.py
    └── pyproject.toml
```

Requirements:

- Replace task classes and registration with ordinary functions. Put `generate_problem` and `is_solution` in trusted `evaluate.py`, preserving generation behavior, correctness rules, tolerances, and necessary helpers.
- Put the original `solve(problem)` in `baseline/main.py`. The evaluator may import this trusted function if needed. Initialize `evo/main.py` with an independent copy of the original `solve(problem)` and its required helpers, plus working input/output scaffolding. The initial candidate must pass correctness checks; do not import the baseline from candidate code.
- Generate representative examples with fixed seeds. Cache them as `Path.home() / ".cache" / "adrevo" / "datasets" / "<unique-task-id>" / "examples.npz"`. Reuse the cache; use locking and atomic writes for concurrent workers. No cache versioning or environment-variable override. Regenerate the cache when example generation or serialization changes.
- Define a simple task-specific NPZ schema that preserves input/output semantics, including structured or variable-sized data where needed. Use `allow_pickle=False` and avoid object arrays. Document the schema briefly.
- Each `main.py` reads the absolute input NPZ path from `sys.argv[1]` and writes `output.npz` beside itself. Both programs receive the same cached examples.
- Adrevo runs `evaluate.py` with the temporary project root as its working directory. Use simple relative paths for local files; no `ROOT` constant or `__file__` path resolution is needed. Only examples belong in the shared cache; outputs and `results.json` belong in the copied project. Each subprocess runs with its own subfolder as `cwd`, so it can write `output.npz` directly.
- Run one warmup per program, then five measured runs each (configurable constants), alternating baseline/candidate order. Time the entire `uv run -qq python main.py <input_path>` subprocess with `time.perf_counter_ns()`, its subfolder as `cwd`, and a timeout. Discard warmup timings; use the median of measured runs for each program. Report `baseline_time` and `evo_time` in seconds and retain raw nanosecond samples. Include startup and I/O; exclude generation and trusted validation. Record measured elapsed times even on execution failure.
- Remove stale output before every run, including warmups, then load its output and validate every example with `is_solution`. Reject missing or malformed output. Write `results.json` with `correct`, `error`, `combined_score`, `baseline_time`, and `evo_time`. If both outputs pass, score `baseline_time / evo_time`; otherwise set `correct` to false and the score to zero, with a useful error. Use null for unmeasured timings. Handle execution failures and timeouts.
- Embed the complete `description.txt` contents as `SYSTEM_MSG` in `config_cerebras.py`, passed to `AdrevoConfig(task_sys_msg=SYSTEM_MSG, ...)`. Adapt the reference model configuration without task-specific strategies. Allow evolution only under `evo/` and give the evaluator enough time for all warmup and measured subprocesses plus validation.
- Give all three environments minimal `pyproject.toml` files with their required dependencies. Remove the obsolete task source and `description.txt` after transferring necessary behavior and text. Preserve attribution and required resources; ignore generated artifacts.

Keep the implementation short and readable. Avoid unnecessary abstractions, duplicated logic, and argument-parsing frameworks. Infer sensible task-specific defaults from the source; ask only when a faithful conversion is blocked.

Verify that both the baseline and initial candidate pass correctness checks, warmup exclusion, sample counts, median runtime-ratio scoring, failure handling, stale-output rejection, execution with a temporary project copy as the working directory, and concurrent cache creation/reuse. Test incorrect and failing candidate replacements in temporary copies; retain the original working solution in the delivered candidate. Its initial score should be approximately 1, allowing for timing noise. Do not launch a paid evolution run.

Summarize the changes, example choices, checks performed, and evaluator run command.
