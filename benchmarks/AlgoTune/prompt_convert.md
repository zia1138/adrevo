Using the provided task folder.

Convert this task into a minimal Adrevo project that evaluates empirical runtime scaling with problem size. Make the changes directly, preserving unrelated files and user edits.

Read `README.md`, `examples/circle_packing/`, and the task's source and `description.txt`. Use `benchmarks/AlgoTune/matrix_multiplication/evaluate.py` as the scaling reference. Follow the repository's current configuration and evaluator contracts.

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
- Choose five geometrically increasing problem sizes (prefer doubling), appropriate to the task and resource budget. Use the same fixed seeds and number of instances at every size. Choose sizes where computation is substantial relative to startup; preserve the generator's semantics and document what its size parameter controls.
- Cache one NPZ per size under `Path.home() / ".cache" / "adrevo" / "datasets" / "<unique-task-id>"`, with size and seeds in the filename. Reuse the cache; use locking and atomic writes for concurrent workers. No cache versioning or environment-variable override. Clear affected caches when generation or serialization changes.
- Define a simple task-specific NPZ schema that preserves input/output semantics, including structured or variable-sized data where needed. Use `allow_pickle=False` and avoid object arrays. Document the schema briefly.
- Each `main.py` reads the absolute input NPZ path from `sys.argv[1]` and writes `output.npz` beside itself. Both programs receive the same cached examples for the size being measured.
- Adrevo runs `evaluate.py` with the temporary project root as its working directory. Use simple relative paths for local files; no `ROOT` constant or `__file__` path resolution is needed. Only examples belong in the shared cache; outputs and `results.json` belong in the copied project. Each subprocess runs with its own subfolder as `cwd`, so it can write `output.npz` directly.
- At each size, run one warmup per program, then five measured runs each (configurable constants), alternating baseline/candidate order. Time the entire `uv run -qq python main.py <input_path>` subprocess with `time.perf_counter_ns()`, its subfolder as `cwd`, and a timeout. Discard warmup timings; use the median batch runtime for each program at that size. Time sizes separately; do not combine all sizes into a single subprocess. Report `baseline_time` and `evo_time` in seconds and retain raw nanosecond samples. Include startup and I/O; exclude generation and trusted validation. Record measured elapsed times even on execution failure.
- Remove stale output before every run, including warmups, then load its output and validate every example with `is_solution`. Reject missing, malformed, or incorrect output.
- Fit a straight line to `log(runtime)` versus `log(problem size)` for each program using its per-size medians: `alpha = float(np.polyfit(np.log(sizes), np.log(times), 1)[0])`. Set `combined_score = baseline_alpha - evo_alpha`. Positive means better empirical scaling; negative scores are valid. Do not clip the difference or replace it with a runtime ratio. Uniform constant-factor speedups should leave the ideal score unchanged.
- Write `results.json` with `correct`, `error`, `combined_score`, `baseline_alpha`, `evo_alpha`, and `per_size`. Each per-size entry contains `n`, median `baseline_time`/`evo_time` in seconds, and raw `baseline_times_ns`/`evo_times_ns`. Set `correct` true only if every output passes. On failure or timeout, write false, a zero score, and a useful error; keep available measurements and use null for unmeasured values.
- Embed the complete `description.txt` contents as `SYSTEM_MSG` in `config_cerebras.py`, passed to `AdrevoConfig(task_sys_msg=SYSTEM_MSG, ...)`. Append a concise explanation of the scaling objective to `SYSTEM_MSG`, including that artificially slowing small cases or caching answers to manipulate the score is not allowed. Adapt the reference model configuration without task-specific strategies. Allow evolution only under `evo/` and give the evaluator enough time for all sizes, warmups, measured subprocesses, and validation.
- Give all three environments minimal `pyproject.toml` files with their required dependencies. Remove the obsolete task source and `description.txt` after transferring necessary behavior and text. Preserve attribution and required resources; ignore generated artifacts.

Keep the implementation short and readable. Avoid unnecessary abstractions, duplicated logic, argument-parsing frameworks, and defensive checks on trusted configuration or generated inputs. Keep correctness checks on candidate outputs. Do not calculate unused statistics such as fit intercepts, R², or per-size speedups. Infer sensible task-specific defaults from the source; ask only when a faithful conversion is blocked.

Verify that both the baseline and initial candidate pass correctness checks, warmup exclusion, sample counts, per-size median timing and signed slope-difference scoring, failure handling, stale-output rejection, execution with a temporary project copy as the working directory, and concurrent cache creation/reuse. Test incorrect and failing candidate replacements in temporary copies; retain the original working solution in the delivered candidate. Its initial score should be approximately zero, allowing for timing noise. Check the slope calculation with synthetic power-law timings and verify that constant-factor changes do not alter the slope. Do not launch a paid evolution run.

Summarize the changes, size/seed choices, checks performed, and evaluator run command. Describe the score as empirical scaling over the measured range, not proof of asymptotic complexity; report if startup or I/O dominates the chosen sizes.
