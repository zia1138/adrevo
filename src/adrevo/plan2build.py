"""Prompts for the research-plan to evaluation-build workflow."""

import textwrap


# research plan prompt below ------------------------------------------------------------------------------

research_plan_prompt = textwrap.dedent("""\
Read the related work in `BACKGROUND.md`. If present, inspect the code in the current folder.
Do not perform any web searches. Use only the information provided in the folder.

Write a research plan to `RESEARCH_PLAN.md`.

Identify a scientifically valuable opportunity for algorithmic improvement and
design one focused experiment to measure progress toward that opportunity.

The plan must define one problem, experiment, key performance metric, and
baseline that together provide a meaningful foundation for algorithm discovery.

Do not propose a new algorithm. Define what is worth improving and how to
reliably determine whether an algorithm is better.

## Research plan

**Identify the opportunity.**

Select one important limitation, tradeoff, or unsolved problem in existing
algorithms. Explain the problem, why existing methods struggle with it, and
what improvement would constitute a scientific advance.

**Design the experiment and key performance metric.**

Propose one experiment and one quantitative metric that directly measure
progress toward the identified opportunity.

Choose the experiment and metric together to satisfy these requirements:

- **Scientific relevance:** Improving the metric represents genuine
  algorithmic progress toward the research objective.
- **Incremental feedback:** The metric distinguishes gradual improvements
  across a broad range of algorithm quality.
- **Intrinsic resistance to gaming:** The metric is inherently difficult
  to optimize through trivial, degenerate, or shortcut solutions.
- **Reliable evaluation:** The metric can be computed reproducibly and
  inexpensively across representative problem instances.
- **Standalone evaluation:** The metric must be computable independently
  for any algorithm using fixed experimental criteria. Its value must not
  depend on baseline performance, relative improvement, or comparisons with
  other methods.

Specify the experimental conditions, how the metric is calculated, and
how its values should be interpreted. Include a numerical example.

Explain why the metric's definition makes obvious shortcuts ineffective
without relying on additional checks.

**Design additional checks against metric gaming.**

Specify a small set of independent checks that detect whether an algorithm
achieves a favorable performance metric through invalid outputs or shortcuts
that do not accomplish the intended task.

For each check, explain what shortcut it detects, how it works, and what
constitutes passing or failing.

Focus on scientific correctness and the meaning of the performance metric.
Do not propose overfitting or generalization tests, source-code hashing,
file integrity checks, tamper detection, or restrictions on modifying
algorithm implementations. Generalization will be evaluated separately.

Apply the same checks to all algorithms. Keep them simple, reproducible,
and separate from the primary optimization metric.

**Select the baseline.**

Choose one strong, self-contained algorithm from previous work that provides
a credible reference for the experiment.

Explain how the baseline works, why it is the appropriate comparison, and
what improvement in the key performance metric would represent progress.

## Scope and presentation

Prioritize scientific impact while keeping the research environment small,
understandable, and inexpensive to evaluate repeatedly.

Favor focused algorithms over complex software systems, extensive datasets,
or expensive infrastructure.

Write for a technically sophisticated reader unfamiliar with the research
area. Explain unfamiliar concepts, provide intuition, and use concrete
examples where helpful.

Make concrete, justified research decisions rather than presenting a list
of possibilities or generic evaluation recommendations.

Organize the plan for clarity. Ensure the problem, experiment, metric,
additional checks, and baseline form a coherent scientific argument.

Do not propose new algorithms or design evaluation infrastructure.
Those belong in subsequent stages.

Be clear, specific, and decisive. Produce one research plan with one
scientific opportunity, one experiment, one robust key performance metric,
and one strong baseline.
""")

# eval plan prompt below ------------------------------------------------------------------------------

eval_plan_prompt = textwrap.dedent("""\
Read `RESEARCH_PLAN.md`, the related work in `BACKGROUND.md`, and, if present,
the code in the current folder. Do not perform any web searches. Use only the
information provided in the folder.

Write an evaluation plan to `EVAL_PLAN.md`.

Treat the algorithm discovery thesis, experiment, key performance metric,
anti-gaming checks, and baseline in `RESEARCH_PLAN.md` as the scientific
objectives. Do not replace them with a different research direction or
introduce additional experiments or baselines.

The goal is to produce a concrete implementation plan for the smallest
effective evaluation harness that can execute the experiment, measure the
proposed improvement, and support rapid discovery of improved algorithms.

Make the essential engineering decisions needed for a coding agent to
build and validate the environment.

Simplicity is a primary design requirement. Favor small, readable
implementations, minimal dependencies, and inexpensive repeated evaluations.

## Scoring system

Use the key performance metric defined in `RESEARCH_PLAN.md` directly as
the numeric `combined_score`. Do not transform, normalize, or combine it
with secondary metrics merely to change its scale or direction.

The metric may be positive, negative, or zero, and may be either maximized
or minimized. Specify its optimization direction clearly in `README.md`.

The score must be computed independently for each implementation using only
its measured performance and fixed evaluation criteria. It must not depend
on the baseline's score or the performance of other implementations.

Specify:

- The exact primary and secondary metrics and how they are computed.
- How the primary metric is reported as `combined_score`.
- The fixed, noncompetitive score assigned to invalid implementations or
  failed correctness and anti-gaming checks, respecting the metric's direction.
- Why the metric captures meaningful progress toward the research thesis.

Use secondary metrics as diagnostics rather than introducing additional
optimization objectives.

The metric definition and evaluation conditions must remain fixed across
implementations and evaluation runs.

The metric should provide a meaningful signal for incremental algorithmic
improvements without depending on relative rankings or baseline performance.

All primary and secondary metrics must be reported independently in
`results.json`, even when they are not included in `combined_score`.

Keep scoring simple, scientifically meaningful, and inexpensive to compute.

## Anti-gaming checks

Specify how to implement the anti-gaming checks in `RESEARCH_PLAN.md`.
Preserve their scientific intent and make the engineering decisions needed
to execute them reliably.

Define the test inputs, procedures, pass/fail criteria, and any required
reference data or thresholds.

Apply the checks consistently across algorithms. Failed anti-gaming checks
must result in a noncompetitive `combined_score`, regardless of the primary
metric.

Report check outcomes separately in `results.json`.

Keep the checks reproducible and inexpensive. Specify how to verify that
they detect their intended failure modes.

Focus on scientific correctness and shortcuts that undermine the primary
metric. Do not design overfitting or generalization tests, which are handled
separately. Do not introduce source-code hashing, file integrity monitoring,
tamper detection, or restrictions on modifying algorithm implementations.

## Preparation

Design `prepare.py` to perform one-time preparation:

    uv run prepare.py

Preparation may include downloading or generating datasets, preprocessing
inputs, or constructing reference data.

Use Python for orchestration and lightweight operations. When preparation
requires substantial computation, use separate source files implemented
in C++, Rust, CUDA, or another suitable language.

`prepare.py` should directly compile and execute these implementations
when needed, without additional orchestration scripts.

Store prepared resources in `data/` and reuse them across evaluations.

Preparation must not be repeated by `evaluate.py`.

Specify the exact preparation steps, implementation languages, source files,
dependencies, and resulting data files.

Keep preparation simple, reproducible, and easy to understand.

## Algorithm interface

Define a simple, common interface between the evaluator and every algorithm.

Each algorithm must:

1. Accept experimental inputs in a specified format.
2. Execute independently using its own implementation and dependencies.
3. Write its outputs to a file in a specified format.

Specify the exact input and output formats, invocation method, and file paths.

Design the algorithm interface to process experimental instances in large
batches, preferably all instances in a single invocation. Avoid repeated
calls from Python into compiled code for individual instances. The algorithm
should process the batch internally and write the results to an output file.

Use the simplest practical interface. Prefer files and command-line arguments
over custom APIs, services, or communication frameworks.

The interface must be identical for the initial algorithm and baseline,
regardless of their implementation language.

Algorithms must not compute their own evaluation metrics or `combined_score`.
They only produce outputs from the provided inputs.

The evaluator independently reads these outputs, validates correctness,
runs anti-gaming checks, and computes all metrics.

New algorithms must be evaluable without modifying the evaluator.
Algorithm implementations must remain freely modifiable and replaceable.
Preserve the evaluation interface and scientific criteria, not the
implementation code.

Choose implementation languages based on the experiment. Do not force
algorithms into Python when compiled or GPU-accelerated implementations
would be more appropriate.

Each algorithm directory should contain its source code and any dependencies
or build configuration genuinely required by its implementation.

Keep algorithm implementations focused, self-contained, and easy to modify.

## Evaluation interface

The evaluator uses a Typer CLI:

    uv run python evaluate.py run <implementation_directory> [--data-dir <data_directory>]

For example:

    uv run python evaluate.py run new_algorithm
    uv run python evaluate.py run baseline_method

`--data-dir` identifies the directory containing prepared experimental data.
It defaults to `data/` alongside `evaluate.py`, but callers may provide an
explicit path to prepared data.

Given an implementation directory, the evaluator:

1. Loads the prepared experimental inputs from `data/`.
2. Builds or prepares the algorithm's execution environment if needed.
3. Executes the algorithm with the required inputs and output file path.
4. Reads the algorithm's output file.
5. Validates correctness and runs anti-gaming checks.
6. Computes all evaluation metrics and `combined_score`.
7. Writes `results.json` to the project root, alongside `evaluate.py`.

Use Python to manage the evaluation lifecycle, including compilation,
execution, output collection, validation, and scoring.

Keep straightforward validation and metric calculations in Python.
If a measurement requires substantial computation, implement that computation
in a separate source file and invoke it from `evaluate.py`.

Do not embed computational kernels, native source code, or complex
algorithm implementations directly in `evaluate.py`.

For compiled implementations, invoke compilers and executables directly
using Python subprocesses or simple bindings. Do not introduce separate
build or execution scripts.

Keep compilation and setup overhead separate from measured algorithm
execution time unless the experiment explicitly requires otherwise.

Keep intermediate algorithm outputs separate from evaluation results.
Remove previous intermediate outputs before each evaluation.

The output contract is:

    {
        "correct": true,
        "error": null,
        "combined_score": 1.42,
        "runtime_seconds": 1.42,
        "accuracy": 0.92
    }

The metrics above are illustrative. In this example, runtime is the primary
metric and lower values are better. Use the actual metrics defined by the
experiment and add fields for anti-gaming check results.

- `correct` indicates whether mandatory validation checks passed.
- `error` describes any evaluation failure, or is null on success.
- `combined_score` is the primary metric's value, with its optimization
  direction documented in `README.md`.
- Every primary and secondary metric must be included as a separate field.
- Report anti-gaming check results as separate Boolean fields, or null when
  a result is unavailable.
- Failed or incomplete mandatory checks set `correct` to false and produce
  the specified noncompetitive `combined_score`.
- Metrics must use consistent names, definitions, and units across methods.
- Unavailable metrics should be reported as null.

Every invocation must produce a fresh `results.json` in the project root.
Failed evaluations must produce a valid result with a finite, noncompetitive
score, never leave stale results from a previous run.

All algorithms must use identical evaluation criteria and comparable
experimental conditions.

When measuring performance, ensure fair comparisons through consistent
hardware, compiler settings, resource limits, and measurement procedures
where applicable.

Keep evaluation execution simple and inexpensive. Do not design for
distributed execution, parallel evaluation, or complex resource management
unless the experiment specifically requires them.

## Implementation design

Specify the concrete implementation design:

- The files to build and their responsibilities.
- The one-time preparation steps and generated data.
- The common algorithm input/output interface and execution command.
- The selected baseline algorithm and how to implement it faithfully.
- The implementation languages and dependencies for preparation and algorithms.
- Which computational components require separate high-performance source files.
- How Python will compile and invoke those components when necessary.
- The correctness checks, anti-gaming checks, and experimental conditions.
- How `evaluate.py` computes the primary and secondary metrics.
- How failures and invalid outputs are handled.

Choose the simplest implementation that satisfies the scientific requirements.

Prefer local execution and lightweight dependencies. Use C++, Rust, CUDA,
or other high-performance technologies when justified by the experiment.

Keep computational implementations in ordinary, independently readable
source files. Keep Python orchestration concise and separate from
performance-critical code.

Use Docker, Modal, or other infrastructure only when necessary. Do not add
general-purpose support for technologies that are not required.

Make concrete engineering choices rather than presenting multiple
architectural alternatives.

## Code clarity and documentation

Plan for code that a technically sophisticated reader unfamiliar with
the research area can understand and modify.

Require:

- Small, readable functions with descriptive names and clear interfaces.
- Comments explaining non-obvious algorithms, mathematics, and assumptions.
- Docstrings or equivalent documentation for important functions.
- Clear explanations of metric calculations and scientific reasoning.
- A concise README explaining preparation, evaluation, the primary metric,
  whether it should be maximized or minimized, and interpretation of results.

Keep orchestration, computational implementations, and evaluation logic
clearly separated so each can be understood independently.

Comments should explain why something is done and how difficult parts work,
rather than restating obvious code.

Avoid unnecessary abstractions, excessive boilerplate, and documentation
that duplicates the implementation.

## Implementation plan

Provide a concrete sequence of steps for building the environment.

For each step, specify:

- What needs to be implemented.
- The essential implementation details.
- How to verify that it works.

Start with the smallest working end-to-end evaluation, then complete
the baseline, measurements, and validation.

Define the commands and expected results needed to verify that:

- `prepare.py` correctly prepares the required data.
- Any native preparation or evaluation components build and execute correctly.
- The baseline and initial algorithm accept the same inputs and produce
  valid output files.
- The evaluator correctly builds and executes the selected implementation.
- The evaluator correctly validates algorithm outputs.
- The anti-gaming checks detect their intended failure modes.
- Failed mandatory checks produce a noncompetitive score.
- All primary and secondary metrics are calculated and reported correctly.
- `combined_score` equals the primary metric for valid evaluations.
- Both algorithms produce comparable, reproducible results.
- Each evaluation writes a fresh `results.json` to the project root.
- Evaluations are inexpensive enough for repeated algorithm discovery.

Use focused tests and simple verification procedures. Do not introduce
elaborate testing infrastructure unless necessary.

The implementation plan should be sufficiently concrete for a coding agent
to execute without making major scientific or architectural decisions.

## Guiding principles

Prioritize scientific validity, simplicity, readability, and reproducibility.

The environment should consist of the smallest amount of code needed to
faithfully execute the experiment and measure algorithmic improvements.

Maintain a clear separation between:

- Preparation: Generate shared experimental data once.
- Algorithms: Transform experimental inputs into output files.
- Evaluation: Validate outputs, run anti-gaming checks, compute metrics,
  and write `results.json`.

Use Python in `prepare.py` and `evaluate.py` for orchestration and
straightforward computations.

Keep substantial computational work in separate source files, using
the most appropriate implementation language.

Do not inline high-performance kernels or native implementations in
the Python orchestration files.

Standardize the algorithm interface and evaluation results, not the
internal execution mechanism.

Avoid additional orchestration scripts, frameworks, infrastructure,
and unnecessary abstractions.

Prefer a straightforward implementation that is easy to understand and
debug over a more general or sophisticated architecture.

Preserve the scientific objectives of `RESEARCH_PLAN.md`. If a requirement
cannot be implemented faithfully within a small research environment,
explain the limitation and the smallest scientifically defensible alternative.

Do not implement the environment. Produce only `EVAL_PLAN.md`.

Be concise, specific, and decisive. The resulting plan should serve directly
as the specification for the build step.
""")

# build plan prompt below ------------------------------------------------------------------------------

build_prompt = textwrap.dedent("""\
Build a complete, scientifically trustworthy research evaluation
environment following the specification in `EVAL_PLAN.md`.

Read `EVAL_PLAN.md`, `RESEARCH_PLAN.md`, and inspect the existing code
before making changes.

Treat `RESEARCH_PLAN.md` as authoritative for the scientific objectives
and `EVAL_PLAN.md` as authoritative for their implementation, including
architecture, interfaces, experimental conditions, validation, and scoring.

The goal is to implement and verify the smallest working environment
capable of executing the selected experiment and reliably measuring
algorithmic improvements.

Prioritize scientific correctness, simplicity, readability, and
inexpensive repeated evaluations.

## Development process

Follow the implementation plan in `EVAL_PLAN.md`.

Work autonomously through implementation, testing, debugging, and
experimental verification.

- Start with the smallest functioning end-to-end evaluation.
- Implement incrementally, using independently verifiable milestones.
- Determine how to verify each substantial component before implementing it.
- Run relevant tests and verification commands after each milestone.
- Diagnose and resolve failures before proceeding.
- Maintain a working system throughout development.
- Track completed work, verification evidence, and unresolved issues.

Exercise independent engineering judgment over implementation details,
but do not redefine the experiment, baseline, interfaces, or scoring.

If the plan contains inconsistencies or infeasible requirements, document
them and make only the smallest scientifically defensible adjustments.
Do not silently change the scientific objectives.

## Implementation requirements

Implement the architecture and interfaces specified in `EVAL_PLAN.md`.

Keep the implementation as simple as possible:

- Use Python in `prepare.py` and `evaluate.py` for all orchestration.
- Allow algorithms to use the programming languages specified in the plan.
- Avoid unnecessary scripts, frameworks, dependencies, and abstractions.
- Prefer small functions, straightforward control flow, and readable code.
- Use descriptive names and meaningful comments explaining non-obvious
  algorithms, mathematical operations, and scientific reasoning.
- Document important interfaces, assumptions, and metric calculations.

Implement the specified file-based boundary between algorithms and the
evaluator. Algorithms produce output files; the evaluator independently
validates those outputs and computes the metrics.

Implement the anti-gaming checks specified in `EVAL_PLAN.md`.
Report their results separately in `results.json`. Failed mandatory
checks must produce the specified noncompetitive `combined_score`.

Preserve the evaluation contract:

    uv run python evaluate.py run <implementation_directory> [--data-dir <data_directory>]

Each invocation must produce a fresh `results.json` in the project root,
containing `correct`, `error`, `combined_score`, all primary and secondary
metrics, and anti-gaming check results specified in the plan.

Keep one-time data preparation separate from repeated evaluations.

Do not introduce additional experiments, baselines, or infrastructure
beyond what the plan requires.

## Scientific verification

Verify the evaluator independently rather than relying only on successful
end-to-end execution.

Establish that:

- Correctness checks accept valid outputs and reject invalid outputs.
- Anti-gaming checks accept valid behavior and detect their intended
  failure modes, using positive and negative test cases.
- Failed anti-gaming checks produce the specified noncompetitive score.
- Primary and secondary metrics are calculated correctly.
- `combined_score` is calculated correctly from the specified metrics.
- `combined_score` is computed independently for each algorithm without
  referencing baseline performance or other implementations.
- Invalid or failed implementations receive noncompetitive scores.
- The baseline faithfully implements the selected algorithm.
- All implementations are evaluated under fair, comparable conditions.
- Measurements are reproducible within appropriate tolerances.

Use known results, simple reference cases, and positive and negative
controls where appropriate.

Investigate unexpected results rather than modifying evaluation criteria
to accommodate them.

Ensure the evaluation measures genuine algorithmic differences rather
than implementation artifacts or weaknesses in the evaluator.

## End-to-end validation

Run the complete evaluation from a clean state.

Verify that:

- `prepare.py` produces the required reusable experimental data.
- The baseline and `new_algorithm/` execute successfully.
- Both follow the same algorithm input/output contract.
- Each evaluation produces a valid root-level `results.json`.
- All specified metrics and anti-gaming check results are present.
- New implementation directories can be evaluated without modifying
  the evaluator.
- Evaluation time and resource usage are practical for repeated runs.

The initial implementation may use the baseline as its starting point.
Do not invent a new algorithm as part of building the environment.

Document the commands required to reproduce the experiment and interpret
the results.

## Completion criteria

The work is complete when the environment specified in `EVAL_PLAN.md`
has been implemented and independently verified.

Report:

- What was implemented.
- The commands used for verification.
- The measured results for the baseline and initial algorithm.
- Evidence that the evaluator, scoring, and anti-gaming checks are correct.
- Any discrepancies, unresolved issues, or scientific limitations.

If completion is blocked, document the blocker and preserve the
verified partial implementation.

## Guiding principles

The goal is a small, understandable, scientifically trustworthy
research environment, not a general-purpose evaluation framework.

Prioritize correctness and reproducibility without unnecessary complexity.

Take ownership of implementation, debugging, and verification while
preserving the scientific requirements in `RESEARCH_PLAN.md` and the
engineering specification in `EVAL_PLAN.md`.

The deliverable is not merely working code. It is a verified research
environment that can reliably evaluate new algorithms.
""")    

