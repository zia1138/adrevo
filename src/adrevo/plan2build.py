"""Prompts for the research-plan to evaluation-build workflow."""

import textwrap


# research plan prompt below ------------------------------------------------------------------------------

research_plan_prompt = textwrap.dedent("""\
Read the related work in `BACKGROUND.md`. If present, inspect the code in the current folder.
Do not perform any web searches. Use only the information provided in the folder.

Write a research plan to `RESEARCH_PLAN.md`.

The goal is to identify the highest-impact opportunity for improving existing
algorithms that can be investigated with a small, focused research environment.

Focus on identifying one key experiment and one primary measurement that would
demonstrate an advance, not proposing an improved algorithm.

Scientific impact is the priority, but implementation complexity and evaluation
cost must remain low enough to support rapid algorithm discovery.

Favor opportunities where incremental algorithmic innovations can produce
measurable improvements over an existing method, without trivial shortcuts
that artificially improve the measurement.

## Clarity and explanation

Write for a technically sophisticated reader who is unfamiliar with the
specific research area. The entire plan should be understandable without
requiring the reader to consult the related literature.

Throughout the plan:

- Explain unfamiliar concepts and terminology when first introduced.
- Provide intuition for important claims, methods, and measurements.
- Use simple, concrete examples to clarify ideas that are difficult to
  understand from abstract descriptions alone.
- Explain why each important detail matters to the scientific objective.
- Make the connections between the thesis, experiment, measurement, and
  baseline explicit.

Assume familiarity with general algorithms, mathematics, and computer science,
but not specialized knowledge of the research topic.

Favor clear, intuitive explanations over dense technical shorthand.
Preserve scientific precision, but provide additional explanation whenever
a concept would otherwise be difficult to understand quickly.

## Algorithm discovery thesis

Start with a section titled **Algorithm discovery thesis**.

Identify the most scientifically valuable limitation, tradeoff, or unsolved
problem in existing methods that presents an opportunity for algorithmic improvement.

State one clear thesis that:

- identifies the specific problem or limitation in existing methods,
- explains why it matters and what makes it scientifically important,
- describes the improvement that would constitute an advance,
- and identifies the conditions under which that improvement would matter.

Describe the desired improvement in terms of observable outcomes, not a
proposed algorithm, mechanism, or implementation.

Explain the underlying problem and why existing algorithms struggle with it.
Use a concrete example where helpful to make the opportunity intuitive.

Prioritize opportunities involving focused, self-contained algorithms rather
than improvements requiring large or complex software systems.

Choose an opportunity that can be investigated through one focused, feasible
experiment with a small implementation and inexpensive evaluations.

## Experiment and measurement

Identify the single most compelling experiment for determining whether a
hypothetical new algorithm represents an improvement over existing methods.

Derive the experiment directly from the algorithm discovery thesis.

1. Describe the experiment, the scientific question it addresses, and the
   conditions or problem instances under which it should be conducted.

   Explain how the experiment works conceptually, what is varied, what is
   held constant, and why the experiment provides evidence for the thesis.
   Use a concrete example to make the experimental setup easy to understand.

2. Under a **Key measurement** subsection, identify the single most
   informative quantitative metric for evaluating algorithmic improvement.

   Explain:
   - exactly what is measured and how the metric is calculated,
   - why the metric directly reflects progress toward the thesis,
   - how to interpret its values, including whether higher or lower is better,
   - how incremental improvements in the algorithm would affect the metric,
   - and what correctness requirements and experimental controls are needed
     to ensure that improvements are meaningful.

   Use a simple numerical example to illustrate the measurement and how
   different algorithms would compare.

   Prefer a measurement that provides a sensitive, informative signal across
   a range of algorithm quality, allowing gradual improvements to be detected.

   Avoid binary success criteria, easily saturated metrics, or measurements
   that only distinguish very large improvements.

   Where important tradeoffs exist, define appropriate constraints or
   supporting diagnostic metrics without introducing competing primary
   objectives.

3. Under a **Resistance to trivial solutions** subsection, critically examine
   whether the key measurement can be improved without genuine algorithmic progress.

   Identify plausible trivial, degenerate, or shortcut strategies that could
   achieve a favorable measurement while failing to solve the intended problem.

   Explain:
   - how these strategies might exploit the proposed measurement,
   - what correctness constraints or experimental conditions would rule them out,
   - and why improving the measurement would then represent meaningful progress.

   Consider shortcuts such as omitting difficult work, producing excessive or
   empty outputs, exploiting fixed test instances, or shifting computational
   costs into unmeasured resources where relevant.

   Prefer measurements that are difficult to game and remain informative
   across representative problem instances.

   If a trivial strategy can achieve a competitive result without satisfying
   the intended scientific objective, revise the measurement or experiment.

   Do not penalize a simple algorithm merely for being simple. A genuinely
   effective simple solution should receive a good measurement.

The experiment should directly test the central claim of the thesis.

Choose an experiment and key measurement that maximize scientific significance
while remaining simple to implement, reproducible, resistant to trivial
shortcuts, and inexpensive to evaluate repeatedly.

## Baseline method

Identify a single strong, self-contained algorithmic baseline for testing
the algorithm discovery thesis in the selected experiment.

Choose a baseline that:

- represents a credible and scientifically meaningful method from previous work,
- directly addresses the limitation or tradeoff identified in the thesis,
- can be implemented faithfully without reproducing a complex software system,
- has a small, well-defined algorithmic interface,
- and can be evaluated efficiently across many candidate implementations.

Prefer a focused algorithm over a complete application, framework, or
large software package.

Explain how the baseline works at a conceptual level, using a simple example
where helpful.

Explain why the baseline is the most informative practical comparison and
what performance or tradeoff it represents.

The baseline should establish a meaningful reference value for the key
measurement, against which incremental algorithmic improvements can be assessed.

## Guiding principles

The research plan should form a coherent scientific argument:

    Highest-impact tractable problem or opportunity
        -> One key experiment
            -> One primary measurement of algorithmic improvement
            -> Resistance to trivial solutions
        -> One strong, self-contained baseline

Focus on defining a valuable, measurable opportunity for algorithm discovery.

Do not propose new algorithms, solution mechanisms, architectures, or
implementation strategies. Discovering how to achieve the improvement is
a separate research activity.

Choose the experiment, measurement, and baseline specifically to test
the thesis, not merely to reproduce a standard benchmark.

Favor problems where the core algorithm can be implemented in a small amount
of readable code and evaluated quickly without substantial infrastructure.

Avoid research directions that require reproducing large software systems,
complex dependencies, extensive datasets, or expensive computational resources.

Prioritize opportunities with a sensitive, gradually improvable measurement
that rewards genuine algorithmic progress rather than benchmark exploitation.

Prioritize scientific significance subject to implementation simplicity,
evaluation efficiency, and feasibility.

Do not design the evaluation infrastructure or prescribe implementation details.
Those belong in a separate evaluation plan.

Be clear, self-contained, specific, and decisive. Provide enough explanation
and intuition for a technically sophisticated reader to understand the entire
plan without specialized background.

Select one high-impact opportunity, one key experiment, one primary measurement,
and one strong baseline rather than proposing multiple research directions,
experiments, or comparison methods.
""")

# eval plan prompt below ------------------------------------------------------------------------------

eval_plan_prompt = textwrap.dedent("""\
Read `RESEARCH_PLAN.md`, the related work in `BACKGROUND.md`, and, if present,
the code in the current folder. Do not perform any web searches. Use only the
information provided in the folder.

Write an evaluation plan to `EVAL_PLAN.md`.

Treat the algorithm discovery thesis, experiment, and baseline in
`RESEARCH_PLAN.md` as the scientific objectives. Do not replace them with
a different research direction or introduce additional experiments or baselines.

The goal is to produce a concrete implementation plan for the smallest
effective evaluation harness that can execute the experiment, measure the
proposed improvement, and support rapid discovery of improved algorithms.

Make the essential engineering decisions needed for a coding agent to
build and validate the environment.

Simplicity is a primary design requirement. Favor small, readable
implementations, minimal dependencies, and inexpensive repeated evaluations.

## Scoring system

Use the key measurement and correctness requirements in `RESEARCH_PLAN.md` to
define a single numeric `combined_score`, with higher values indicating better
algorithms.

The score must be computed independently for each implementation using only
its measured performance and fixed evaluation criteria. It must not depend
on the baseline's score or the performance of other implementations.

Specify:

- The exact primary and secondary metrics and how they are computed.
- The formula for `combined_score`, including any fixed normalization,
  weighting, or constraints.
- How incorrect, invalid, or failed implementations are scored.
- Why the score captures meaningful progress toward the research thesis.

Prefer a simple score derived directly from the key measurement. Use additional
metrics as constraints or diagnostics rather than introducing unnecessary
weighted objectives.

The scoring formula and all normalization constants must be fixed across
implementations and evaluation runs.

The score should provide a meaningful signal for incremental algorithmic
improvements without depending on relative rankings or baseline performance.

All primary and secondary metrics must be reported independently in
`results.json`, even when they are not included in `combined_score`.

Keep scoring simple, scientifically meaningful, and inexpensive to compute.

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

Use the simplest practical interface. Prefer files and command-line arguments
over custom APIs, services, or communication frameworks.

The interface must be identical for the initial algorithm and baseline,
regardless of their implementation language.

Algorithms must not compute their own evaluation metrics or `combined_score`.
They only produce outputs from the provided inputs.

The evaluator independently reads these outputs, validates correctness,
and computes all metrics.

New algorithms must be evaluable without modifying the evaluator.

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
5. Validates correctness and computes all evaluation metrics.
6. Writes `results.json` to the project root, alongside `evaluate.py`.

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
Prevent stale outputs from affecting subsequent evaluations.

The output contract is:

    {
        "correct": true,
        "error": null,
        "combined_score": 0.85,
        "accuracy": 0.92,
        "runtime_seconds": 1.42
    }

The metrics above are illustrative. Use the actual primary and secondary
metrics defined by the experiment.

- `correct` indicates whether the implementation passed validation.
- `error` describes any evaluation failure, or is null on success.
- `combined_score` is a finite numeric value, with higher values being better.
- Every primary and secondary metric must be included as a separate field.
- Metrics must use consistent names, definitions, and units across methods.
- Unavailable metrics should be reported as null.

Every invocation must produce a fresh `results.json` in the project root.
Failed evaluations must produce a valid result with a noncompetitive score,
never leave stale results from a previous run.

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
- The correctness checks and experimental conditions.
- How `evaluate.py` computes each metric and `combined_score`.
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
- A concise README explaining preparation, evaluation, and interpretation
  of results.

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
- All primary and secondary metrics are calculated and reported correctly.
- `combined_score` is calculated correctly.
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
- Evaluation: Validate outputs, compute metrics, and write results.json.

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

Treat `EVAL_PLAN.md` as authoritative for the architecture, interfaces,
algorithms, experimental conditions, measurements, and scoring.

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

Preserve the evaluation contract:

    uv run evaluate.py run <implementation_directory>

Each invocation must produce a fresh `results.json` in the project root,
containing `correct`, `error`, `combined_score`, and all primary and
secondary metrics specified in the plan.

Keep one-time data preparation separate from repeated evaluations.

Do not introduce additional experiments, baselines, or infrastructure
beyond what the plan requires.

## Scientific verification

Verify the evaluator independently rather than relying only on successful
end-to-end execution.

Establish that:

- Correctness checks accept valid outputs and reject invalid outputs.
- Primary and secondary metrics are calculated correctly.
- `combined_score` is calculated correctly from the specified measurements.
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
- All specified measurements are present and directly comparable.
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
- Evidence that the evaluator and scoring are correct.
- Any discrepancies, unresolved issues, or scientific limitations.

If completion is blocked, document the blocker and preserve the
verified partial implementation.

## Guiding principles

The goal is a small, understandable, scientifically trustworthy
research environment, not a general-purpose evaluation framework.

Prioritize correctness and reproducibility without unnecessary complexity.

Take ownership of implementation, debugging, and verification while
preserving the scientific requirements in `EVAL_PLAN.md`.

The deliverable is not merely working code. It is a verified research
environment that can reliably evaluate new algorithms.
""")    

