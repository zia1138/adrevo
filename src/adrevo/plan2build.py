"""Prompts for the research-plan to evaluation-build workflow."""

import textwrap


# research plan prompt below ------------------------------------------------------------------------------

research_plan_prompt = textwrap.dedent("""\
Read the related work in `BACKGROUND.md`. If present, inspect the code in the current folder.
Do not perform any web searches. Use only the information provided in the folder.

Write a research plan to `RESEARCH_PLAN.md`.

The goal is to identify the highest-impact opportunity for improving existing
algorithms that can be investigated with a small, focused research environment.

Focus on identifying one decisive experiment and its measurements that would
demonstrate an advance, not proposing an improved algorithm.

Scientific impact is the priority, but implementation complexity and evaluation
cost must remain low enough to support rapid algorithm discovery.

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
- Make the connections between the thesis, experiment, measurements, and
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

## Experiment and measurements

Identify the single most compelling experiment for determining whether a
hypothetical new algorithm represents an improvement over existing methods.

Derive the experiment directly from the algorithm discovery thesis.

1. Describe the experiment, the scientific question it addresses, and the
   conditions or problem instances under which it should be conducted.

   Explain how the experiment works conceptually, what is varied, what is
   held constant, and why the experiment provides evidence for the thesis.
   Use a concrete example to make the experimental setup easy to understand.

2. Under a **Measurements** subsection, specify:
   - exactly what must be measured and how each metric is defined,
   - correctness requirements and important performance tradeoffs,
   - what must be controlled to ensure fair comparisons.

   For each important measurement, explain:
   - what the metric represents and why it matters,
   - how it is calculated or obtained,
   - how to interpret its values, including whether higher or lower is better,
   - and how changes in the metric relate to the proposed improvement.

   Provide simple numerical or conceptual examples where they help clarify
   the measurement. Explain relationships and tradeoffs between metrics,
   rather than merely listing their definitions.

The experiment should directly test the central claim of the thesis.
Where multiple objectives matter, specify the relevant tradeoffs.

Choose an experiment that maximizes scientific significance while remaining
simple to implement, reproducible, and inexpensive to evaluate repeatedly.

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

## Guiding principles

The research plan should form a coherent scientific argument:

    Highest-impact tractable problem or opportunity
        -> One decisive experiment
            -> Measurements required to demonstrate improvement
        -> One strong, self-contained baseline

Focus on defining a valuable, measurable opportunity for algorithm discovery.

Do not propose new algorithms, solution mechanisms, architectures, or
implementation strategies. Discovering how to achieve the improvement is
a separate research activity.

Choose the experiment, measurements, and baseline specifically to test
the thesis, not merely to reproduce a standard benchmark.

Favor problems where the core algorithm can be implemented in a small amount
of readable code and evaluated quickly without substantial infrastructure.

Avoid research directions that require reproducing large software systems,
complex dependencies, extensive datasets, or expensive computational resources.

Prioritize scientific significance subject to implementation simplicity,
evaluation efficiency, and feasibility.

Do not design the evaluation infrastructure or prescribe implementation details.
Those belong in a separate evaluation plan.

Be clear, self-contained, specific, and decisive. Provide enough explanation
and intuition for a technically sophisticated reader to understand the entire
plan without specialized background.

Select one high-impact opportunity, one compelling experiment, and one strong
baseline rather than proposing multiple research directions, experiments,
or comparison methods.
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

Make the scientific and engineering decisions needed for a coding agent to
build and validate the complete environment.

Prioritize small, self-contained implementations and inexpensive repeated
evaluations. Avoid unnecessary software complexity and infrastructure.

## Scoring system

Use the measurements and correctness requirements in `RESEARCH_PLAN.md` to
define a single numeric `combined_score`, with higher values indicating better
algorithms.

Specify:

- The exact primary and secondary metrics and how they are computed.
- The formula for `combined_score`, including normalization or weighting.
- How incorrect, invalid, or failed implementations are scored.
- Why the score captures meaningful progress toward the research thesis.

The score must distinguish genuine algorithmic improvements from artifacts
while preserving the important tradeoffs identified in the experiment.

All primary and secondary metrics must be computed and reported independently
in `results.json`, even when they are not included in `combined_score`.

Keep scoring computationally inexpensive and suitable for repeated evaluations
during algorithm discovery.

## What needs to be built

Design the smallest evaluation harness capable of executing the experiment
and producing the measurements specified in `RESEARCH_PLAN.md`.

Use a simple, flat project structure:

    prepare.py
    evaluate.py
    pyproject.toml
    data/
    new_algorithm/
    baseline_method/

Replace baseline_method with the name of the selected baseline algorithm.

- `prepare.py` performs one-time preparation of experimental data and resources.
- `evaluate.py` executes and evaluates any implementation directory.
- `pyproject.toml` defines the uv environment for preparation and evaluation.
- `data/` contains shared experimental data and configurations.
- `new_algorithm/` contains the initial algorithm intended for improvement.
- The baseline directory contains an independent implementation of the
  selected comparison algorithm.

The initial algorithm may use the baseline implementation as its starting
point. Do not design a new algorithm as part of the evaluation environment.

Prefer concise, readable implementations with minimal dependencies.
Avoid reproducing large software systems when the essential algorithm
can be implemented independently.

## Preparation

Design `prepare.py` to perform one-time preparation when needed:

    uv run prepare.py

Preparation may include downloading or generating datasets, preprocessing
inputs, constructing reference data, or preparing other shared resources.

Prepared resources must be stored in `data/` and reused across evaluations.

Preparation must be separate from evaluation and must not be repeated for
each algorithm or evaluation run.

Specify the exact preparation steps, generated files, and procedures needed
to ensure reproducibility.

## Algorithm interface

Define a simple, common interface between the evaluator and every algorithm.

Each algorithm must:

1. Accept experimental inputs in a specified format.
2. Execute independently using its own implementation and dependencies.
3. Write its outputs to a file in a specified format.

Specify the exact input and output formats, invocation method, and file paths.

The interface must be identical for the initial algorithm and baseline.

Algorithms must not compute their own evaluation metrics or `combined_score`.
They only produce outputs from the provided inputs.

The evaluator must independently read these output files, validate correctness,
and compute all metrics.

Keep algorithms independent of evaluation logic, validation rules, and scoring.
New algorithms should be evaluable without modifying `evaluate.py`.

The interface should support different programming languages and execution
environments without introducing unnecessary abstractions.

## Evaluation interface

The evaluator uses a Typer CLI:

    uv run evaluate.py run <implementation_directory>

For example:

    uv run evaluate.py run new_algorithm
    uv run evaluate.py run baseline_method

Given an implementation directory, the evaluator:

1. Loads the prepared experimental inputs from `data/`.
2. Prepares the algorithm's execution environment if needed.
3. Executes the algorithm with the required inputs and output file path.
4. Reads the algorithm's output file.
5. Validates correctness and computes all evaluation metrics.
6. Writes `results.json` to the project root, alongside `evaluate.py`.

The evaluator must not repeat the one-time preparation performed by
`prepare.py`.

Algorithm outputs and intermediate artifacts must remain separate from
evaluation results. Use fresh output paths to prevent stale or overlapping
results.

The output contract is:

    {
        "correct": true,
        "error": null,
        "combined_score": 0.85,
        "accuracy": 0.92,
        "runtime_seconds": 1.42,
        "memory_mb": 128.5
    }

The metrics above are illustrative. Use the actual primary and secondary
metrics defined by the experiment.

- `correct` indicates whether the implementation passed validation.
- `error` describes any evaluation failure, or is null on success.
- `combined_score` is a finite numeric value, with higher values being better.
- Every primary and secondary metric must be included as a separate field.
- Metrics must use consistent names, definitions, and units across methods.
- Unavailable metrics should be reported as null rather than misleading values.

Every invocation must produce a fresh `results.json` in the project root,
regardless of which implementation directory is evaluated. Failed evaluations
must produce a valid result with a noncompetitive score, never leave stale
results from a previous run.

All algorithms must be evaluated using the same criteria and comparable
experimental conditions.

The evaluation lifecycle should be efficient enough to execute repeatedly
across many candidate algorithms. Minimize unnecessary setup, compilation,
data loading, and infrastructure overhead.

## Implementation design

Specify the concrete implementation design:

- The files and components to build and their responsibilities.
- The one-time preparation steps and the data or resources they produce.
- The common algorithm interface, including exact input and output formats,
  file paths, and how the evaluator invokes each implementation.
- The datasets or problem instances required for the experiment and how
  to obtain or generate them.
- The selected baseline algorithm and how to implement it faithfully.
- The dependencies, build procedures, and execution environments.
- The correctness checks, experimental conditions, and resource budgets.
- How the evaluator reads algorithm outputs and computes the primary and
  secondary metrics and `combined_score`.
- How errors, invalid outputs, and execution failures are handled.

The evaluator must remain independent of the algorithms being tested.
New algorithms should be evaluable without changing the evaluation criteria.

Favor simple algorithms, small datasets, and lightweight execution environments
where these are sufficient for scientifically meaningful measurements.

Use external libraries, Docker, Modal, or other infrastructure only when
justified by the experiment. Prefer local execution when practical.

Choose concrete technologies, interfaces, and implementation approaches.
Do not leave important architectural decisions to the build agent.

## Code clarity and documentation

Plan for implementations that are easy for a technically sophisticated reader
unfamiliar with the specific research area to understand, inspect, and modify.

Specify how the implementation should:

- Use simple, readable code with descriptive names and clear interfaces.
- Document the purpose, inputs, outputs, and behavior of important functions.
- Explain non-obvious algorithms, mathematical operations, and assumptions
  with meaningful comments.
- Document how each metric is computed and why it matters scientifically.
- Include a concise README describing preparation, execution, evaluation,
  and interpretation of results.
- Keep documentation close to the code it explains.

Comments should explain scientific reasoning and non-obvious behavior,
not merely restate the code.

Prefer straightforward implementations over clever optimizations or
unnecessary abstractions. Keep documentation concise but sufficiently
detailed for someone unfamiliar with the research topic.

## Implementation plan

Provide a clear sequence of steps for building the complete evaluation
environment.

For each step, specify:

- What files or components must be implemented.
- The essential implementation details and dependencies.
- How the completed step should be tested.

Order the steps so that the environment can be built and validated incrementally.

Define the tests, commands, and expected outputs needed to verify that:

- `prepare.py` produces the required shared data and resources.
- Prepared data can be reused without repeating preparation.
- Both algorithms accept the same input format and produce valid output files.
- The evaluator correctly reads algorithm outputs and validates correctness.
- All primary and secondary metrics are calculated correctly and reported.
- `combined_score` is calculated correctly from the specified measurements.
- The baseline algorithm is implemented faithfully.
- Experimental conditions are fair and reproducible.
- `new_algorithm/` and the baseline evaluate successfully.
- Each evaluation writes a valid `results.json` to the project root.
- The resulting scores and metrics are valid and directly comparable.
- Evaluations are sufficiently inexpensive for repeated algorithm discovery.
- The code and documentation are understandable and easy to modify.

Keep the implementation plan focused on the smallest complete working system.

## Guiding principles

Prioritize a scientifically trustworthy, simple, and efficient evaluation
system suitable for rapid algorithm discovery.

The plan must be concrete enough for a coding agent to execute without
redesigning the environment, while avoiding unnecessary implementation details.

Maintain clear separation between:

- Preparation: Generate shared experimental data and resources once.
- Algorithms: Transform experimental inputs into output files.
- Evaluation: Validate outputs, compute metrics, and produce results.json.

Standardize the algorithm interface and evaluation results, not the internal
execution mechanism.

Favor small, readable codebases, clear documentation, minimal dependencies,
and fast evaluations.

Avoid complex software systems, excessive infrastructure, and unnecessary
abstractions.

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
harness following the specification in `EVAL_PLAN.md`.

Read `EVAL_PLAN.md`, `RESEARCH_PLAN.md`, and inspect the existing code
before making changes.

Treat `EVAL_PLAN.md` as authoritative for the evaluation architecture,
algorithm interfaces, baseline implementation, experimental conditions,
metrics, and scientific requirements.

The objective is a small, working, reproducible evaluation environment
that executes the selected experiment, evaluates the baseline and initial
algorithm, and provides a reliable optimization signal for discovering
improved algorithms.

Prioritize simplicity, correctness, and inexpensive repeated evaluations.

## Development Process

Follow the implementation plan in `EVAL_PLAN.md`.

Develop incrementally, starting with the smallest functioning end-to-end
evaluation, then implementing and validating the baseline, experimental
inputs, correctness checks, and metrics.

For each substantial component, determine how its correctness will
be verified before implementing it.

Work autonomously through the implementation:

- Organize the work into independently verifiable milestones.
- Run relevant tests and verification commands after each milestone.
- Diagnose and resolve failures before proceeding.
- Record completed work, verification evidence, and unresolved issues.
- Adjust implementation details as necessary while preserving the plan.

Maintain a working system throughout development.

Exercise independent engineering judgment, but do not redefine the
scientific experiment, baseline, interfaces, or evaluation criteria.

Document any inconsistencies, infeasible requirements, or proposed
scientific changes for human review.

## Evaluation Architecture

Implement the evaluation architecture and interfaces specified in
`EVAL_PLAN.md`.

The environment uses a single `evaluate.py` with a Typer CLI:

    uv run evaluate.py run <implementation_directory>

Each algorithm resides in an independent, self-contained directory.

Given an implementation directory, the evaluator must:

1. Prepare the required execution environment.
2. Execute the implementation.
3. Collect its outputs and artifacts.
4. Validate correctness and problem constraints.
5. Compute the specified metrics.
6. Write `results.json` to the project root, alongside `evaluate.py`.

The results must follow this contract:

    {
        "correct": true,
        "error": null,
        "combined_score": 0.85
    }

- `correct` indicates whether the implementation passed validation.
- `error` describes an evaluation failure, or is null on success.
- `combined_score` is a finite numeric value, with higher values being better.
- Additional diagnostic metrics may be included when useful.

Every invocation must produce a fresh `results.json` in the project root.
Failed evaluations must produce a valid result with a noncompetitive score,
never leave stale results from a previous invocation.

The evaluator owns the complete execution lifecycle.

Implement the algorithm input/output contract, measurements, scoring,
and results format as specified in `EVAL_PLAN.md`.

All implementations must use the same validation rules, scoring
criteria, and comparable experimental conditions.

Keep evaluation logic independent of implementation code.
Implementations must not control validation or scoring.

Support the implementation languages and execution environments
required by the plan, without introducing unnecessary infrastructure.

Standardize inputs and outputs, not internal execution mechanisms.

Handle build failures, timeouts, execution errors, and invalid
outputs robustly.

Keep the architecture simple. Avoid unnecessary frameworks,
abstractions, dependencies, or infrastructure.

## Scientific Verification

Use verification to drive development.

Verify the evaluator independently using known results, reference
implementations, and positive and negative controls where appropriate.

Demonstrate that the evaluator:

- Accepts valid algorithm outputs.
- Rejects invalid outputs.
- Computes the specified metrics and combined_score correctly.
- Applies consistent criteria to every implementation.
- Produces reproducible measurements under controlled conditions.

Validate the baseline against known results or independent
correctness checks where available.

Investigate unexpected results rather than modifying evaluation
criteria to accommodate them.

Ensure the scoring system rewards genuine algorithmic improvements
rather than experimental artifacts.

## Experimental Validation

Run the evaluator on the selected baseline and the initial
implementation in `new_algorithm/`.

The initial implementation may use the baseline as its starting point.
Do not invent a new algorithm as part of building the environment.

Collect and compare the measurements according to `EVAL_PLAN.md`.

Verify that:

- Both implementations execute successfully.
- Each evaluation produces a valid root-level `results.json`.
- The measured scores are meaningful and directly comparable.
- Results are reproducible under the specified experimental conditions.
- Evaluation time and resource requirements are practical for repeated
  algorithm discovery.

Demonstrate that another implementation directory can be evaluated
without modifying the evaluator.

Verify the complete evaluation from a clean state using documented commands.

## Completion Criteria

The work is complete when the evaluation environment specified in
`EVAL_PLAN.md` has been implemented, verified, and demonstrated.

At minimum:

- The evaluation CLI works end to end.
- The selected baseline is implemented faithfully.
- The initial algorithm in `new_algorithm/` evaluates successfully.
- Both implementations satisfy the specified input/output contract.
- Correctness checks and metrics have been independently validated.
- Every evaluation writes a valid `results.json` to the project root.
- The scores are correct, comparable, and reproducible.
- The evaluator can evaluate new implementations without modification.
- The environment is sufficiently lightweight for repeated evaluations.

Report the measured results, verification evidence, discrepancies
with previous work, and remaining limitations.

If completion is blocked, document the blocker and preserve the
verified partial implementation.

## Guiding Principles

Prioritize scientific correctness, simplicity, and reproducibility.

Build the smallest complete system that faithfully implements the
experiment in `EVAL_PLAN.md`.

Favor concise, readable code, minimal dependencies, and fast evaluations.

Do not introduce additional experiments, baselines, or infrastructure
beyond what is required by the plan.

Maintain a working system throughout development.

Take ownership of the entire engineering and verification process.

Preserve the scientific specification while exercising independent
judgment over routine implementation decisions.

The deliverable is not merely working code. It is a small, verified
research environment capable of reliably measuring improvements
in new algorithms.
""")    

