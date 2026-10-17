"""Prompts for the research-plan to evaluation-build workflow."""

import textwrap


# research plan prompt below ------------------------------------------------------------------------------

research_plan_prompt = textwrap.dedent("""\
Read the related work in `BACKGROUND.md`. If present, inspect the code in the current folder.
Do not perform any web searches. Use only the information provided in the folder.

Write a research plan to `RESEARCH_PLAN.md`.

The goal is to identify the highest-impact opportunity for improving existing
algorithms and define the evidence needed to demonstrate an advance.

Focus on identifying one decisive experiment and its measurements that would
demonstrate an advance, not proposing an improved algorithm.

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

Prioritize scientific impact while ensuring the opportunity can be investigated
through one focused, feasible experiment.

## Experiment and measurements

Identify the single most compelling experiment for determining whether a
hypothetical new algorithm represents an improvement over existing methods.

Derive the experiment directly from the algorithm discovery thesis.

1. Describe the experiment, the scientific question it addresses, and the
   conditions or problem instances under which it should be conducted.

2. Under a **Measurements** subsection, specify:
   - exactly what must be measured and how each metric is defined,
   - correctness requirements and important performance tradeoffs,
   - what must be controlled to ensure fair comparisons,
   - and what results would constitute evidence of improvement.

The experiment should directly test the central claim of the thesis rather
than reproduce a generic benchmark. Where multiple objectives matter,
specify the relevant tradeoffs or Pareto frontier.

Choose an experiment that maximizes scientific significance while remaining
focused and feasible to implement.

## Baseline method

Identify the single strongest relevant existing method for testing the
algorithm discovery thesis in the selected experiment.

Choose a baseline that:

- represents a strong, credible reference from previous work,
- directly addresses the limitation or tradeoff identified in the thesis,
- provides a meaningful standard for demonstrating improvement,
- and is feasible to implement and evaluate faithfully.

Explain why this method is the most informative comparison, what performance
or tradeoff it represents, and what a hypothetical improved algorithm would
need to demonstrate relative to it.

Prioritize scientific relevance and strength over simplicity, while keeping
the baseline feasible to implement.

## Guiding principles

The research plan should form a coherent scientific argument:

    Highest-impact problem or opportunity
        -> One decisive experiment
            -> Measurements required to demonstrate improvement
        -> One strong baseline to compare against

Focus on defining a valuable, measurable opportunity for algorithm discovery.

Do not propose new algorithms, solution mechanisms, architectures, or
implementation strategies. Discovering how to achieve the improvement is
a separate research activity.

Choose the experiment, measurements, and baseline specifically to test
the thesis, not merely to reproduce a standard benchmark.

Prioritize scientific significance, rigorous evidence, and feasibility.

Do not design the evaluation infrastructure or prescribe implementation details.
Those belong in a separate evaluation plan.

Be concise, specific, and decisive. Select one high-impact opportunity,
one compelling experiment, and one strong baseline rather than proposing
multiple research directions, experiments, or comparison methods.
""")

# eval plan prompt below ------------------------------------------------------------------------------

eval_plan_prompt = textwrap.dedent("""\
Read `RESEARCH_PLAN.md`, the related work in `BACKGROUND.md`, and, if present,
the code in the current folder. Do not perform any web searches. Use only the
information provided in the folder.

Write an evaluation plan to `EVAL_PLAN.md`.

Treat the algorithm discovery thesis and experiment in `RESEARCH_PLAN.md` as
the scientific objectives. Do not replace them with a different research
direction or introduce additional experiments or baselines.

The goal is to produce a concrete implementation plan for the smallest
effective evaluation harness that can execute the experiment, measure the
proposed improvement, and support discovery of improved algorithms.

Make the scientific and engineering decisions needed for a coding agent to
build and validate the complete environment.

## Scoring system

Use the measurements and correctness requirements in `RESEARCH_PLAN.md` to
define a single numeric `combined_score`, with higher values indicating better
algorithms.

Specify:

- The exact metrics and how they are computed.
- The formula for `combined_score`, including normalization or weighting.
- How incorrect, invalid, or failed implementations are scored.
- Why the score captures meaningful progress toward the research thesis.

The score must distinguish genuine algorithmic improvements from artifacts
while preserving the important tradeoffs identified in the experiment.

## What needs to be built

Design the smallest evaluation harness capable of executing the experiment
and producing the measurements specified in `RESEARCH_PLAN.md`.

Use a simple, flat project structure:

    evaluate.py
    pyproject.toml
    data/
    new_algorithm/
    baseline_method/

Replace baseline_method with the name of the selected baseline algorithm.

- `evaluate.py` evaluates any implementation directory.
- `pyproject.toml` defines the uv environment for `evaluate.py`.
- `data/` contains shared experimental data and configurations.
- `new_algorithm/` contains the initial algorithm intended for improvement.
- The baseline directory contains an independent implementation of the
  selected comparison algorithm.

The initial algorithm may use the baseline implementation as its starting
point. Do not design a new algorithm as part of the evaluation environment.

## Evaluation interface

The evaluator uses a Typer CLI:

    uv run evaluate.py run <implementation_directory>

For example:

    uv run evaluate.py run new_algorithm
    uv run evaluate.py run baseline_method

Given an implementation directory, the evaluator:

1. Prepares the execution environment, including building or deploying as needed.
2. Executes the selected algorithm and collects its outputs.
3. Validates correctness and computes the evaluation metrics.
4. Writes `results.json` to the project root, alongside `evaluate.py`.

The output contract is:

    {
        "correct": true,
        "error": null,
        "combined_score": 0.85
    }

- `correct` indicates whether the implementation passed validation.
- `error` describes any evaluation failure, or is null on success.
- `combined_score` is a finite numeric value, with higher values being better.
- Additional diagnostic metrics may be included when useful.

Every invocation must produce a fresh `results.json` in the project root,
regardless of which implementation directory is evaluated. Failed evaluations
must produce a valid result with a noncompetitive score, never leave stale
results from a previous run.

The evaluator must preserve this interface for every implementation.

All algorithms must be evaluated using the same criteria and comparable
experimental conditions.

## Implementation design

Specify the concrete implementation design:

- The files and components to build and their responsibilities.
- The common algorithm interface, including inputs, outputs, and how
  the evaluator invokes each implementation.
- The datasets or problem instances required for the experiment and how
  to obtain or generate them.
- The selected baseline algorithm and how to implement it faithfully.
- The dependencies, build procedures, and execution environments.
- The correctness checks, experimental conditions, and resource budgets.
- How the evaluator computes the measurements and score defined above.
- How errors, invalid outputs, and execution failures are handled.

The evaluator must remain independent of the algorithms being tested.
New algorithms should be evaluable without changing the evaluation criteria.

Choose concrete technologies, interfaces, and implementation approaches.
Do not leave important architectural decisions to the build agent.

## Implementation plan

Provide a clear sequence of steps for building the complete evaluation
environment.

For each step, specify:

- What files or components must be implemented.
- The essential implementation details and dependencies.
- How the completed step should be tested.

Order the steps so that the environment can be built and validated incrementally.

Define the tests, commands, and expected outputs needed to verify that:

- The evaluator correctly implements the experiment and its measurements.
- The baseline algorithm is implemented faithfully.
- Experimental conditions are fair and reproducible.
- `new_algorithm/` and the baseline evaluate successfully.
- Each evaluation writes a valid `results.json` to the project root.
- The resulting scores are valid and directly comparable.

Keep the implementation plan focused on the smallest complete working system.

## Guiding principles

Prioritize a scientifically trustworthy, simple, and meaningful evaluation
system.

The plan must be concrete enough for a coding agent to execute without
redesigning the environment, while avoiding unnecessary implementation details.

Standardize the evaluation interface and results, not the execution mechanism.

Avoid unnecessary abstractions, infrastructure, and implementation complexity.

Preserve the scientific objectives of `RESEARCH_PLAN.md`. If a requirement
cannot be implemented faithfully, explain the limitation and the smallest
scientifically defensible alternative.

Do not implement the environment. Produce only `EVAL_PLAN.md`.

Be concise, specific, and decisive. The resulting plan should serve directly
as the specification for the build step.
""")

# build plan prompt below ------------------------------------------------------------------------------

build_prompt = textwrap.dedent("""\
Build a complete, scientifically trustworthy research evaluation
harness following the specification in `PLAN.md`.

Treat `PLAN.md` as authoritative for the evaluation architecture,
algorithm interfaces, comparison algorithms, experimental conditions, metrics,
and scientific requirements.

You are responsible for the entire engineering process, including
implementation planning, development, testing, debugging, and
experimental verification.

The objective is a working, reproducible research evaluation
harness that evaluates existing and novel algorithms using consistent criteria
and produces meaningful, comparable results.

## Development Process

Read `PLAN.md` and inspect the existing code before making changes.

Develop and maintain an incremental implementation plan with concrete
milestones and verification criteria.

Start with the smallest functioning end-to-end evaluation, then
progressively implement and validate the required comparison algorithms,
experimental inputs, correctness checks, and metrics.

For each substantial component, determine how its correctness
will be verified before implementing it.

Work autonomously through the implementation:

- Organize the work into independently verifiable milestones.
- Run relevant tests and verification commands after each milestone.
- Diagnose and resolve failures before proceeding.
- Record completed work, verification evidence, and unresolved issues.
- Revise the implementation plan as necessary.
- Parallelize independent tasks when beneficial while maintaining
  responsibility for integration and end-to-end correctness.

Maintain a working system throughout development.

Exercise independent engineering judgment, but do not redefine
the scientific experiment, interfaces, or evaluation criteria
specified in `PLAN.md`.

Document any inconsistencies, infeasible requirements, or proposed
scientific changes for human review.

## Evaluation Architecture

Implement the evaluation architecture and interfaces specified
in `PLAN.md`.

The environment uses a single `evaluate.py` with a Typer CLI:

    uv run evaluate.py run <implementation_directory>

Each algorithm resides in an independent, self-contained directory.

Given an implementation directory, the evaluator must:

1. Prepare the required execution environment.
2. Execute the implementation.
3. Collect its outputs and artifacts.
4. Validate correctness and problem constraints.
5. Compute the specified metrics.
6. Produce a standardized `results.json`.

The evaluator owns the complete execution lifecycle.

Implement the algorithm input/output contract, metrics, and results
format as specified in `PLAN.md`.

All implementations must use the same validation rules, scoring
criteria, and comparable experimental conditions.

Keep evaluation logic independent of implementation code.
Implementations must not control validation or scoring.

Support different implementation languages and execution
environments as required by the plan.

Standardize inputs and outputs, not internal execution mechanisms.

Ensure independent and concurrent evaluations cannot interfere
with each other or overwrite results.

Handle build failures, timeouts, execution errors, and invalid
outputs robustly.

Keep the architecture simple. Avoid unnecessary frameworks,
abstractions, or infrastructure.

## Scientific Verification

Use verification to drive development.

Verify the evaluator independently using known results, reference
implementations, and positive and negative controls where appropriate.

Demonstrate that the evaluator:

- Accepts valid algorithm outputs.
- Rejects invalid outputs.
- Computes the specified metrics correctly.
- Detects meaningful differences between algorithms.
- Applies consistent criteria to every implementation.

Validate each baseline against known results or independent
correctness checks.

Investigate unexpected results rather than modifying evaluation
criteria to accommodate them.

Ensure measured improvements reflect genuine algorithmic differences
rather than experimental artifacts.

## Experimental Validation

Run the evaluator independently on every required baseline and
the initial implementation intended for improvement.

Collect the resulting metrics and compare the algorithms according
to the methodology specified in `PLAN.md`.

Verify that results are reproducible under the specified experimental
conditions, including appropriate control of randomness, dependencies,
execution parameters, and resource limits.

Demonstrate that a new implementation can be evaluated without
modifying the evaluator's validation or scoring logic.

Reproduce the complete evaluation from a clean state using
documented commands.

## Completion Criteria

The work is complete when the research evaluation harness specified in
`PLAN.md` has been implemented, independently verified, and
demonstrated through successful evaluations of the required algorithms.

At minimum:

- The evaluation CLI works end to end.
- Every required algorithm has a working implementation.
- All implementations satisfy the specified input/output contracts.
- Correctness checks and metrics have been independently validated.
- Each algorithm produces a valid `results.json`.
- Baseline results have been checked against expected behavior.
- Algorithms have been evaluated and compared under fair conditions.
- Results are reproducible within documented tolerances.
- New algorithms can be evaluated through the same interface.
- The experiments and results have been inspected for inconsistencies.

Report the measured results, verification evidence, discrepancies
with previous work, and remaining limitations.

If completion is blocked, document the blocker and preserve the
verified partial implementation.

## Guiding Principles

Prioritize scientific correctness, reproducibility, and simplicity.

Maintain a working system throughout development.

Take ownership of the entire engineering and verification process.

Preserve the scientific specification in `PLAN.md` while exercising
independent judgment over implementation decisions.

The deliverable is not merely working code. It is a verified research
environment capable of reliably measuring improvements in new algorithms.
""")    

