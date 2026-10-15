"""Prompts for the research-plan to evaluation-build workflow."""

import textwrap

plan_prompt = textwrap.dedent("""\
Read the related work in `BACKGROUND.md`.
If present, inspect the code in the current folder.
Do not perform any web searches. Use only the information provided in the folder.

Write a research plan to `PLAN.md`.

The goal is to design the smallest effective research environment for discovering significant improvements over existing algorithms.

## Algorithm discovery thesis

Start with a section titled **Algorithm discovery thesis**.

Identify the most valuable opportunity for a new algorithm to improve on existing methods.

State one clear thesis that:

- addresses an important limitation, tradeoff, or opportunity in previous work,
- explains what kind of algorithmic improvement would matter,
- identifies the conditions under which that improvement is meaningful,
- and explains why success would constitute a significant advance.

Prioritize scientific impact while ensuring the proposed improvement can be evaluated with a focused, feasible implementation.

## What must be measured

Derive the evaluation objectives directly from the algorithm discovery thesis.

Identify the measurements needed to determine whether a new algorithm achieves the proposed improvement, including any correctness requirements or important tradeoffs.

Define a single numeric `combined_score`, with higher values indicating better algorithms. Explain how the score is calculated and why it captures meaningful progress toward the thesis.

The measurements should distinguish genuine algorithmic improvements from existing methods under fair experimental conditions.

## What needs to be built

Design the smallest research environment capable of producing these measurements and supporting novel algorithm discovery.

Use a simple, flat project structure:

    evaluate.py
    data/
    evo/
    method1/
    method2/

Replace method1 and method2 with the names of the selected baseline methods.

- `evaluate.py` evaluates any implementation directory.
- `data/` contains shared experimental data and configurations.
- `evo/` contains the initial algorithm intended for evolution.
- Each baseline has its own independent implementation directory.

The evaluator uses a Typer CLI:

    uv run evaluate.py run <implementation_directory>

Given an implementation directory, the evaluator prepares and executes
the algorithm, validates its outputs, computes its score, and writes
`results.json`.

The primary output is `combined_score`, with additional diagnostic metrics
when useful.

All algorithms must be evaluated using the same criteria and comparable
experimental conditions.

Specify:

- Which existing algorithms to implement as baselines and which to use
  as the initial `evo/` implementation.
- The common algorithm interface, including inputs, outputs, and how
  the evaluator invokes each implementation.
- The datasets or problem instances required and how to obtain or
  generate them.
- The correctness checks and experimental conditions needed for
  reliable measurements.
- How the evaluator computes the measurements and score defined above.

The evaluator must remain independent of the algorithms being tested.
New algorithms should be evaluable without changing the evaluation
criteria.

## Guiding principles

Prioritize scientific impact, simplicity, and meaningful evaluation.

The thesis determines what must be measured, and those measurements
determine what needs to be built.

Favor the **minimum convincing evaluation** needed to support the thesis.

Avoid unnecessary baselines, experiments, or infrastructure.

Specify what needs to be evaluated and how improvement is measured.
Leave engineering decisions and implementation details to the build stage.
""")




build_prompt = textwrap.dedent("""\
Build a complete, scientifically trustworthy research evaluation
environment following the specification in `PLAN.md`.

Treat `PLAN.md` as authoritative for the evaluation architecture,
algorithm interfaces, baselines, experimental conditions, metrics,
and scientific requirements.

You are responsible for the entire engineering process, including
implementation planning, development, testing, debugging, and
experimental verification.

The objective is a working, reproducible research environment that
evaluates existing and novel algorithms using consistent criteria
and produces meaningful, comparable results.

## Development Process

Read `PLAN.md` and inspect the existing code before making changes.

Develop and maintain an incremental implementation plan with concrete
milestones and verification criteria.

Start with the smallest functioning end-to-end evaluation, then
progressively implement and validate the required baselines,
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
the initial implementation intended for evolution.

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

The work is complete when the research environment specified in
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

