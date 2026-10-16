"""Prompts for the research-plan to evaluation-build workflow."""

import textwrap

research_plan_prompt = textwrap.dedent("""\
Read the related work in `BACKGROUND.md`. If present, inspect the code in the current folder.
Do not perform any web searches. Use only the information provided in the folder.

Write a research plan to `RESEARCH_PLAN.md`.

The goal is to identify the most scientifically valuable opportunity for a new
algorithm to improve on existing methods and define the evidence needed to
demonstrate that improvement.

## Algorithm discovery thesis

Start with a section titled **Algorithm discovery thesis**.

Identify the highest-impact opportunity for a new algorithm to improve on existing methods.

State one clear thesis that:

- addresses an important limitation, tradeoff, or opportunity in previous work,
- explains what kind of algorithmic improvement would matter,
- identifies the conditions under which that improvement is meaningful,
- and explains why success would constitute a significant advance.

Prioritize scientific impact while ensuring the proposed improvement can be evaluated
with a focused, feasible implementation.

## What must be measured

Derive the evaluation objectives directly from the algorithm discovery thesis.

Identify:

- the primary metrics needed to demonstrate the claimed improvement,
- correctness requirements and important tradeoffs,
- experimental conditions and problem instances where the improvement matters,
- controls needed to distinguish genuine improvements from experimental artifacts,
- and what results would constitute convincing evidence of a significant advance.

Focus on measurements that directly test the thesis rather than generic benchmarks.
Where multiple objectives matter, specify the relevant tradeoffs or Pareto frontier.

## Methods to compare against

Identify the existing methods most relevant to testing the algorithm discovery thesis.

Select comparison methods that collectively establish whether the proposed
improvement is meaningful, including:

- the strongest relevant methods from previous work,
- methods representing important performance tradeoffs or competing approaches,
- and simpler baselines when they provide essential context.

For each method, explain:

- why it is necessary to test the thesis,
- what performance or tradeoff it represents,
- and what a new algorithm would need to demonstrate relative to it.

Prioritize comparisons that establish whether a new algorithm advances the
state of the art or improves the relevant performance frontier.

Keep the comparison set focused, scientifically compelling, and feasible to implement.

## Guiding principles

The research plan should form a coherent scientific argument:

    Highest-impact improvement thesis
        -> What evidence would demonstrate success
        -> Which methods must be compared

Choose measurements and comparison methods specifically to test the thesis,
not merely to reproduce standard benchmarks.

Prioritize scientific significance, rigorous evidence, and feasibility.

Do not design the evaluation infrastructure or prescribe implementation details.
Those belong in a separate evaluation plan.

Be concise, specific, and decisive. Select one research direction rather than
proposing multiple competing theses.
""")


eval_plan_prompt = textwrap.dedent("""\
Read `RESEARCH_PLAN.md`, the related work in `BACKGROUND.md`, and, if present,
the code in the current folder. Do not perform any web searches. Use only the
information provided in the folder.

Write an evaluation implementation plan to `EVAL_PLAN.md`.

Treat the algorithm discovery thesis in `RESEARCH_PLAN.md` as the scientific
objective. Do not replace it with a different research direction.

The goal is to produce a concrete, actionable plan for building the smallest
effective evaluation harness that can convincingly test the thesis and support
discovery of improved algorithms.

Make the scientific and engineering decisions necessary for a coding agent
to implement and validate the complete environment without redesigning it.

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
and preserve the important tradeoffs identified in the research plan.

## What needs to be built

Design the smallest evaluation harness capable of producing the measurements
specified in `RESEARCH_PLAN.md` and supporting novel algorithm discovery.

Use a simple, flat project structure:

    evaluate.py
    pyproject.toml
    data/
    new_algorithm/
    method1/
    method2/
    ..
    methodN/

Replace method1, method2, ..., methodN with the names of the selected comparison algorithms.

- `evaluate.py` evaluates any implementation directory.
- `pyproject.toml` defines the uv environment for `evaluate.py`.
- `data/` contains shared experimental data and configurations.
- `new_algorithm/` contains the initial algorithm intended for improvement.
- Each comparison algorithm has its own independent implementation directory.

The evaluator uses a Typer CLI:

    uv run evaluate.py run <implementation_directory>

For example:

    uv run evaluate.py run new_algorithm
    uv run evaluate.py run method1
    uv run evaluate.py run method2

Given an implementation directory, the evaluator prepares and executes
the algorithm, validates its outputs, computes its score, and writes
`results.json`.

The primary output is `combined_score`, with additional diagnostic metrics
when useful.

All algorithms must be evaluated using the same criteria and comparable
experimental conditions.

Specify the concrete implementation design:

- The files and components to build and their responsibilities.
- The common algorithm interface, including inputs, outputs, and how
  the evaluator invokes each implementation.
- The datasets or problem instances to use, including how to obtain
  or generate them and how they are partitioned.
- The selected comparison algorithms and how to implement them faithfully.
- The required dependencies, build procedures, and execution environments.
- The correctness checks, resource budgets, and experimental controls.
- How the evaluator collects outputs, computes metrics, and writes results.
- How errors, invalid outputs, and execution failures are handled.

The evaluator must remain independent of the algorithms being tested.
New algorithms should be evaluable without changing the evaluation criteria.

Choose concrete technologies, interfaces, and implementation approaches.
Do not leave important architectural decisions to the build agent.

## Implementation plan

Provide a clear sequence of steps for building the complete environment.

For each step, specify:

- What files or components must be implemented.
- The essential implementation details and dependencies.
- How the completed step should be tested.

Order the steps so that the evaluator and methods can be developed and
validated incrementally.

Keep the plan focused on the smallest complete working system.
Do not introduce unnecessary abstractions, frameworks, or infrastructure.

## Validation and acceptance criteria

Define how the build agent will establish that the environment is complete
and scientifically trustworthy.

Specify:

- Tests for correctness checks, metric calculations, and scoring.
- Reference results or known cases for validating the evaluator.
- Checks that comparison methods are implemented faithfully.
- Checks for reproducibility and fair experimental conditions.
- Commands to evaluate every implementation independently.
- Expected outputs and criteria for successful completion.

The environment must successfully evaluate `new_algorithm/` and all
comparison methods, producing valid and comparable `results.json` files.

## Guiding principles

Prioritize a scientifically trustworthy, simple, and meaningful evaluation
system.

The plan must be sufficiently concrete for a coding agent to execute,
but avoid unnecessary pseudocode or low-level implementation details.

Make decisive, justified engineering choices. Preserve the scientific
objectives and comparison requirements from `RESEARCH_PLAN.md`.

If a requirement cannot be implemented faithfully, explain the limitation
and specify the smallest scientifically defensible alternative.

Do not implement the environment. Produce only `EVAL_PLAN.md`.

Be concise, specific, and actionable. The resulting plan should serve
directly as the specification for the build step.
""")




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

