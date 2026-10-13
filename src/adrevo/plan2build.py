"""Prompts for the research-plan to evaluation-build workflow."""

import textwrap

initial_plan_prompt = textwrap.dedent("""\
    Read the code in the current folder and the related work in `BACKGROUND.md`.

    Write version 1 of a research plan to `PLAN_v1.md`.

    The goal is to identify the **highest-impact algorithm discovery thesis that is still feasible to evaluate with a focused implementation**.

    Start with a section titled **Algorithm discovery thesis**.

    State one clear thesis describing the most valuable direction in which a new algorithm should improve on existing methods. The thesis should:

    - address an important limitation, tradeoff, or open opportunity in previous work,
    - explain what kind of algorithmic improvement would matter,
    - identify the conditions or constraints under which the improvement is meaningful,
    - and explain why success would constitute a significant advance.

    Before selecting the thesis, consider a small number of plausible discovery directions from the related work. Choose primarily based on expected scientific impact, while rejecting directions whose evaluation would be impractically large or difficult.

    Then include a section titled **What needs to be built**.

    Describe clearly and comprehensively the smallest evaluation system needed to establish the current baseline and support the algorithm discovery thesis, including:

    - which previous methods must be implemented or integrated,
    - the common interface they should use,
    - the datasets or inputs required,
    - the metrics that must be measured,
    - the experimental conditions and controls,
    - and the outputs needed to compare the methods fairly.

    The rest of the plan should:

    - identify the core problem shared by the related work,
    - summarize only the previous approaches necessary for a convincing comparison,
    - explain the limitations or tradeoffs that motivate the algorithm discovery thesis,
    - define what “better” means using the smallest useful set of metrics and constraints,
    - describe how baseline results will be checked for correctness and consistency with previous work,
    - state what empirical observations establish the current performance frontier,
    - state what result from a future algorithm would constitute convincing support for the thesis,
    - and identify major uncertainties, risks, or reasons the thesis might fail.

    Favor directions with substantial potential impact, but reduce the experimental scope to the **minimum convincing evaluation**.

    Avoid low-impact directions chosen only because they are easy. Also avoid broad or ambitious evaluation plans whose implementation burden is not necessary to establish the central thesis.
""")


critique_prompt = textwrap.dedent("""\
    Given the plan you created in `PLAN_v{version}.md`, the code in the current folder, and `BACKGROUND.md`.

    Write a critique to `CRITIQUE_v{version}.md`.

    The goal is to determine whether the plan identifies the **highest-impact feasible algorithm discovery thesis** and whether the proposed evaluation can be reduced to the smallest implementation that still makes a convincing scientific case.

    Start by evaluating the **Algorithm discovery thesis**.

    Ask:

    - Does the thesis target an important limitation, tradeoff, or open opportunity in the related work?
    - Would success represent a meaningful algorithmic advance?
    - Is the proposed direction stronger than obvious lower-impact alternatives?
    - Is the thesis specific enough to guide downstream algorithm discovery?
    - Are the conditions and constraints under which improvement matters clearly defined?
    - Is there a more impactful thesis that is still realistically evaluable?

    Then evaluate **What needs to be built**.

    Focus aggressively on simplicity and feasibility:

    - Is every baseline necessary?
    - Is every metric required to support the thesis?
    - Can the datasets or experimental regimes be reduced?
    - Are the comparisons fair and aligned?
    - Can baseline correctness be established realistically?
    - Does any component require disproportionate engineering effort?
    - Is the plan trying to prove more than is necessary?
    - Could a substantially smaller evaluation support the same thesis?

    Do not simplify the plan by replacing a high-impact thesis with an easier but less important one unless the original thesis is genuinely impractical.

    Instead, preserve the strongest feasible thesis and minimize the machinery required to evaluate it.

    Identify anything that is unnecessary, overly ambitious, weakly connected to the thesis, difficult to reproduce, or likely to make the build stage fragile.

    Then propose the **minimum convincing version of the plan**, including:

    1. the final algorithm discovery thesis,
    2. the exact components that need to be built,
    3. the minimum set of baselines,
    4. the minimum set of metrics,
    5. the minimum datasets or experimental conditions,
    6. the checks required to trust the baseline results,
    7. and the evidence required to establish the current frontier and support the thesis.

    Prefer maximum scientific impact per unit of implementation effort.

    Make the last line of `CRITIQUE_v{version}.md` exactly REFINE if the plan can be improved, or ACCEPT if the plan is already the strongest feasible version. Do not add Markdown formatting to this line.
""")


refine_plan_prompt = textwrap.dedent("""\
    Read `PLAN_v{version}.md`, `CRITIQUE_v{version}.md`, the code in the
    current folder, and `BACKGROUND.md`.

    Refine the plan in response to the critique and write the complete,
    self-contained revised plan to `PLAN_v{next_version}.md`.
""")




build_prompt = textwrap.dedent("""\
    Build a simple, reproducible evaluation environment for the given
    research problem and algorithm discovery thesis described in `{plan_file}`.

    ## Core Design

    The evaluation environment follows a minimal contract:

    - Input: A required implementation directory.
    - Output: A results.json file containing correctness, errors, and metrics.
    - Execution: evaluate.py owns the complete evaluation lifecycle.

    The evaluator must be independent of the implementation being tested.
    All implementations, including baselines and evolved algorithms, must
    be evaluated using identical criteria.

    ## Directory Structure

    Use a flat project structure:

    project/
    ├── evaluate.py
    ├── evo/
    │   ├── main.py
    │   └── pyproject.toml
    ├── method1/
    │   ├── main.py
    │   └── pyproject.toml
    └── method2/
        ├── main.py
        └── pyproject.toml

    `evo/` contains the implementation intended for evolution.
    `method1/` and `method2/` contain independent comparison methods.
    Any implementation directory may also serve as a starting point
    for evolution.

    ## Evaluation Interface

    Use Typer with an explicit `run` command that requires an
    implementation directory.

    Usage:

        uv run evaluate.py run evo
        uv run evaluate.py run method1
        uv run evaluate.py run method2

    Example:

        from pathlib import Path
        import typer

        app = typer.Typer()

        @app.callback()
        def cli():
            \"\"\"Research evaluation environment.\"\"\"
            pass

        @app.command()
        def run(implementation_dir: Path):
            \"\"\"Evaluate an implementation.\"\"\"
            evaluate(implementation_dir.resolve())

        def evaluate(implementation_dir: Path):
            # Prepare environment, build, execute, collect outputs,
            # validate, score, and write results.json.
            ...

        if __name__ == "__main__":
            app()

    The explicit command structure should allow additional commands
    to be introduced later without changing the existing interface.
    Do not implement additional commands unless needed.

    ## Evaluation Lifecycle

    Given an implementation directory, evaluate.py must:

    1. Prepare: Build, compile, install dependencies, create containers,
    or deploy to remote infrastructure as needed.
    2. Execute: Run the selected implementation in its environment.
    3. Collect: Retrieve outputs, including artifacts from remote jobs.
    4. Validate: Check correctness, feasibility, and problem constraints.
    5. Score: Compute meaningful metrics that distinguish solution quality.
    6. Report: Write results.json using a consistent schema.

    Example results.json:

        {
            "correct": true,
            "error": null,
            "combined_score": 0.85,
            "accuracy": 0.92,
            "runtime_seconds": 1.42
        }

    The evaluator determines how each implementation is executed.
    Implementations may use Python, C++, Rust, Docker, Modal, or other
    technologies without requiring changes to the evaluation interface.

    ## Requirements

    - Identical evaluation: Use the same datasets, validation rules,
    scoring functions, and comparable resource budgets for all methods.
    - Self-contained implementations: Each directory contains the code
    and dependencies required to execute its method.
    - Reproducibility: Control randomness, execution parameters, and
    resource limits where appropriate.
    - Isolation: Prevent collisions between builds, intermediate files,
    outputs, and concurrent evaluations.
    - Robustness: Handle build failures, timeouts, execution errors,
    and invalid outputs gracefully.
    - Scientific validity: Ensure metrics measure meaningful algorithmic
    improvements rather than artifacts of the evaluation process.
    - Evolution compatibility: Keep the implementation in evo/ easy
    to modify, evaluate, and replace with another starting method.

    ## Deliverables

    1. A complete, working evaluate.py using Typer.
    2. An evolvable implementation in evo/.
    3. At least one meaningful comparison implementation in method1/.
    4. Additional comparison methods when useful.
    5. A demonstration evaluating each implementation independently
    and comparing their results.

    ## Guiding Principles

    Prioritize simplicity, scientific validity, and reproducibility.

    The evaluator owns the complete execution lifecycle. Do not introduce
    separate baseline evaluators, execution frameworks, plugin systems,
    or unnecessary abstractions.

    Keep evaluation logic trusted and separate from implementation code.
    Implementations must not be able to modify validation rules, scoring
    criteria, or evaluation datasets.

    Standardize the input and output, not the execution mechanism.

    Build only what is necessary to evaluate and compare algorithms
    reliably. Prefer concise, readable code and straightforward interfaces.
""")



