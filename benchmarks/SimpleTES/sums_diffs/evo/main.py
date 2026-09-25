"""Baseline constructor for the sums/differences benchmark.

This entire file is evolvable. The trusted evaluator only depends on the JSON
output contract documented in the task prompt.
"""

import json
from pathlib import Path


OUTPUT_FILE = Path("sums_diffs.json")


def construct_set() -> list[int]:
    """Return the original SimpleTES candidate integer set."""
    return [0, 1, 2, 4, 5, 9, 12, 13, 14, 16, 17, 21, 24, 25, 26, 28, 29]


def main() -> None:
    OUTPUT_FILE.write_text(
        json.dumps({"values": construct_set()}),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
