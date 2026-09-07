"""Evaluator for LLM SQL prompt caching column reordering optimization."""

import fcntl
import json
import shutil
import time
import traceback
import tempfile
import urllib.request
from typing import Tuple
from collections import Counter

import pandas as pd

import warnings
import subprocess
from pathlib import Path

warnings.filterwarnings(
    "ignore",
    message="Setting an item of incompatible dtype is deprecated",
    category=FutureWarning,
)


_DATASET_VERSION = "llm-sql-v1"
_DATASET_BASE_URL = (
    "https://huggingface.co/datasets/f20180301/adrs-data/resolve/main/llm_sql"
)
_DATASET_FILES = ("movies.csv", "beer.csv", "BIRD.csv", "PDMX.csv", "products.csv")


def _dataset_cache_dir() -> Path:
    return Path.home() / ".cache" / "adrevo" / "datasets" / _DATASET_VERSION


def _dataset_is_ready(dataset_dir: Path) -> bool:
    return all((dataset_dir / filename).is_file() for filename in _DATASET_FILES)


def ensure_llm_sql_data() -> Path:
    """Return the node-local LLM-SQL dataset, downloading it once if needed."""
    dataset_dir = _dataset_cache_dir()
    if _dataset_is_ready(dataset_dir):
        return dataset_dir

    dataset_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = dataset_dir.parent / f".{_DATASET_VERSION}.lock"
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            if _dataset_is_ready(dataset_dir):
                return dataset_dir

            if dataset_dir.exists():
                shutil.rmtree(dataset_dir)
            staging_dir = Path(
                tempfile.mkdtemp(
                    prefix=f".{_DATASET_VERSION}.",
                    dir=dataset_dir.parent,
                )
            )
            try:
                for filename in _DATASET_FILES:
                    urllib.request.urlretrieve(
                        f"{_DATASET_BASE_URL}/datasets/{filename}",
                        staging_dir / filename,
                    )
                if not _dataset_is_ready(staging_dir):
                    raise RuntimeError("LLM-SQL dataset download was incomplete")
                staging_dir.replace(dataset_dir)
            finally:
                if staging_dir.exists():
                    shutil.rmtree(staging_dir)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)

    return dataset_dir


# ---------- Trie for prefix hit calculation (from utils.py) ----------

class TrieNode:
    def __init__(self):
        self.children = {}
        self.end_of_word = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.end_of_word = True

    def longest_common_prefix(self, word):
        node = self.root
        common_prefix_length = 0
        for char in word:
            if char in node.children:
                common_prefix_length += len(char)
                node = node.children[char]
            else:
                break
        return common_prefix_length


def evaluate_df_prefix_hit_cnt(df: pd.DataFrame) -> Tuple[int, float]:
    """Evaluate the prefix hit count of a DataFrame."""

    trie = Trie()
    total_prefix_hit_count = 0
    total_string_length = 0
    for row in df.itertuples(index=False, name=None):
        row_string = "".join(row)
        total_prefix_hit_count += trie.longest_common_prefix(row_string)
        total_string_length += len(row_string)
        trie.insert(row_string)

    hit_rate = total_prefix_hit_count / total_string_length if total_string_length else 0.0
    return total_prefix_hit_count, hit_rate * 100


def prepare_input(df: pd.DataFrame, merge_groups: list[list[str]]) -> pd.DataFrame:
    """Merge configured columns in their declared order before candidate execution."""
    df = df.copy()
    for group in merge_groups:
        if not group or len(set(group)) != len(group) or any(col not in df for col in group):
            raise ValueError(f"Invalid merge group: {group}")
        name = "_".join(group)
        if name in df.columns and name not in group:
            raise ValueError(f"Merged column name already exists: {name}")
        merged = df[group].agg("".join, axis=1)
        df = df.drop(columns=group)
        df[name] = merged
    return df


def row_contents(df: pd.DataFrame) -> Counter:
    """Preserve cell and row multiplicity while allowing their reordering."""
    return Counter(tuple(sorted(row)) for row in df.itertuples(index=False, name=None))


def validate_output(expected: pd.DataFrame, actual: pd.DataFrame) -> None:
    if expected.shape != actual.shape or row_contents(expected) != row_contents(actual):
        raise ValueError("Output changed, dropped, or duplicated input data")


# ---------- Main evaluation ----------

if __name__ == "__main__":
    try:
        datasets_dir = ensure_llm_sql_data()

        test_files = [
            datasets_dir / "movies.csv",
            datasets_dir / "beer.csv",
            datasets_dir / "BIRD.csv",
            datasets_dir / "PDMX.csv",
            datasets_dir / "products.csv",
        ]

        col_merges = [
            [['movieinfo', 'movietitle', 'rottentomatoeslink']],
            [['beer/beerId', 'beer/name']],
            [['PostId', 'Body']],
            [['path', 'metadata'], ['hasmetadata', 'isofficial', 'isuserpublisher', 'isdraft', 'hasannotations', 'subsetall']],
            [['product_title', 'parent_asin']],
        ]

        prepared_dir = Path("evo/inputs")
        prepared_dir.mkdir(exist_ok=True)
        output_dir = Path("evo/outputs")
        output_dir.mkdir(exist_ok=True)
        expected_frames = []
        requests = []
        for filename, groups in zip(test_files, col_merges):
            original = pd.read_csv(filename, dtype=str, keep_default_na=False)
            expected = prepare_input(original, groups)
            expected_frames.append(expected)
            prepared_path = prepared_dir / filename.name
            expected.to_csv(prepared_path, index=False)
            (output_dir / filename.name).unlink(missing_ok=True)
            requests.append({
                "input_file": str(prepared_path.resolve()),
                "output_file": filename.name,
                "options": {"early_stop": 100000, "distinct_value_threshold": 0.7,
                            "row_stop": 4, "col_stop": 2, "col_merge": []},
            })

        Path("evo/input.json").write_text(json.dumps({"requests": requests}), encoding="utf-8")
        started = time.perf_counter()
        subprocess.run(["uv", "run", "-qq", "--directory", "evo", "python", "main.py"], check=True)
        total_runtime = time.perf_counter() - started

        hit_rates = []
        for filename, expected in zip(test_files, expected_frames):
            reordered = pd.read_csv(output_dir / filename.name, dtype=str, keep_default_na=False)
            validate_output(expected, reordered)
            _, hit_rate = evaluate_df_prefix_hit_cnt(reordered)
            hit_rates.append(hit_rate / 100)

        average_hit_rate = sum(hit_rates) / len(hit_rates)
        average_runtime = total_runtime / len(test_files)
        score = 0.95 * average_hit_rate + 0.05 * (12 - min(12, average_runtime)) / 12
        result = {"correct": True, "error": None, "combined_score": float(score),
                  "average_hit_rate": average_hit_rate, "average_runtime": average_runtime}

        with open("results.json", "w") as f:
            json.dump(result, f, indent=4)

    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        traceback.print_exc()
        result = {"correct": False, "error": str(e), "combined_score": 0.0}
        with open("results.json", "w") as f:
            json.dump(result, f, indent=4)
