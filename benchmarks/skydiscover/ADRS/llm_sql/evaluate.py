"""Evaluator for LLM SQL prompt caching column reordering optimization."""

import fcntl
import json
import shutil
import time
import traceback
import tempfile
import urllib.request
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor

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

    def max_overlap(trie, row_string):
        return min(len(row_string), trie.longest_common_prefix(row_string))

    trie = Trie()
    total_prefix_hit_count = 0
    total_string_length = 0

    def process_row(index, row):
        nonlocal total_string_length
        row_string = "".join(row.fillna("").astype(str).values)
        total_string_length += len(row_string)
        row_prefix_hit_count = max_overlap(trie, row_string)
        trie.insert(row_string)
        return row_prefix_hit_count

    with ThreadPoolExecutor() as executor:
        results = executor.map(process_row, df.index, [row for _, row in df.iterrows()])

    total_prefix_hit_count = sum(results)
    total_prefix_hit_rate = total_prefix_hit_count / total_string_length
    assert total_prefix_hit_count <= total_string_length
    return total_prefix_hit_count, total_prefix_hit_rate * 100


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

        input_path = Path("evo/input.json")
        output_path = Path("evo/output.json")
        output_path.unlink(missing_ok=True)
        input_path.write_text(json.dumps({"requests": [
            {
                "input_file": str(filename.resolve()),
                "output_file": filename.name,
                "options": {"early_stop": 100000, "distinct_value_threshold": 0.7, "row_stop": 4, "col_stop": 2, "col_merge": col_merge},
            }
            for filename, col_merge in zip(test_files, col_merges)
            if filename.is_file()
        ]}), encoding="utf-8")
        subprocess.run(["uv", "run", "-qq", "--directory", "evo", "python", "main.py"], check=True)
        candidate_runtimes = json.loads(output_path.read_text(encoding="utf-8"))["runtimes"]

        failed_files = 0
        hit_rates = []
        total_runtime = 0.0
        successful_files = 0

        for index, (filename, col_merge) in enumerate(zip(test_files, col_merges)):
            try:
                if not filename.is_file():
                    print(f"Dataset not found: {filename}, skipping...")
                    failed_files += 1
                    continue

                print(f"Processing dataset: {filename}")
                master_df = pd.read_csv(filename)

                total_chars_before = master_df.astype(str).apply(lambda x: x.str.len().sum(), axis=1).sum()
                original_row_count = len(master_df)

                reordered = pd.read_csv(Path("evo/outputs") / filename.name)
                runtime = candidate_runtimes[index]

                # Validate row count
                reordered_row_count = len(reordered)
                if reordered_row_count != original_row_count:
                    diff = reordered_row_count - original_row_count
                    if diff < 0:
                        error_msg = f"Row count decreases by {abs(diff)} rows. Data were lost."
                    else:
                        error_msg = f"Row count increases by {diff} rows. Data were duplicated."
                    result = {"correct": False, "error": error_msg, "combined_score": 0.0}
                    with open("results.json", "w") as f:
                        json.dump(result, f, indent=4)
                    sys.exit(0)

                # Validate character count
                total_chars_after = reordered.astype(str).apply(lambda x: x.str.len().sum(), axis=1).sum()

                if total_chars_after < total_chars_before:
                    char_diff_pct = ((total_chars_before - total_chars_after) / total_chars_before * 100) if total_chars_before > 0 else 0
                    error_msg = f"Character count decreases by {char_diff_pct:.2f}%. Data were lost."
                    result = {"correct": False, "error": error_msg, "combined_score": 0.0}
                    with open("results.json", "w") as f:
                        json.dump(result, f, indent=4)
                    sys.exit(0)

                results_tuple = evaluate_df_prefix_hit_cnt(reordered)
                print(f"Results: {results_tuple}, Runtime: {runtime}")

                hit_rate = results_tuple[1] / 100

                hit_rates.append(hit_rate)
                total_runtime += runtime
                successful_files += 1

            except Exception as e:
                print(f"Failed to process {filename.name}: {str(e)}")
                traceback.print_exc()
                failed_files += 1
                break

        if successful_files == 0:
            result = {"correct": False, "error": "No files processed successfully", "combined_score": 0.0}
        elif failed_files > 0:
            result = {"correct": False, "error": "1 or more files failed to run", "combined_score": 0.0}
        else:
            average_hit_rate = sum(hit_rates) / successful_files
            average_runtime = total_runtime / successful_files
            score = 0.95 * average_hit_rate + 0.05 * (12 - min(12, average_runtime)) / 12

            result = {"correct": True, "error": None, "combined_score": float(score)}

        with open("results.json", "w") as f:
            json.dump(result, f, indent=4)

    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        traceback.print_exc()
        result = {"correct": False, "error": str(e), "combined_score": 0.0}
        with open("results.json", "w") as f:
            json.dump(result, f, indent=4)
