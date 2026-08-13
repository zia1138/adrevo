"""Evaluator for LLM SQL prompt caching column reordering optimization."""

import json
import os
import time
import traceback
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from main import Evolved

import warnings

warnings.filterwarnings(
    "ignore",
    message="Setting an item of incompatible dtype is deprecated",
    category=FutureWarning,
)


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
        # Dataset configuration
        eval_dir = os.path.dirname(os.path.abspath(__file__))
        datasets_dir = os.path.join(eval_dir, "datasets")

        test_files = [
            os.path.join(datasets_dir, "movies.csv"),
            os.path.join(datasets_dir, "beer.csv"),
            os.path.join(datasets_dir, "BIRD.csv"),
            os.path.join(datasets_dir, "PDMX.csv"),
            os.path.join(datasets_dir, "products.csv"),
        ]

        col_merges = [
            [['movieinfo', 'movietitle', 'rottentomatoeslink']],
            [['beer/beerId', 'beer/name']],
            [['PostId', 'Body']],
            [['path', 'metadata'], ['hasmetadata', 'isofficial', 'isuserpublisher', 'isdraft', 'hasannotations', 'subsetall']],
            [['product_title', 'parent_asin']],
        ]

        failed_files = 0
        hit_rates = []
        total_runtime = 0.0
        successful_files = 0

        for filename, col_merge in zip(test_files, col_merges):
            try:
                if not os.path.exists(filename):
                    print(f"Dataset not found: {filename}, skipping...")
                    failed_files += 1
                    continue

                print(f"Processing dataset: {filename}")
                master_df = pd.read_csv(filename)

                total_chars_before = master_df.astype(str).apply(lambda x: x.str.len().sum(), axis=1).sum()
                original_row_count = len(master_df)

                st = time.time()
                reordered, _ = Evolved().reorder(
                    master_df,
                    early_stop=100000,
                    distinct_value_threshold=0.7,
                    row_stop=4,
                    col_stop=2,
                    col_merge=col_merge,
                )
                runtime = time.time() - st

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
                print(f"Failed to process {os.path.basename(filename)}: {str(e)}")
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
