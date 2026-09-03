"""Evaluator for the Real-Time Adaptive Signal Processing Algorithm."""

import json
import subprocess
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

INPUT_FILE = Path("evo/signal_processing_input.json")
OUTPUT_FILE = Path("evo/signal_processing.json")


def safe_float(value):
    try:
        if np.isnan(value) or np.isinf(value):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def calculate_slope_changes(signal_data):
    if len(signal_data) < 3:
        return 0
    diffs = np.diff(signal_data)
    sign_changes = 0
    for i in range(1, len(diffs)):
        if np.sign(diffs[i]) != np.sign(diffs[i - 1]) and diffs[i - 1] != 0:
            sign_changes += 1
    return sign_changes


def calculate_lag_error(filtered_signal, original_signal, window_size):
    if len(filtered_signal) == 0:
        return 1.0
    delay = window_size - 1
    if len(original_signal) <= delay:
        return 1.0
    recent_filtered = filtered_signal[-1]
    recent_original = original_signal[delay + len(filtered_signal) - 1]
    return abs(recent_filtered - recent_original)


def calculate_average_tracking_error(filtered_signal, original_signal, window_size):
    if len(filtered_signal) == 0:
        return 1.0
    delay = window_size - 1
    if len(original_signal) <= delay:
        return 1.0
    aligned_original = original_signal[delay : delay + len(filtered_signal)]
    min_length = min(len(filtered_signal), len(aligned_original))
    if min_length == 0:
        return 1.0
    return np.mean(np.abs(filtered_signal[:min_length] - aligned_original[:min_length]))


def calculate_false_reversal_penalty(filtered_signal, clean_signal, window_size):
    if len(filtered_signal) < 3 or len(clean_signal) < 3:
        return 0
    delay = window_size - 1
    if len(clean_signal) <= delay:
        return 1.0
    aligned_clean = clean_signal[delay : delay + len(filtered_signal)]
    min_length = min(len(filtered_signal), len(aligned_clean))
    if min_length < 3:
        return 0
    filtered_diffs = np.diff(filtered_signal[:min_length])
    clean_diffs = np.diff(aligned_clean[:min_length])
    false_reversals = 0
    for i in range(1, len(filtered_diffs)):
        filtered_change = (
            np.sign(filtered_diffs[i]) != np.sign(filtered_diffs[i - 1])
            and filtered_diffs[i - 1] != 0
        )
        clean_change = (
            np.sign(clean_diffs[i]) != np.sign(clean_diffs[i - 1]) and clean_diffs[i - 1] != 0
        )
        if filtered_change and not clean_change:
            false_reversals += 1
    return false_reversals


def calculate_composite_score(S, L_recent, L_avg, R, alpha=(0.3, 0.2, 0.2, 0.3)):
    S_norm = min(S / 50.0, 2.0)
    L_recent_norm = min(L_recent, 2.0)
    L_avg_norm = min(L_avg, 2.0)
    R_norm = min(R / 25.0, 2.0)
    penalty = alpha[0] * S_norm + alpha[1] * L_recent_norm + alpha[2] * L_avg_norm + alpha[3] * R_norm
    return 1.0 / (1.0 + penalty)


def generate_test_signals(num_signals=5):
    test_signals = []
    for i in range(num_signals):
        np.random.seed(42 + i)
        length = 500 + i * 100
        noise_level = 0.2 + i * 0.1
        t = np.linspace(0, 10, length)
        if i == 0:
            clean = 2 * np.sin(2 * np.pi * 0.5 * t) + 0.1 * t
        elif i == 1:
            clean = np.sin(2 * np.pi * 0.5 * t) + 0.5 * np.sin(2 * np.pi * 2 * t) + 0.2 * np.sin(2 * np.pi * 5 * t)
        elif i == 2:
            clean = np.sin(2 * np.pi * (0.5 + 0.2 * t) * t)
        elif i == 3:
            clean = np.concatenate([np.ones(length // 3), 2 * np.ones(length // 3), 0.5 * np.ones(length - 2 * (length // 3))])
        else:
            clean = np.cumsum(np.random.randn(length) * 0.1) + 0.05 * t
        noise = np.random.normal(0, noise_level, length)
        noisy = clean + noise
        test_signals.append((noisy, clean))
    return test_signals


if __name__ == "__main__":
    test_signals = generate_test_signals(5)
    try:
        OUTPUT_FILE.unlink(missing_ok=True)
        INPUT_FILE.write_text(
            json.dumps({
                "noisy_signals": [signal.tolist() for signal, _ in test_signals],
                "window_size": 20,
            }),
            encoding="utf-8",
        )
        subprocess.run(["uv", "run", "-qq", "--directory", "evo", "python", "main.py"], check=True)
        candidate_outputs = json.loads(OUTPUT_FILE.read_text(encoding="utf-8"))["filtered_signals"]
        if len(candidate_outputs) != len(test_signals):
            raise ValueError("Candidate returned the wrong number of filtered signals")
    except Exception as exc:
        Path("results.json").write_text(
            json.dumps({"correct": False, "error": str(exc), "combined_score": 0.0}, indent=4),
            encoding="utf-8",
        )
        raise SystemExit

    all_scores = []
    all_metrics = []
    successful_runs = 0

    for (noisy_signal, clean_signal), candidate_output in zip(test_signals, candidate_outputs):
        try:
            filtered_signal = np.array(candidate_output)
            if len(filtered_signal) == 0:
                continue

            window_size = 20
            S = calculate_slope_changes(filtered_signal)
            L_recent = calculate_lag_error(filtered_signal, noisy_signal, window_size)
            L_avg = calculate_average_tracking_error(filtered_signal, noisy_signal, window_size)
            R = calculate_false_reversal_penalty(filtered_signal, clean_signal, window_size)
            composite_score = calculate_composite_score(S, L_recent, L_avg, R)

            correlation = 0.0
            noise_reduction = 0.0
            try:
                delay = window_size - 1
                aligned_clean = clean_signal[delay : delay + len(filtered_signal)]
                min_length = min(len(filtered_signal), len(aligned_clean))
                if min_length > 1:
                    corr_result = pearsonr(filtered_signal[:min_length], aligned_clean[:min_length])
                    correlation = corr_result[0] if not np.isnan(corr_result[0]) else 0.0
                aligned_noisy = noisy_signal[delay : delay + len(filtered_signal)][:min_length]
                aligned_clean = aligned_clean[:min_length]
                if min_length > 0:
                    noise_before = np.var(aligned_noisy - aligned_clean)
                    noise_after = np.var(filtered_signal[:min_length] - aligned_clean)
                    noise_reduction = (noise_before - noise_after) / noise_before if noise_before > 0 else 0
                    noise_reduction = max(0, noise_reduction)
            except Exception:
                pass

            all_scores.append(composite_score)
            all_metrics.append({
                "slope_changes": safe_float(S),
                "lag_error": safe_float(L_recent),
                "avg_error": safe_float(L_avg),
                "false_reversals": safe_float(R),
                "composite_score": safe_float(composite_score),
                "correlation": safe_float(correlation),
                "noise_reduction": safe_float(noise_reduction),
            })
            successful_runs += 1
        except Exception:
            continue

    if successful_runs == 0:
        result = {"correct": False, "error": "All test signals failed", "combined_score": 0.0}
    else:
        avg_composite_score = np.mean(all_scores)
        avg_correlation = np.mean([m["correlation"] for m in all_metrics])
        avg_noise_reduction = np.mean([m["noise_reduction"] for m in all_metrics])
        avg_slope_changes = np.mean([m["slope_changes"] for m in all_metrics])
        success_rate = successful_runs / len(test_signals)

        smoothness_score = 1.0 / (1.0 + avg_slope_changes / 20.0)
        accuracy_score = max(0, avg_correlation)

        overall_score = (
            0.4 * avg_composite_score
            + 0.2 * smoothness_score
            + 0.2 * accuracy_score
            + 0.1 * avg_noise_reduction
            + 0.1 * success_rate
        )

        if accuracy_score < 0.1:
            overall_score = 0.0

        result = {
            "correct": True,
            "error": None,
            "combined_score": safe_float(overall_score),
        }

    Path("results.json").write_text(json.dumps(result, indent=4), encoding="utf-8")
