"""Baseline constructor for the second autocorrelation inequality problem.

This entire file is evolvable. The trusted evaluator only depends on the JSON
output contract documented below.
"""

import json
from pathlib import Path

import numpy as np


OUTPUT_FILE = Path("autocorrelation.json")


def simpson_l2_squared(convolution: np.ndarray) -> tuple[float, np.ndarray]:
    """Return the piecewise-linear L2 norm squared and its gradient."""
    count = convolution.size
    width = 1.0 / (count + 1)
    y = np.empty(count + 2, dtype=convolution.dtype)
    y[0] = 0.0
    y[1:-1] = convolution
    y[-1] = 0.0

    left = y[:-1]
    right = y[1:]
    value = (width / 3.0) * np.sum(left * left + left * right + right * right)
    gradient_y = (width / 3.0) * (4.0 * y + np.roll(y, 1) + np.roll(y, -1))
    return float(value), gradient_y[1:-1]


def objective_and_gradient(convolution: np.ndarray) -> tuple[float, np.ndarray]:
    """Return R(f) and its gradient with respect to the convolution."""
    l2_squared, gradient_l2 = simpson_l2_squared(convolution)
    width = 1.0 / (convolution.size + 1)
    l1 = width * float(np.sum(convolution))
    gradient_l1 = np.full_like(convolution, width)

    linfinity = float(np.max(convolution))
    maxima = convolution == linfinity
    gradient_linfinity = maxima.astype(convolution.dtype) / int(np.sum(maxima))

    denominator = l1 * linfinity
    value = l2_squared / denominator
    gradient = (
        gradient_l2 * denominator
        - l2_squared * (gradient_l1 * linfinity + l1 * gradient_linfinity)
    ) / (denominator * denominator)
    return float(value), gradient


def sequence_gradient(sequence: np.ndarray, convolution_gradient: np.ndarray) -> np.ndarray:
    """Backpropagate through the full self-convolution."""
    return 2.0 * np.convolve(convolution_gradient, sequence[::-1], mode="valid")


class Adam:
    def __init__(self, shape: tuple[int, ...], learning_rate: float) -> None:
        self.mean = np.zeros(shape, dtype=np.float32)
        self.variance = np.zeros(shape, dtype=np.float32)
        self.steps = 0
        self.learning_rate = learning_rate

    def step(self, values: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        self.steps += 1
        self.mean = 0.9 * self.mean + 0.1 * gradient
        self.variance = 0.999 * self.variance + 0.001 * gradient * gradient
        corrected_mean = self.mean / (1.0 - 0.9**self.steps)
        corrected_variance = self.variance / (1.0 - 0.999**self.steps)
        return values + self.learning_rate * corrected_mean / (
            np.sqrt(corrected_variance) + 1e-8
        )


def score_batch(batch: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    scores = np.zeros(batch.shape[0], dtype=np.float32)
    gradients: list[np.ndarray] = []
    for index, sequence in enumerate(batch):
        convolution = np.convolve(sequence, sequence, mode="full")
        scores[index], gradient = objective_and_gradient(convolution)
        gradients.append(gradient)
    return scores, gradients


def optimize_batch(
    batch: np.ndarray,
    optimizers: list[Adam],
    rng: np.random.Generator,
    step_number: int,
) -> tuple[np.ndarray, np.ndarray]:
    scores, convolution_gradients = score_batch(batch)
    noise_scale = 1e-3 / ((step_number + 1) ** 0.55)
    for index in range(batch.shape[0]):
        gradient = sequence_gradient(batch[index], convolution_gradients[index])
        gradient = gradient + noise_scale * rng.normal(size=gradient.shape)
        batch[index] = np.clip(
            optimizers[index].step(batch[index], gradient.astype(np.float32)),
            0.0,
            None,
        )
    return batch, scores


def refine(sequence: np.ndarray, steps: int = 40_000) -> np.ndarray:
    optimizer = Adam(sequence.shape, learning_rate=3e-3)
    for _ in range(steps):
        convolution = np.convolve(sequence, sequence, mode="full")
        _, convolution_gradient = objective_and_gradient(convolution)
        gradient = sequence_gradient(sequence, convolution_gradient)
        sequence = np.clip(
            optimizer.step(sequence, gradient.astype(np.float32)), 0.0, None
        )
    return sequence


def upsample(sequence: np.ndarray) -> np.ndarray:
    old_x = np.linspace(-0.5, 0.5, sequence.size)
    new_x = np.linspace(-0.5, 0.5, 2 * sequence.size)
    return np.interp(new_x, old_x, sequence).astype(np.float32)


def construct_sequence() -> list[float]:
    """Run the original SimpleTES-style population search and refinement."""
    rng = np.random.default_rng(0)
    population_size = 64
    sequence_size = 256
    batch = rng.uniform(0.0, 1.0, size=(population_size, sequence_size)).astype(
        np.float32
    )
    batch[0] = 1.0
    optimizers = [Adam((sequence_size,), learning_rate=3e-2) for _ in batch]
    best_sequences = batch.copy()
    best_scores = np.full(population_size, -np.inf, dtype=np.float32)

    for step_number in range(10_000):
        batch, scores = optimize_batch(batch, optimizers, rng, step_number)
        improved = scores > best_scores
        best_scores = np.where(improved, scores, best_scores)
        best_sequences[improved] = batch[improved]

    sequence = best_sequences[int(np.argmax(best_scores))]
    sequence = refine(upsample(sequence))
    sequence = refine(upsample(sequence))
    return sequence.tolist()


def main() -> None:
    OUTPUT_FILE.write_text(
        json.dumps({"sequence": construct_sequence()}),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
