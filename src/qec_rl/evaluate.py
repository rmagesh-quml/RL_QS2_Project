"""
Benchmarking harness comparing decoders across physical error rates.

This module provides the experimental machinery to evaluate any set of
decoders side by side. The standard experiment:

    1. Pick a sequence of physical error rates (e.g., 0.01 to 0.20).
    2. For each rate, run many noisy circuits via Qiskit.
    3. Have each decoder predict a correction for every shot.
    4. Compute the logical error rate per decoder per physical rate.
    5. Plot the curves on a single figure.

The resulting plot is the main figure of the project: x-axis physical error
rate, y-axis logical error rate, one curve per decoder. A good decoder
shows logical < physical across the regime of interest.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from qec_rl.circuit import NoiseConfig, NoiseType, run_noisy_experiment
from qec_rl.decoder import Decoder
from qec_rl.syndrome import is_logical_error


@dataclass
class BenchmarkResult:
    """Container for a single benchmark sweep across error rates and decoders.

    Attributes:
        physical_error_rates: The x-axis values that were swept.
        decoder_names: Names of the decoders evaluated, in plot order.
        logical_error_rates: 2D array of shape
            (len(decoder_names), len(physical_error_rates)) holding the
            measured logical error rate for each (decoder, rate) pair.
        shots_per_point: How many shots were used per (decoder, rate) pair.
            Useful for computing confidence intervals later.
        noise_type: Which noise channel was used across the sweep.

    The shape convention is (decoders, rates) so that
    `result.logical_error_rates[i]` is the curve for decoder `i`, ready to
    plot directly.
    """

    physical_error_rates: np.ndarray
    decoder_names: list[str]
    logical_error_rates: np.ndarray
    shots_per_point: int
    noise_type: NoiseType

    def to_table(self) -> str:
        """Return a human-readable table of results.

        Useful for printing to the terminal and for inclusion in reports.
        """
        header = "Physical p | " + " | ".join(
            f"{name:>12s}" for name in self.decoder_names
        )
        lines = [header, "-" * len(header)]
        for i, p in enumerate(self.physical_error_rates):
            row = f"{p:10.4f} | " + " | ".join(
                f"{self.logical_error_rates[j, i]:12.4f}"
                for j in range(len(self.decoder_names))
            )
            lines.append(row)
        return "\n".join(lines)


def measure_logical_error_rate(
    decoder: Decoder,
    noise_config: NoiseConfig,
    shots: int = 5000,
    seed: int | None = None,
) -> float:
    """Run a decoder against many noisy shots and return its logical error rate.

    Each shot is independent. We encode `|0>_L` and `|1>_L` in equal halves
    so the result is unbiased with respect to the encoded value. The
    syndrome alone is fed to the decoder; the data bits are used after the
    fact to compute whether decoding succeeded.

    Args:
        decoder: Any object implementing the Decoder interface.
        noise_config: Noise channel and rate to apply.
        shots: Total shots, split evenly between logical 0 and 1.
        seed: Optional RNG seed for reproducibility.

    Returns:
        The fraction of shots for which the decoder produced a logical error
        (a number in [0, 1]).
    """
    half = shots // 2
    logical_errors = 0
    total = 0

    # Run logical_state = 0 then logical_state = 1, both with the same noise.
    for logical_state, n in ((0, half), (1, shots - half)):
        result = run_noisy_experiment(
            config=noise_config,
            logical_state=logical_state,
            shots=n,
            # Distinct seeds per half so the two halves don't share noise.
            seed=None if seed is None else seed + logical_state,
        )
        syndromes = result["syndromes"]
        data_bits = result["data_bits"]

        for i in range(n):
            action = decoder.decode(int(syndromes[i]))
            if is_logical_error(data_bits[i], action, logical_state):
                logical_errors += 1
            total += 1

    return logical_errors / total


def run_benchmark(
    decoders: Iterable[Decoder],
    physical_error_rates: Iterable[float] = (
        0.01,
        0.02,
        0.04,
        0.06,
        0.08,
        0.10,
        0.12,
        0.15,
        0.20,
    ),
    noise_type: NoiseType = "bit_flip",
    shots_per_point: int = 5000,
    seed: int | None = None,
    show_progress: bool = True,
) -> BenchmarkResult:
    """Sweep all decoders across the given physical error rates.

    For each (decoder, physical_rate) pair, runs `shots_per_point` shots and
    measures the logical error rate. Returns a `BenchmarkResult` ready for
    plotting and tabulation.

    Args:
        decoders: Iterable of Decoder subclass instances. The order of this
            iterable determines plot legend order.
        physical_error_rates: Sequence of error rates to sweep.
        noise_type: Which Pauli channel to use ('bit_flip', 'phase_flip',
            'depolarizing'). Bit-flip is the right choice for the 3-qubit
            bit-flip code; the others are exposed for ablation studies.
        shots_per_point: Shots used per (decoder, rate) pair. 5000 gives
            standard error roughly +/- 0.5% on the measured logical rate.
        seed: Optional RNG seed for reproducibility.
        show_progress: Whether to print a tqdm progress bar.

    Returns:
        BenchmarkResult holding the swept rates, decoder names, and a 2D
        array of measured logical error rates.
    """
    decoders = list(decoders)
    rates = np.array(list(physical_error_rates), dtype=np.float64)

    decoder_names = [d.name for d in decoders]
    logical_rates = np.empty((len(decoders), len(rates)), dtype=np.float64)

    iterator = tqdm(
        total=len(decoders) * len(rates),
        desc="Benchmarking",
        disable=not show_progress,
    )
    for j, decoder in enumerate(decoders):
        for i, p in enumerate(rates):
            config = NoiseConfig(noise_type=noise_type, error_rate=float(p))
            # Offset seed per (decoder, rate) so each cell has independent noise.
            cell_seed = None if seed is None else seed + 1000 * j + i
            logical_rates[j, i] = measure_logical_error_rate(
                decoder=decoder,
                noise_config=config,
                shots=shots_per_point,
                seed=cell_seed,
            )
            iterator.update(1)
            iterator.set_postfix(
                decoder=decoder.name,
                p=f"{p:.3f}",
                logical=f"{logical_rates[j, i]:.4f}",
            )
    iterator.close()

    return BenchmarkResult(
        physical_error_rates=rates,
        decoder_names=decoder_names,
        logical_error_rates=logical_rates,
        shots_per_point=shots_per_point,
        noise_type=noise_type,
    )


def plot_benchmark(
    result: BenchmarkResult,
    save_path: str | Path | None = None,
    show: bool = False,
    title: str | None = None,
) -> plt.Figure:
    """Render the headline comparison plot.

    Plots logical error rate vs physical error rate for every decoder in
    the result, plus a dashed diagonal y = x as a visual reference. A
    decoder whose curve dips below the diagonal is genuinely helping;
    above the diagonal means encoding is hurting more than it helps at
    that physical rate.

    Args:
        result: A BenchmarkResult from run_benchmark.
        save_path: If given, save the figure as PNG to this path.
        show: Whether to call plt.show(). Set False in scripts that just
            want to save the figure.
        title: Optional plot title; auto-generated if None.

    Returns:
        The matplotlib Figure object, in case the caller wants to customize
        further before saving.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    rates = result.physical_error_rates

    # Reference line: logical = physical. Decoders below this line help.
    ax.plot(
        rates,
        rates,
        linestyle="--",
        color="gray",
        alpha=0.6,
        label="No encoding (y = x)",
    )

    # One line per decoder.
    for j, name in enumerate(result.decoder_names):
        ax.plot(
            rates,
            result.logical_error_rates[j],
            marker="o",
            label=name,
        )

    ax.set_xlabel("Physical error rate p")
    ax.set_ylabel("Logical error rate")
    auto_title = (
        f"3-qubit bit-flip code: decoder comparison "
        f"({result.noise_type}, {result.shots_per_point} shots/point)"
    )
    ax.set_title(title or auto_title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)

    if show:
        plt.show()

    return fig