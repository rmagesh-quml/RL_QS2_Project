"""
qec_rl — Reinforcement learning decoder for the 3-qubit bit-flip code.

Quick start
-----------
    from qec_rl import NoiseConfig, RandomDecoder, MWPMDecoder, LookupDecoder
    from qec_rl.agent import train_dqn, RLDecoder
    from qec_rl.evaluate import run_benchmark, plot_benchmark

    # Train the RL agent
    train_dqn(save_path="models/dqn_qec.zip", total_timesteps=20_000)

    # Benchmark all four decoders
    result = run_benchmark([
        RandomDecoder(),
        LookupDecoder(),
        MWPMDecoder(),
        RLDecoder("models/dqn_qec.zip"),
    ])
    print(result.to_table())
    plot_benchmark(result, save_path="results/comparison.png")

Module layout
-------------
    circuits.py   - Qiskit encoding, noise injection, syndrome measurement
    syndrome.py   - syndrome-to-correction logic, action constants
    decoders.py   - Decoder ABC, RandomDecoder, MWPMDecoder, LookupDecoder
    env.py        - Gymnasium environment wrapping the QEC problem
    agent.py      - DQN training (train_dqn) and RLDecoder wrapper
    evaluate.py   - benchmark harness, BenchmarkResult, plot_benchmark

Note: agent.py is not imported here because importing stable-baselines3
is slow (~1s). Users who only need classical decoders or the Qiskit
simulation layer don't pay that cost. Import from qec_rl.agent explicitly.
"""

from qec_rl.circuit import (
    NoiseConfig,
    build_encoded_circuit,
    apply_noise,
    add_syndrome_measurement,
    measure_data_qubits,
    run_noisy_experiment,
)
from qec_rl.syndrome import (
    ACTION_NONE,
    ACTION_FLIP_Q0,
    ACTION_FLIP_Q1,
    ACTION_FLIP_Q2,
    NUM_ACTIONS,
    SYNDROME_TO_CORRECTION,
    lookup_correction,
    apply_correction,
    majority_vote,
    is_logical_error,
)
from qec_rl.decoder import (
    Decoder,
    RandomDecoder,
    MWPMDecoder,
    LookupDecoder,
)
from qec_rl.env import QECEnv
from qec_rl.evaluate import (
    BenchmarkResult,
    measure_logical_error_rate,
    run_benchmark,
    plot_benchmark,
)

__version__ = "0.1.0"

__all__ = [
    # circuits
    "NoiseConfig",
    "build_encoded_circuit",
    "apply_noise",
    "add_syndrome_measurement",
    "measure_data_qubits",
    "run_noisy_experiment",
    # syndrome
    "ACTION_NONE",
    "ACTION_FLIP_Q0",
    "ACTION_FLIP_Q1",
    "ACTION_FLIP_Q2",
    "NUM_ACTIONS",
    "SYNDROME_TO_CORRECTION",
    "lookup_correction",
    "apply_correction",
    "majority_vote",
    "is_logical_error",
    # decoders
    "Decoder",
    "RandomDecoder",
    "MWPMDecoder",
    "LookupDecoder",
    # env
    "QECEnv",
    # evaluate
    "BenchmarkResult",
    "measure_logical_error_rate",
    "run_benchmark",
    "plot_benchmark",
]