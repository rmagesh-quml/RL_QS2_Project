"""
Quantum circuit construction for the 3-qubit bit-flip code.

This module builds Qiskit circuits that:
    1. Encode a logical qubit |psi>_L into 3 physical qubits.
    2. Apply a chosen noise channel (bit-flip, phase-flip, or depolarizing).
    3. Measure stabilizers via 2 ancilla qubits to extract a 2-bit syndrome.

The 3-qubit bit-flip code protects against single-qubit X (bit-flip) errors only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import numpy as np

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, pauli_error, depolarizing_error

NoiseType = Literal["bit_flip", "phase_flip", "depolarizing"]


@dataclass(frozen=True)
class NoiseConfig:
    """Configuration for a noise channel applied to each data qubit.

    Attributes:
        noise_type: Which Pauli channel to apply. 'bit_flip' applies X with
            probability `error_rate`; 'phase_flip' applies Z; 'depolarizing'
            applies X, Y, or Z each with probability error_rate/3.
        error_rate: Physical error probability per qubit, in [0, 1].
            Typical hardware values are 0.001-0.01; we sweep higher for testing.

    Example:
        >>> cfg = NoiseConfig(noise_type="bit_flip", error_rate=0.1)
    """

    noise_type: NoiseType = "bit_flip"
    error_rate: float = 0.1

    def __post_init__(self) -> None:
        if not 0.0 <= self.error_rate <= 1.0:
            raise ValueError(
                f"error_rate must be in [0, 1], got {self.error_rate}"
            )


def build_encoded_circuit(logical_state: int = 0) -> QuantumCircuit:
    """Build a circuit that encodes a logical qubit into 3 physical qubits.

    The 3-qubit bit-flip encoding maps:
        |0>_L -> |000>
        |1>_L -> |111>

    We start with all qubits in |0>. If the logical state is 1, we apply X to
    the first data qubit to get |100>. Then two CNOTs (q0 -> q1, q0 -> q2)
    spread the information to produce |111>.

    Note that this encoding is deterministic — we are encoding a computational
    basis state, not a superposition. For a general state alpha|0> + beta|1>
    you would prepare that state on q0 first, then apply the CNOTs. We use
    basis states here because they suffice to evaluate logical error rates.

    Args:
        logical_state: 0 or 1, the classical logical value to encode.

    Returns:
        A QuantumCircuit with 3 data qubits in the encoded state and
        2 ancilla qubits in |0> ready for syndrome measurement.

    Raises:
        ValueError: If logical_state is not 0 or 1.
    """
    if logical_state not in (0, 1):
        raise ValueError(f"logical_state must be 0 or 1, got {logical_state}")

    data = QuantumRegister(3, name="data")
    ancilla = QuantumRegister(2, name="ancilla")
    syndrome_bits = ClassicalRegister(2, name="syndrome")
    logical_bits = ClassicalRegister(3, name="logical")

    qc = QuantumCircuit(data, ancilla, syndrome_bits, logical_bits)

    # Prepare the logical state on the first data qubit.
    if logical_state == 1:
        qc.x(data[0])

    # Spread the state across all 3 data qubits via CNOTs.
    qc.cx(data[0], data[1])
    qc.cx(data[0], data[2])

    qc.barrier(label="encoded")
    return qc


def _build_noise_model(config: NoiseConfig) -> NoiseModel:
    """Construct a Qiskit Aer NoiseModel for the given configuration.

    The noise is attached to the identity gate, which we explicitly apply to
    each data qubit after encoding. This lets us control exactly when and
    where noise is injected (rather than adding it to every gate in the
    circuit, which would also corrupt the encoding itself).

    Args:
        config: The noise configuration.

    Returns:
        A NoiseModel ready to be passed to AerSimulator.
    """
    noise_model = NoiseModel()
    p = config.error_rate

    if config.noise_type == "bit_flip":
        # X with probability p, identity with probability 1-p.
        error = pauli_error([("X", p), ("I", 1 - p)])
    elif config.noise_type == "phase_flip":
        error = pauli_error([("Z", p), ("I", 1 - p)])
    elif config.noise_type == "depolarizing":
        # Depolarizing: X, Y, or Z each with probability p/3.
        error = depolarizing_error(p, num_qubits=1)
    else:
        raise ValueError(f"Unknown noise type: {config.noise_type}")

    # Attach the error channel to the identity gate on any qubit.
    noise_model.add_all_qubit_quantum_error(error, ["id"])
    return noise_model


def apply_noise(qc: QuantumCircuit, data_qubits: QuantumRegister) -> None:
    """Insert identity gates on the data qubits as noise injection points.

    The identity gates are mathematically no-ops on an ideal simulator, but
    when run with a NoiseModel attached, they become the trigger points where
    Pauli errors are stochastically applied. This is the standard Qiskit
    pattern for injecting noise at specific locations.

    Args:
        qc: The circuit to modify in place.
        data_qubits: The qubits that should experience noise.
    """
    for q in data_qubits:
        qc.id(q)
    qc.barrier(label="noise")


def add_syndrome_measurement(qc: QuantumCircuit) -> None:
    """Add stabilizer measurement circuit in place.

    Measures two stabilizers of the 3-qubit bit-flip code:
        S1 = Z_0 Z_1  (parity of data qubits 0 and 1)
        S2 = Z_1 Z_2  (parity of data qubits 1 and 2)

    Each stabilizer is measured by entangling the data qubits with an ancilla
    via CNOTs, then measuring the ancilla in the computational basis. The
    ancilla ends up in |0> if the parity is even (no error or double error)
    and |1> if odd (single error on one of the two qubits).

    The two syndrome bits (s1, s2) uniquely identify which of the 3 data
    qubits, if any, was flipped:
        (0, 0) -> no error
        (1, 0) -> data qubit 0 flipped
        (1, 1) -> data qubit 1 flipped
        (0, 1) -> data qubit 2 flipped

    Args:
        qc: The circuit to modify in place. Must have registers named
            'data' (3 qubits), 'ancilla' (2 qubits), and 'syndrome'
            (2 classical bits).
    """
    data = qc.qregs[0]  # 'data' register
    ancilla = qc.qregs[1]  # 'ancilla' register
    syndrome = qc.cregs[0]  # 'syndrome' register

    # Stabilizer S1 = Z_0 Z_1 measured via ancilla[0].
    qc.cx(data[0], ancilla[0])
    qc.cx(data[1], ancilla[0])

    # Stabilizer S2 = Z_1 Z_2 measured via ancilla[1].
    qc.cx(data[1], ancilla[1])
    qc.cx(data[2], ancilla[1])

    qc.barrier(label="stabilizers")

    # Measure the ancillas into the syndrome classical register.
    qc.measure(ancilla[0], syndrome[0])
    qc.measure(ancilla[1], syndrome[1])


def measure_data_qubits(qc: QuantumCircuit) -> None:
    """Measure the 3 data qubits into the 'logical' classical register.

    Called at the very end of the circuit to read out the final state of the
    encoded qubit, after noise and any correction has been applied. The
    majority vote of the three bits recovers the logical value.

    Args:
        qc: The circuit to modify in place. Must have a 'data' register
            (3 qubits) and a 'logical' register (3 classical bits).
    """
    data = qc.qregs[0]
    logical = qc.cregs[1]  # second classical register
    for i in range(3):
        qc.measure(data[i], logical[i])


def run_noisy_experiment(
    config: NoiseConfig,
    logical_state: int = 0,
    shots: int = 1024,
    seed: int | None = None,
) -> dict[str, object]:
    """Run the full encode -> noise -> syndrome -> measure pipeline.

    Each shot independently samples the noise channel, so the returned arrays
    contain `shots` independent realizations of syndrome and final data
    measurements. This is the data the decoders will operate on.

    Args:
        config: Noise channel to apply.
        logical_state: 0 or 1, the logical value to encode and test.
        shots: Number of independent runs.
        seed: Optional RNG seed for reproducibility.

    Returns:
        A dict with keys:
            - 'syndromes': np.ndarray of shape (shots,), each entry in [0, 3]
              representing the 2-bit syndrome as s0 + 2*s1.
            - 'data_bits': np.ndarray of shape (shots, 3), the raw measured
              data qubit values per shot (0 or 1). Needed by decoders to
              apply corrections to specific qubits.
            - 'logical_measured': np.ndarray of shape (shots,), majority-vote
              readout of the data qubits per shot (0 or 1).
            - 'logical_expected': int, the logical_state that was encoded.
    """
    qc = build_encoded_circuit(logical_state=logical_state)
    apply_noise(qc, qc.qregs[0])
    add_syndrome_measurement(qc)
    measure_data_qubits(qc)

    simulator = AerSimulator(
        noise_model=_build_noise_model(config),
        seed_simulator=seed,
    )
    counts = simulator.run(qc, shots=shots).result().get_counts()

    # Qiskit returns bitstrings like "010 01" where the space separates the
    # two classical registers. The LAST-declared register appears FIRST in
    # the string. We declared 'syndrome' first, then 'logical', so the format
    # is "<logical_bits> <syndrome_bits>", both printed MSB-first.
    syndromes = np.empty(shots, dtype=np.int8)
    data_bits_arr = np.empty((shots, 3), dtype=np.int8)
    logical_measured = np.empty(shots, dtype=np.int8)

    idx = 0
    for bitstring, count in counts.items():
        logical_str, syndrome_str = bitstring.split()
        # reversed() so index 0 of the result corresponds to data qubit 0
        data_bits = [int(b) for b in reversed(logical_str)]
        syndrome = int(syndrome_str[-1]) + 2 * int(syndrome_str[-2])
        majority = int(sum(data_bits) >= 2)

        syndromes[idx : idx + count] = syndrome
        data_bits_arr[idx : idx + count] = data_bits
        logical_measured[idx : idx + count] = majority
        idx += count

    return {
        "syndromes": syndromes,
        "data_bits": data_bits_arr,
        "logical_measured": logical_measured,
        "logical_expected": logical_state,
    }