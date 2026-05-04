"""
Syndrome decoding logic for the 3-qubit bit-flip code.

This module provides the canonical syndrome-to-correction mapping for the
3-qubit bit-flip code, plus utilities for applying corrections to measured
bitstrings and checking whether decoding succeeded.

The syndrome table is small and unambiguous: for the 3-qubit bit-flip code,
each of the 4 possible syndromes maps to exactly one most-likely correction.
This is why the code is degenerate-free at distance 1 — there is no "choice"
to make given a syndrome, so any correct decoder (lookup, MWPM, RL) converges
to the same policy. We still isolate this logic because:

    1. A trained RL agent should rediscover this table from reward signal alone.
    2. The random baseline decoder intentionally ignores this table.
    3. Extending to larger codes requires replacing this mapping with a
       matching algorithm; isolating it keeps that change local.
"""

from __future__ import annotations

import numpy as np


# Actions the decoder can choose. Each action either does nothing or applies
# an X correction to a specific data qubit.
ACTION_NONE = 0
ACTION_FLIP_Q0 = 1
ACTION_FLIP_Q1 = 2
ACTION_FLIP_Q2 = 3

NUM_ACTIONS = 4

# Canonical syndrome-to-correction table for the 3-qubit bit-flip code.
#
# Syndrome encoding: s0 + 2*s1, where s0 is the outcome of stabilizer
# Z_0 Z_1 and s1 is the outcome of Z_1 Z_2.
#
#   Syndrome 0 = (s0=0, s1=0) -> no error detected -> do nothing
#   Syndrome 1 = (s0=1, s1=0) -> qubit 0 flipped   -> flip q0
#   Syndrome 3 = (s0=1, s1=1) -> qubit 1 flipped   -> flip q1
#   Syndrome 2 = (s0=0, s1=1) -> qubit 2 flipped   -> flip q2
SYNDROME_TO_CORRECTION: dict[int, int] = {
    0: ACTION_NONE,
    1: ACTION_FLIP_Q0,
    2: ACTION_FLIP_Q2,
    3: ACTION_FLIP_Q1,
}


def lookup_correction(syndrome: int) -> int:
    """Return the canonical correction action for a given syndrome.

    This is the optimal decoder for the 3-qubit bit-flip code under the
    assumption of independent single-qubit bit-flip errors. It is what both
    MWPM and a well-trained RL agent should converge to.

    Args:
        syndrome: Integer in [0, 3] encoding the 2-bit syndrome as s0 + 2*s1.

    Returns:
        One of ACTION_NONE, ACTION_FLIP_Q0, ACTION_FLIP_Q1, ACTION_FLIP_Q2.

    Raises:
        ValueError: If syndrome is outside [0, 3].
    """
    if syndrome not in SYNDROME_TO_CORRECTION:
        raise ValueError(f"Invalid syndrome {syndrome}; expected 0-3.")
    return SYNDROME_TO_CORRECTION[syndrome]


def apply_correction(
    data_bits: tuple[int, int, int] | np.ndarray,
    action: int,
) -> tuple[int, int, int]:
    """Apply a correction action to a triple of measured data bits.

    Because we only measure the data qubits at the end of the circuit, applying
    an X correction is equivalent to flipping the corresponding classical bit.
    This function operates on measured bitstrings, not quantum states — it is
    used after the fact to compute what the logical outcome would have been
    if the correction had been applied in-circuit.

    Args:
        data_bits: Length-3 sequence of 0s and 1s, the measured data qubits.
        action: One of ACTION_NONE, ACTION_FLIP_Q0, ACTION_FLIP_Q1, ACTION_FLIP_Q2.

    Returns:
        A tuple of 3 ints with the correction applied.

    Raises:
        ValueError: If action is not a valid action ID.
    """
    bits = list(int(b) for b in data_bits)

    if action == ACTION_NONE:
        pass
    elif action == ACTION_FLIP_Q0:
        bits[0] ^= 1
    elif action == ACTION_FLIP_Q1:
        bits[1] ^= 1
    elif action == ACTION_FLIP_Q2:
        bits[2] ^= 1
    else:
        raise ValueError(f"Invalid action {action}; expected 0-3.")

    return tuple(bits)


def majority_vote(data_bits: tuple[int, int, int] | np.ndarray) -> int:
    """Return the majority-vote logical value of 3 data bits.

    This is the final decoding step: after error correction, read the logical
    value as whichever classical bit appears at least twice.

    Args:
        data_bits: Length-3 sequence of 0s and 1s.

    Returns:
        0 or 1, the decoded logical value.
    """
    return int(sum(int(b) for b in data_bits) >= 2)


def is_logical_error(
    data_bits: tuple[int, int, int] | np.ndarray,
    action: int,
    logical_expected: int,
) -> bool:
    """Return True if applying `action` to `data_bits` yields the wrong logical value.

    This is the success criterion used for both RL rewards and benchmarking:
    a decoder's job is to choose an action such that the post-correction
    majority vote matches the originally encoded logical value.

    Args:
        data_bits: The 3 measured data bits after noise (before correction).
        action: The correction action the decoder chose.
        logical_expected: 0 or 1, the logical value that was originally encoded.

    Returns:
        True if the decoded logical value does NOT match what was encoded
        (i.e., a logical error occurred), False otherwise.
    """
    corrected = apply_correction(data_bits, action)
    decoded = majority_vote(corrected)
    return decoded != logical_expected