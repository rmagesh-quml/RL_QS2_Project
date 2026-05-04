"""
Classical baseline decoders for the 3-qubit bit-flip code.

This module defines a common decoder interface and provides two implementations
that the RL agent will be benchmarked against:

    - RandomDecoder: picks a correction uniformly at random regardless of the
      syndrome. Serves as the performance floor — any decoder worth its name
      should beat this.

    - MWPMDecoder: Minimum Weight Perfect Matching via the PyMatching library.
      For the 3-qubit bit-flip code this degenerates to the canonical lookup
      table, but we keep the interface intact so the same code path scales to
      larger codes where MWPM does real work.

All decoders share a `decode(syndrome) -> action` method. This lets the
evaluation harness swap them in and out without special cases.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from pymatching import Matching

from qec_rl.syndrome import (
    ACTION_NONE,
    ACTION_FLIP_Q0,
    ACTION_FLIP_Q1,
    ACTION_FLIP_Q2,
    NUM_ACTIONS,
    lookup_correction,
)


class Decoder(ABC):
    """Abstract base class for all syndrome decoders.

    Every decoder — random, MWPM, RL — implements a single method that
    takes a syndrome in [0, 3] and returns a correction action.

    Using a common interface means the evaluation harness treats all
    decoders identically, so benchmarking is apples-to-apples.
    """

    @abstractmethod
    def decode(self, syndrome: int) -> int:
        """Return the correction action for a given syndrome.

        Args:
            syndrome: Integer in [0, 3] encoding the 2-bit syndrome.

        Returns:
            One of ACTION_NONE, ACTION_FLIP_Q0, ACTION_FLIP_Q1, ACTION_FLIP_Q2.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Short human-readable name, used in plots and result tables."""
        ...


class RandomDecoder(Decoder):
    """Decoder that ignores the syndrome and picks a uniform random action.

    This is the intentional floor of performance. Any decoder that uses
    syndrome information should comfortably beat this. Beating the random
    decoder confirms the benchmarking pipeline is wired up correctly — if
    the RL agent can't even beat random, something is broken upstream.

    The randomness uses a local numpy Generator so results are reproducible
    with a seed.

    Example:
        >>> dec = RandomDecoder(seed=42)
        >>> dec.decode(syndrome=2)  # any of 0-3 with equal probability
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)

    def decode(self, syndrome: int) -> int:
        # Note: the syndrome argument is intentionally unused.
        return int(self._rng.integers(0, NUM_ACTIONS))

    @property
    def name(self) -> str:
        return "Random"


class MWPMDecoder(Decoder):
    """Minimum Weight Perfect Matching decoder via PyMatching.

    MWPM treats decoding as a graph problem: stabilizer measurements form a
    "check graph" where each stabilizer is a node and each data qubit is an
    edge connecting the two stabilizers it touches. A syndrome marks certain
    nodes as "lit". MWPM finds the minimum-weight set of edges (data qubits
    to flip) such that lighting those edges matches the observed syndrome.

    For the 3-qubit bit-flip code the check graph has:
        - 2 stabilizer nodes (S0 = Z_0 Z_1, S1 = Z_1 Z_2)
        - 1 boundary node (required by PyMatching when some errors affect
          only one stabilizer, like q0 which only appears in S0)
        - 3 edges: q0 (S0 <-> boundary), q1 (S0 <-> S1), q2 (S1 <-> boundary)

    Given a 2-bit syndrome, PyMatching returns a length-3 binary vector
    indicating which qubits to flip. We translate this into our discrete
    action space (ACTION_NONE / ACTION_FLIP_Qi). For this code, PyMatching
    will always output a single-qubit flip or no flip, which fits cleanly.

    Attributes:
        _matcher: The PyMatching Matching object that solves the graph problem.
    """

    def __init__(self) -> None:
        # Build a graph with 2 stabilizers and 3 data-qubit edges. Each edge
        # has weight 1 because all single-qubit errors are equally likely
        # under our symmetric noise model. The `fault_ids` label lets us
        # recover which qubits to flip after matching.
        self._matcher = Matching()

        # Edge for q0: connects S0 to the virtual boundary (since q0 only
        # appears in stabilizer S0).
        self._matcher.add_boundary_edge(0, fault_ids={0}, weight=1.0)

        # Edge for q1: connects S0 to S1 (q1 appears in both stabilizers).
        self._matcher.add_edge(0, 1, fault_ids={1}, weight=1.0)

        # Edge for q2: connects S1 to the boundary.
        self._matcher.add_boundary_edge(1, fault_ids={2}, weight=1.0)

    def decode(self, syndrome: int) -> int:
        # Convert syndrome integer (s0 + 2*s1) back into a 2-bit vector
        # that PyMatching expects.
        s0 = syndrome & 1
        s1 = (syndrome >> 1) & 1
        syndrome_vec = np.array([s0, s1], dtype=np.uint8)

        # PyMatching returns a length-3 binary array: which qubits to flip.
        correction = self._matcher.decode(syndrome_vec)

        # Translate from qubit-index to our action encoding.
        flipped = np.flatnonzero(correction)
        if len(flipped) == 0:
            return ACTION_NONE
        if len(flipped) == 1:
            qubit = int(flipped[0])
            return (ACTION_FLIP_Q0, ACTION_FLIP_Q1, ACTION_FLIP_Q2)[qubit]
        # For the 3-qubit code this branch should never fire, because every
        # syndrome is produced by at most one single-qubit error. We keep it
        # as defensive programming — if PyMatching ever returns a multi-qubit
        # correction (on a larger code in a future extension), we fail loudly.
        raise RuntimeError(
            f"MWPM returned multi-qubit correction {correction.tolist()} "
            f"for syndrome {syndrome}; 3-qubit bit-flip code should not need this."
        )

    @property
    def name(self) -> str:
        return "MWPM"


class LookupDecoder(Decoder):
    """Decoder that uses the canonical syndrome-to-correction lookup table.

    This is the theoretical optimum for the 3-qubit bit-flip code under the
    assumption of independent single-qubit errors. We include it as a
    sanity-check baseline: MWPM and a well-trained RL agent should both
    converge to identical decisions as this decoder.

    This is essentially a one-line wrapper around syndrome.lookup_correction,
    lifted into the Decoder interface so the evaluation harness can treat it
    like any other decoder.
    """

    def decode(self, syndrome: int) -> int:
        return lookup_correction(syndrome)

    @property
    def name(self) -> str:
        return "Lookup"