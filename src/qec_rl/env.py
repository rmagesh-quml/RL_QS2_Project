"""
Gymnasium environment wrapping the 3-qubit bit-flip QEC problem.

This module exposes the syndrome decoding problem as a standard RL environment
so that any Gymnasium-compatible agent (DQN, PPO, A2C, etc.) can train on it
without knowing anything about quantum circuits.

Episode structure:
    1. reset(): sample a fresh noise realization, encode a logical state
       (0 or 1 chosen at random), measure the syndrome, return the syndrome
       as the initial observation.
    2. step(action): treat `action` as the decoder's chosen correction,
       apply it to the measured data bits, check whether the logical value
       was preserved, return the reward and terminate the episode.

This is a bandit-style formulation — one observation, one action, done.
That matches the 3-qubit code's single-round structure. For larger codes
with repeated syndrome extraction, the episode would span multiple steps.

Observation space: Discrete(4) — the four possible 2-bit syndromes.
Action space:      Discrete(4) — none, flip q0, flip q1, flip q2.
Reward:            +1 if correction preserved the logical value,
                   -1 if it caused a logical error,
                   -step_penalty (small) for every non-trivial action to
                   encourage minimal intervention when syndrome is trivial.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from qec_rl.circuit import NoiseConfig, run_noisy_experiment
from qec_rl.syndrome import (
    ACTION_NONE,
    NUM_ACTIONS,
    apply_correction,
    is_logical_error,
    majority_vote,
)


class QECEnv(gym.Env):
    """Gymnasium environment for decoding the 3-qubit bit-flip code.

    The environment pre-samples a buffer of noisy experiments from Qiskit on
    each refill, then feeds them to the agent one at a time. This avoids the
    per-episode overhead of calling the simulator for every reset(), which
    would dominate training wall-clock time.

    Each episode is a single step:
        - `reset()` returns a syndrome (int 0-3) as the observation.
        - `step(action)` applies the action, computes the reward based on
          whether logical information was preserved, and ends the episode.

    Attributes:
        noise_config: The noise channel used when pre-sampling experiments.
        step_penalty: Small negative reward per non-trivial action, breaking
            ties toward doing nothing when the syndrome is 0.
        buffer_size: How many shots to pre-sample per refill.
        observation_space: Discrete(4).
        action_space: Discrete(4).

    Example:
        >>> env = QECEnv(NoiseConfig("bit_flip", 0.1))
        >>> obs, info = env.reset(seed=42)
        >>> obs, reward, terminated, truncated, info = env.step(action=1)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        noise_config: NoiseConfig | None = None,
        step_penalty: float = 0.01,
        buffer_size: int = 2048,
    ) -> None:
        super().__init__()
        self.noise_config = noise_config or NoiseConfig()
        self.step_penalty = float(step_penalty)
        self.buffer_size = int(buffer_size)

        # Gymnasium spaces: 4 possible syndromes, 4 possible actions.
        self.observation_space = spaces.Discrete(4)
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        # Per-shot state, refreshed by _refill_buffer.
        self._rng = np.random.default_rng()
        self._buffer_syndromes: np.ndarray | None = None
        self._buffer_data_bits: np.ndarray | None = None
        self._buffer_logical: np.ndarray | None = None
        self._buffer_index = 0

        # Per-episode state, set by reset().
        self._current_syndrome: int = 0
        self._current_data_bits: tuple[int, int, int] = (0, 0, 0)
        self._current_logical: int = 0

    def _refill_buffer(self) -> None:
        """Run Qiskit once to generate `buffer_size` noise realizations.

        We pick logical_state (0 or 1) uniformly at random for each refill.
        This means across many episodes the agent sees balanced exposure to
        both logical values, preventing any shortcut of the form "always
        output the majority bit." Within a single refill all shots share
        the same logical_state — that's acceptable because the agent only
        observes the syndrome, never the logical state directly.
        """
        logical_state = int(self._rng.integers(0, 2))

        result = run_noisy_experiment(
            config=self.noise_config,
            logical_state=logical_state,
            shots=self.buffer_size,
            # Seed per-refill from our Generator so refills are distinct
            # but reproducible if the env was seeded.
            seed=int(self._rng.integers(0, 2**31 - 1)),
        )
        self._buffer_syndromes = result["syndromes"]
        self._buffer_data_bits = result["data_bits"]
        self._buffer_logical = np.full(
            self.buffer_size, fill_value=logical_state, dtype=np.int8
        )
        self._buffer_index = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        """Start a new episode and return the initial observation.

        Follows the Gymnasium reset() contract: keyword-only arguments,
        returns (observation, info_dict).

        Args:
            seed: Optional seed to reinitialize the internal RNG. Only the
                first reset() after construction needs it; subsequent calls
                can pass None and will continue the RNG stream.
            options: Unused here; kept for Gymnasium compatibility.

        Returns:
            A tuple (syndrome, info). `syndrome` is the integer observation
            in [0, 3]. `info` contains diagnostic data like the true data
            bits and logical state, useful for debugging but hidden from
            the agent.
        """
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Refill the buffer if exhausted or not yet initialized.
        if (
            self._buffer_syndromes is None
            or self._buffer_index >= self.buffer_size
        ):
            self._refill_buffer()

        # Pull the next pre-sampled shot.
        assert self._buffer_syndromes is not None  # for type-checkers
        assert self._buffer_data_bits is not None
        assert self._buffer_logical is not None

        i = self._buffer_index
        self._current_syndrome = int(self._buffer_syndromes[i])
        self._current_data_bits = tuple(int(b) for b in self._buffer_data_bits[i])
        self._current_logical = int(self._buffer_logical[i])
        self._buffer_index += 1

        info = {
            "data_bits": self._current_data_bits,
            "logical_expected": self._current_logical,
            "pre_correction_logical": majority_vote(self._current_data_bits),
        }
        return self._current_syndrome, info

    def step(
        self, action: int
    ) -> tuple[int, float, bool, bool, dict[str, Any]]:
        """Apply the chosen correction and return the transition.

        Reward structure (all three pieces are pedagogically motivated):
            +1  if the corrected data decodes to the correct logical value.
            -1  if the correction results in a logical error.
            -step_penalty if the action was non-trivial (flipped a qubit).

        The step penalty is small (default 0.01) so it does not dominate the
        correctness signal, but large enough to break ties: when the syndrome
        is 0 and no correction is needed, doing nothing yields +1.0 while
        any flip yields +1.0 - step_penalty. This nudges the agent toward
        minimal intervention, which is the correct policy for the trivial
        syndrome.

        Args:
            action: Integer in [0, 3], one of the ACTION_* constants.

        Returns:
            Tuple (next_obs, reward, terminated, truncated, info).
            - next_obs: unused (episode ends), returned as the current syndrome
              for Gymnasium compatibility.
            - reward: float per above.
            - terminated: always True — this is a single-step environment.
            - truncated: always False — we never cut episodes short artificially.
            - info: diagnostics including whether the decoding succeeded.
        """
        action = int(action)
        if not 0 <= action < NUM_ACTIONS:
            raise ValueError(
                f"Action {action} out of range [0, {NUM_ACTIONS})."
            )

        error_occurred = is_logical_error(
            self._current_data_bits, action, self._current_logical
        )
        corrected_bits = apply_correction(self._current_data_bits, action)

        base_reward = -1.0 if error_occurred else 1.0
        action_cost = self.step_penalty if action != ACTION_NONE else 0.0
        reward = base_reward - action_cost

        terminated = True
        truncated = False
        info = {
            "data_bits": self._current_data_bits,
            "corrected_bits": corrected_bits,
            "logical_expected": self._current_logical,
            "logical_error": error_occurred,
            "action_taken": action,
        }
        return self._current_syndrome, reward, terminated, truncated, info

    def close(self) -> None:
        """Release resources. No-op for this environment."""
        # Qiskit simulators are garbage-collected by Python; nothing to do.
        self._buffer_syndromes = None
        self._buffer_data_bits = None
        self._buffer_logical = None