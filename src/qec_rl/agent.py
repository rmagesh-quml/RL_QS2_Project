"""
DQN training and inference for the QEC decoding environment.

This module is the bridge between stable-baselines3 and our project. It does
two things:

    1. `train_dqn(...)`: runs DQN training against `QECEnv` and saves the
       trained model to disk.

    2. `RLDecoder`: a Decoder subclass that loads a trained DQN model and
       exposes it via the same `decode(syndrome) -> action` interface as
       the classical decoders. This means the benchmark harness treats the
       RL agent identically to Random / MWPM / Lookup.

DQN is overpowered for the 3-qubit code (4 states, 4 actions, single-step
episodes). A tabular Q-learner would converge faster. We use DQN because:
    - it is the standard RL library tool, demonstrating the pipeline works,
    - it scales unchanged to larger codes where tabular methods cannot, and
    - it lets us study reward design in a familiar framework.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

from qec_rl.circuit import NoiseConfig
from qec_rl.decoder import Decoder
from qec_rl.env import QECEnv
from qec_rl.syndrome import NUM_ACTIONS


def train_dqn(
    noise_config: NoiseConfig | None = None,
    total_timesteps: int = 20_000,
    learning_rate: float = 1e-3,
    buffer_size: int = 10_000,
    learning_starts: int = 500,
    batch_size: int = 64,
    gamma: float = 0.99,
    exploration_fraction: float = 0.3,
    exploration_final_eps: float = 0.05,
    save_path: str | Path = "models/dqn_qec.zip",
    log_path: str | Path | None = None,
    seed: int | None = None,
    verbose: int = 1,
) -> DQN:
    """Train a DQN agent on QECEnv and save it to disk.

    The defaults are tuned for the 3-qubit bit-flip code, where the problem
    is small enough that 20k timesteps is more than sufficient. For larger
    codes, you would scale most of these up.

    Args:
        noise_config: Noise channel for training. Defaults to bit-flip at p=0.1.
        total_timesteps: How many environment steps to train for. Each
            timestep is one (reset, step) pair since episodes are length 1.
        learning_rate: Adam learning rate for the Q-network.
        buffer_size: Replay buffer capacity. Stores past (s, a, r, s') tuples
            for off-policy updates. 10k is plenty here since states recur.
        learning_starts: Collect this many random transitions before any
            gradient updates begin. Lets the buffer fill with diverse data.
        batch_size: Minibatch size for each gradient update.
        gamma: Discount factor. With single-step episodes gamma is irrelevant
            — there's no future reward to discount — but stable-baselines3
            requires a value, and 0.99 is the conventional default.
        exploration_fraction: Fraction of training over which epsilon decays
            from 1.0 to `exploration_final_eps`.
        exploration_final_eps: Final epsilon (random-action probability)
            after the decay schedule completes.
        save_path: Where to write the trained model.
        log_path: Optional directory for Monitor logs (episode rewards/lengths).
        seed: RNG seed for reproducibility.
        verbose: 0 = silent, 1 = info, 2 = debug.

    Returns:
        The trained DQN object. Callers usually do not need this — they can
        load the saved model with `DQN.load(save_path)` or just construct
        an `RLDecoder(save_path)`.
    """
    # Build the training environment.
    env = QECEnv(noise_config=noise_config)
    if log_path is not None:
        log_path = Path(log_path)
        log_path.mkdir(parents=True, exist_ok=True)
        env = Monitor(env, filename=str(log_path / "monitor"))

    # Configure DQN. MlpPolicy = a small fully-connected network, fine for
    # discrete observations of size 4.
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=batch_size,
        gamma=gamma,
        exploration_fraction=exploration_fraction,
        exploration_final_eps=exploration_final_eps,
        seed=seed,
        verbose=verbose,
    )

    model.learn(total_timesteps=total_timesteps)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))

    env.close()
    return model


class RLDecoder(Decoder):
    """Decoder that wraps a trained DQN policy.

    Loads a model saved by `train_dqn` and exposes its action selection via
    the same `decode(syndrome) -> action` interface as the classical decoders.
    This means the benchmark harness can swap RL in and out exactly like it
    swaps in Random or MWPM.

    The decoder runs the policy in deterministic mode (always picks
    argmax_a Q(s, a)) — no exploration noise during evaluation.

    Attributes:
        _model: The loaded stable-baselines3 DQN object.

    Example:
        >>> dec = RLDecoder("models/dqn_qec.zip")
        >>> dec.decode(syndrome=2)
    """

    def __init__(self, model_path: str | Path) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"No DQN model at {model_path}. "
                f"Run train_dqn() first or pass a different path."
            )
        self._model = DQN.load(str(model_path))

    def decode(self, syndrome: int) -> int:
        # stable-baselines3 expects observations as numpy arrays, even for
        # discrete spaces. We wrap the integer syndrome accordingly.
        obs = np.array(syndrome, dtype=np.int64)
        action, _state = self._model.predict(obs, deterministic=True)
        action_int = int(action)
        if not 0 <= action_int < NUM_ACTIONS:
            raise RuntimeError(
                f"DQN returned out-of-range action {action_int}; "
                f"expected [0, {NUM_ACTIONS})."
            )
        return action_int

    @property
    def name(self) -> str:
        return "RL (DQN)"