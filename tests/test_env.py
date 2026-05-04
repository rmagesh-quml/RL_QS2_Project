"""
Unit tests for qec_rl.env

"""

import pytest
import numpy as np

from qec_rl.circuit import NoiseConfig
from qec_rl.env import QECEnv
from qec_rl.syndrome import ACTION_NONE, ACTION_FLIP_Q0, NUM_ACTIONS


class TestQECEnvConstruction:
    def test_default_construction(self):
        env = QECEnv()
        assert env is not None
        env.close()

    def test_observation_space_is_discrete_4(self):
        env = QECEnv()
        assert env.observation_space.n == 4
        env.close()

    def test_action_space_is_discrete_4(self):
        env = QECEnv()
        assert env.action_space.n == NUM_ACTIONS
        env.close()

    def test_step_penalty_stored(self):
        env = QECEnv(step_penalty=0.05)
        assert env.step_penalty == 0.05
        env.close()

    def test_buffer_size_stored(self):
        env = QECEnv(buffer_size=512)
        assert env.buffer_size == 512
        env.close()


class TestQECEnvReset:
    @pytest.fixture
    def env(self):
        e = QECEnv(NoiseConfig("bit_flip", 0.1), buffer_size=256)
        yield e
        e.close()

    def test_returns_tuple(self, env):
        result = env.reset(seed=0)
        assert isinstance(result, tuple) and len(result) == 2

    def test_obs_is_int_in_range(self, env):
        obs, _ = env.reset(seed=0)
        assert isinstance(obs, int) and 0 <= obs <= 3

    def test_info_has_required_keys(self, env):
        _, info = env.reset(seed=0)
        assert "data_bits" in info
        assert "logical_expected" in info
        assert "pre_correction_logical" in info

    def test_info_data_bits_length(self, env):
        _, info = env.reset(seed=0)
        assert len(info["data_bits"]) == 3

    def test_info_logical_expected_binary(self, env):
        _, info = env.reset(seed=0)
        assert info["logical_expected"] in (0, 1)

    def test_multiple_resets_work(self, env):
        for seed in range(10):
            obs, _ = env.reset(seed=seed)
            assert 0 <= obs <= 3

    def test_seeded_reset_reproducible(self):
        env1 = QECEnv(NoiseConfig("bit_flip", 0.1), buffer_size=256)
        env2 = QECEnv(NoiseConfig("bit_flip", 0.1), buffer_size=256)
        obs1, info1 = env1.reset(seed=42)
        obs2, info2 = env2.reset(seed=42)
        assert obs1 == obs2
        assert info1["data_bits"] == info2["data_bits"]
        env1.close()
        env2.close()


class TestQECEnvStep:
    @pytest.fixture
    def env(self):
        e = QECEnv(NoiseConfig("bit_flip", 0.1), step_penalty=0.01)
        e.reset(seed=0)
        yield e
        e.close()

    def test_returns_five_tuple(self, env):
        assert len(env.step(ACTION_NONE)) == 5

    def test_terminated_is_true(self, env):
        _, _, terminated, _, _ = env.step(ACTION_NONE)
        assert terminated is True

    def test_truncated_is_false(self, env):
        _, _, _, truncated, _ = env.step(ACTION_NONE)
        assert truncated is False

    def test_reward_is_float(self, env):
        _, reward, _, _, _ = env.step(ACTION_NONE)
        assert isinstance(reward, float)

    def test_reward_in_valid_range(self, env):
        for action in range(NUM_ACTIONS):
            env.reset()
            _, reward, _, _, _ = env.step(action)
            assert -1.1 <= reward <= 1.0

    def test_info_has_required_keys(self, env):
        _, _, _, _, info = env.step(ACTION_NONE)
        assert "logical_error" in info
        assert "action_taken" in info
        assert "corrected_bits" in info

    def test_action_taken_stored_in_info(self, env):
        _, _, _, _, info = env.step(ACTION_FLIP_Q0)
        assert info["action_taken"] == ACTION_FLIP_Q0

    def test_invalid_action_raises(self, env):
        with pytest.raises(ValueError):
            env.step(99)


class TestQECEnvRewardSignal:
    def test_no_noise_action_none_always_positive(self):
        env = QECEnv(NoiseConfig("bit_flip", 0.0), step_penalty=0.0)
        successes = 0
        env.reset(seed=0)
        for _ in range(50):
            env.reset()
            _, reward, _, _, _ = env.step(ACTION_NONE)
            if reward > 0:
                successes += 1
        env.close()
        assert successes == 50

    def test_step_penalty_zero_for_action_none(self):
        env = QECEnv(NoiseConfig("bit_flip", 0.0), step_penalty=0.5)
        env.reset(seed=0)
        _, reward, _, _, _ = env.step(ACTION_NONE)
        assert reward == 1.0
        env.close()


class TestQECEnvBuffer:
    def test_buffer_exhaustion_triggers_refill(self):
        env = QECEnv(NoiseConfig("bit_flip", 0.1), buffer_size=10)
        env.reset(seed=0)
        for _ in range(25):
            obs, _ = env.reset()
            assert 0 <= obs <= 3
            env.step(ACTION_NONE)
        env.close()

    def test_close_clears_buffer(self):
        env = QECEnv()
        env.reset(seed=0)
        env.close()
        assert env._buffer_syndromes is None
        assert env._buffer_data_bits is None