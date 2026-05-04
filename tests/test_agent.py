"""
Unit tests for qec_rl.agent

"""

import os
import tempfile
import pytest

from qec_rl.circuit import NoiseConfig
from qec_rl.agent import train_dqn, RLDecoder
from qec_rl.syndrome import NUM_ACTIONS


@pytest.fixture(scope="module")
def trained_model_path():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test_model.zip")
        train_dqn(
            noise_config=NoiseConfig("bit_flip", 0.1),
            total_timesteps=500,
            save_path=path,
            seed=0,
            verbose=0,
        )
        yield path


class TestTrainDQN:
    def test_saves_model_file(self, trained_model_path):
        assert os.path.exists(trained_model_path)

    def test_returns_dqn_object(self):
        from stable_baselines3 import DQN
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.zip")
            result = train_dqn(total_timesteps=200, save_path=path, seed=1, verbose=0)
            assert isinstance(result, DQN)

    def test_creates_parent_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "nested", "dir", "model.zip")
            train_dqn(total_timesteps=100, save_path=path, seed=2, verbose=0)
            assert os.path.exists(path)

    def test_accepts_noise_types(self):
        for noise_type in ("bit_flip", "depolarizing"):
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "model.zip")
                train_dqn(
                    noise_config=NoiseConfig(noise_type, 0.1),
                    total_timesteps=100,
                    save_path=path,
                    verbose=0,
                )
                assert os.path.exists(path)


class TestRLDecoder:
    def test_loads_from_path(self, trained_model_path):
        assert RLDecoder(trained_model_path) is not None

    @pytest.mark.parametrize("syndrome", [0, 1, 2, 3])
    def test_decode_output_in_range(self, trained_model_path, syndrome):
        assert 0 <= RLDecoder(trained_model_path).decode(syndrome) < NUM_ACTIONS

    def test_decode_returns_int(self, trained_model_path):
        assert isinstance(RLDecoder(trained_model_path).decode(0), int)

    def test_decode_is_deterministic(self, trained_model_path):
        dec = RLDecoder(trained_model_path)
        results = [dec.decode(2) for _ in range(10)]
        assert len(set(results)) == 1

    def test_name_property(self, trained_model_path):
        dec = RLDecoder(trained_model_path)
        assert isinstance(dec.name, str) and len(dec.name) > 0

    def test_missing_model_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            RLDecoder("totally/nonexistent/path.zip")

    def test_is_decoder_subclass(self, trained_model_path):
        from qec_rl.decoder import Decoder
        assert isinstance(RLDecoder(trained_model_path), Decoder)