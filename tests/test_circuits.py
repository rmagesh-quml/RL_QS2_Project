"""
Unit tests for qec_rl.circuits

"""

import pytest
import numpy as np

from qec_rl.circuit import (
    NoiseConfig,
    build_encoded_circuit,
    run_noisy_experiment,
)


class TestNoiseConfig:
    def test_default_values(self):
        cfg = NoiseConfig()
        assert cfg.noise_type == "bit_flip"
        assert cfg.error_rate == 0.1

    def test_custom_values(self):
        cfg = NoiseConfig(noise_type="depolarizing", error_rate=0.05)
        assert cfg.noise_type == "depolarizing"
        assert cfg.error_rate == 0.05

    def test_error_rate_zero_allowed(self):
        cfg = NoiseConfig(error_rate=0.0)
        assert cfg.error_rate == 0.0

    def test_error_rate_one_allowed(self):
        cfg = NoiseConfig(error_rate=1.0)
        assert cfg.error_rate == 1.0

    def test_error_rate_above_one_raises(self):
        with pytest.raises(ValueError, match="error_rate"):
            NoiseConfig(error_rate=1.01)

    def test_error_rate_below_zero_raises(self):
        with pytest.raises(ValueError, match="error_rate"):
            NoiseConfig(error_rate=-0.01)

    def test_frozen_prevents_mutation(self):
        cfg = NoiseConfig()
        with pytest.raises(Exception):
            cfg.error_rate = 0.5


class TestBuildEncodedCircuit:
    def test_has_two_quantum_registers(self):
        qc = build_encoded_circuit(0)
        assert len(qc.qregs) == 2

    def test_data_register_has_three_qubits(self):
        qc = build_encoded_circuit(0)
        assert qc.qregs[0].size == 3

    def test_ancilla_register_has_two_qubits(self):
        qc = build_encoded_circuit(0)
        assert qc.qregs[1].size == 2

    def test_has_two_classical_registers(self):
        qc = build_encoded_circuit(0)
        assert len(qc.cregs) == 2

    def test_logical_state_zero(self):
        qc = build_encoded_circuit(logical_state=0)
        assert qc is not None

    def test_logical_state_one(self):
        qc = build_encoded_circuit(logical_state=1)
        assert qc is not None

    def test_invalid_logical_state_raises(self):
        with pytest.raises(ValueError):
            build_encoded_circuit(logical_state=2)

    def test_negative_logical_state_raises(self):
        with pytest.raises(ValueError):
            build_encoded_circuit(logical_state=-1)


class TestRunNoisyExperiment:
    @pytest.fixture
    def default_config(self):
        return NoiseConfig(noise_type="bit_flip", error_rate=0.1)

    def test_syndromes_shape(self, default_config):
        result = run_noisy_experiment(default_config, shots=64, seed=0)
        assert result["syndromes"].shape == (64,)

    def test_data_bits_shape(self, default_config):
        result = run_noisy_experiment(default_config, shots=64, seed=0)
        assert result["data_bits"].shape == (64, 3)

    def test_logical_measured_shape(self, default_config):
        result = run_noisy_experiment(default_config, shots=64, seed=0)
        assert result["logical_measured"].shape == (64,)

    def test_logical_expected_stored(self, default_config):
        result = run_noisy_experiment(default_config, logical_state=1, shots=16, seed=0)
        assert result["logical_expected"] == 1

    def test_syndromes_in_range(self, default_config):
        result = run_noisy_experiment(default_config, shots=200, seed=1)
        assert set(result["syndromes"]).issubset({0, 1, 2, 3})

    def test_data_bits_binary(self, default_config):
        result = run_noisy_experiment(default_config, shots=100, seed=2)
        assert set(result["data_bits"].flatten()).issubset({0, 1})

    def test_logical_measured_binary(self, default_config):
        result = run_noisy_experiment(default_config, shots=100, seed=3)
        assert set(result["logical_measured"]).issubset({0, 1})

    def test_zero_noise_all_syndromes_zero(self):
        cfg = NoiseConfig(error_rate=0.0)
        result = run_noisy_experiment(cfg, shots=50, seed=0)
        assert np.all(result["syndromes"] == 0)

    def test_zero_noise_logical_state_preserved_0(self):
        cfg = NoiseConfig(error_rate=0.0)
        result = run_noisy_experiment(cfg, logical_state=0, shots=50, seed=0)
        assert np.all(result["logical_measured"] == 0)

    def test_zero_noise_logical_state_preserved_1(self):
        cfg = NoiseConfig(error_rate=0.0)
        result = run_noisy_experiment(cfg, logical_state=1, shots=50, seed=0)
        assert np.all(result["logical_measured"] == 1)

    def test_reproducible_with_seed(self, default_config):
        r1 = run_noisy_experiment(default_config, shots=32, seed=99)
        r2 = run_noisy_experiment(default_config, shots=32, seed=99)
        np.testing.assert_array_equal(r1["syndromes"], r2["syndromes"])

    def test_different_seeds_give_different_results(self, default_config):
        r1 = run_noisy_experiment(default_config, shots=64, seed=0)
        r2 = run_noisy_experiment(default_config, shots=64, seed=1)
        assert not np.array_equal(r1["syndromes"], r2["syndromes"])

    def test_depolarizing_noise(self):
        cfg = NoiseConfig(noise_type="depolarizing", error_rate=0.1)
        result = run_noisy_experiment(cfg, shots=64, seed=0)
        assert result["syndromes"].shape == (64,)

    def test_phase_flip_noise(self):
        cfg = NoiseConfig(noise_type="phase_flip", error_rate=0.5)
        result = run_noisy_experiment(cfg, shots=100, seed=0)
        assert np.all(result["syndromes"] == 0)