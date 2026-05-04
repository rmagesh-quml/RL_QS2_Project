"""
End-to-end integration smoke test for qec_rl.

Run this script first on any new machine to verify the full pipeline works
before running full benchmarks or training:

    python examples/integration_test.py

Each section tests one layer of the stack. If a section fails, the error
message tells you exactly which layer is broken so you know where to debug.
Nothing here is slow — the whole script runs in under 30 seconds.

Exit codes:
    0  — all checks passed
    1  — at least one check failed (error printed to stdout)
"""

import sys
import traceback

import numpy as np

PASS = "\033[92m PASS\033[0m"
FAIL = "\033[91m FAIL\033[0m"
HEAD = "\033[94m{}\033[0m"

failures = []


def check(name: str, fn):
    """Run fn(), print PASS/FAIL, accumulate failures."""
    try:
        fn()
        print(f"  {PASS}  {name}")
    except Exception:
        print(f"  {FAIL}  {name}")
        traceback.print_exc()
        failures.append(name)


print(HEAD.format("\n[1/6] circuits.py — encoding, noise, syndrome"))

from qec_rl.circuit import NoiseConfig, run_noisy_experiment, build_encoded_circuit


def test_noise_config_validation():
    try:
        NoiseConfig(noise_type="bit_flip", error_rate=1.5)
        raise AssertionError("Should have raised ValueError for error_rate > 1")
    except ValueError:
        pass


def test_noisy_experiment_returns_correct_shapes():
    config = NoiseConfig(noise_type="bit_flip", error_rate=0.1)
    result = run_noisy_experiment(config, logical_state=0, shots=100, seed=42)
    assert result["syndromes"].shape == (100,), "syndromes wrong shape"
    assert result["data_bits"].shape == (100, 3), "data_bits wrong shape"
    assert result["logical_measured"].shape == (100,), "logical_measured wrong shape"
    assert result["logical_expected"] == 0


def test_syndromes_in_valid_range():
    config = NoiseConfig(noise_type="bit_flip", error_rate=0.1)
    result = run_noisy_experiment(config, logical_state=0, shots=200, seed=0)
    assert set(result["syndromes"]).issubset({0, 1, 2, 3}), \
        f"Unexpected syndrome values: {set(result['syndromes'])}"


def test_zero_noise_gives_zero_syndrome():
    config = NoiseConfig(noise_type="bit_flip", error_rate=0.0)
    result = run_noisy_experiment(config, logical_state=0, shots=50, seed=1)
    assert all(s == 0 for s in result["syndromes"]), \
        f"Expected all-zero syndromes with p=0, got {result['syndromes']}"


def test_encoded_circuit_has_correct_registers():
    qc = build_encoded_circuit(logical_state=0)
    assert len(qc.qregs) == 2, "Expected 2 quantum registers (data, ancilla)"
    assert qc.qregs[0].size == 3, "data register should have 3 qubits"
    assert qc.qregs[1].size == 2, "ancilla register should have 2 qubits"


check("NoiseConfig rejects error_rate > 1", test_noise_config_validation)
check("run_noisy_experiment returns correct shapes", test_noisy_experiment_returns_correct_shapes)
check("syndromes are always in {0,1,2,3}", test_syndromes_in_valid_range)
check("p=0 noise gives all-zero syndromes", test_zero_noise_gives_zero_syndrome)
check("build_encoded_circuit has correct register structure", test_encoded_circuit_has_correct_registers)


print(HEAD.format("\n[2/6] syndrome.py — correction logic"))

from qec_rl.syndrome import (
    ACTION_NONE, ACTION_FLIP_Q0, ACTION_FLIP_Q1, ACTION_FLIP_Q2,
    lookup_correction, apply_correction, majority_vote, is_logical_error,
    SYNDROME_TO_CORRECTION,
)


def test_lookup_table_covers_all_syndromes():
    for s in range(4):
        action = lookup_correction(s)
        assert 0 <= action <= 3, f"Invalid action {action} for syndrome {s}"


def test_syndrome_0_gives_no_correction():
    assert lookup_correction(0) == ACTION_NONE


def test_syndrome_table_is_injective():
    actions = [lookup_correction(s) for s in range(4)]
    assert len(set(actions)) == 4, f"Non-injective table: {actions}"


def test_apply_correction_flips_correct_qubit():
    bits = (0, 0, 0)
    assert apply_correction(bits, ACTION_FLIP_Q0) == (1, 0, 0)
    assert apply_correction(bits, ACTION_FLIP_Q1) == (0, 1, 0)
    assert apply_correction(bits, ACTION_FLIP_Q2) == (0, 0, 1)
    assert apply_correction(bits, ACTION_NONE) == (0, 0, 0)


def test_majority_vote():
    assert majority_vote((0, 0, 0)) == 0
    assert majority_vote((1, 1, 1)) == 1
    assert majority_vote((1, 1, 0)) == 1 
    assert majority_vote((0, 0, 1)) == 0 


def test_is_logical_error_correct_correction():
    assert not is_logical_error((1, 0, 0), ACTION_FLIP_Q0, logical_expected=0)


def test_is_logical_error_wrong_correction():
    assert is_logical_error((1, 0, 0), ACTION_FLIP_Q1, logical_expected=0)


check("lookup table covers all 4 syndromes", test_lookup_table_covers_all_syndromes)
check("syndrome 0 -> ACTION_NONE", test_syndrome_0_gives_no_correction)
check("syndrome table is injective (4 distinct actions)", test_syndrome_table_is_injective)
check("apply_correction flips the right qubit", test_apply_correction_flips_correct_qubit)
check("majority_vote correct on all cases", test_majority_vote)
check("is_logical_error returns False for correct correction", test_is_logical_error_correct_correction)
check("is_logical_error returns True for wrong correction", test_is_logical_error_wrong_correction)


print(HEAD.format("\n[3/6] decoders.py — Random, MWPM, Lookup"))

from qec_rl.decoder import RandomDecoder, MWPMDecoder, LookupDecoder


def test_random_decoder_output_range():
    dec = RandomDecoder(seed=0)
    actions = {dec.decode(s) for _ in range(200) for s in range(4)}
    assert actions.issubset({0, 1, 2, 3}), f"Random decoder produced invalid action"


def test_random_decoder_is_random():
    # With enough samples, all 4 actions should appear.
    dec = RandomDecoder(seed=7)
    actions = [dec.decode(0) for _ in range(100)]
    assert len(set(actions)) > 1, "Random decoder returned same action every time"


def test_mwpm_matches_lookup():
    mwpm = MWPMDecoder()
    lookup = LookupDecoder()
    for s in range(4):
        assert mwpm.decode(s) == lookup.decode(s), \
            f"MWPM and Lookup disagree on syndrome {s}: " \
            f"mwpm={mwpm.decode(s)}, lookup={lookup.decode(s)}"


def test_lookup_decoder_matches_table():
    lookup = LookupDecoder()
    from qec_rl.syndrome import SYNDROME_TO_CORRECTION
    for s in range(4):
        assert lookup.decode(s) == SYNDROME_TO_CORRECTION[s]


def test_decoders_have_name():
    for dec in [RandomDecoder(), MWPMDecoder(), LookupDecoder()]:
        assert isinstance(dec.name, str) and len(dec.name) > 0


check("RandomDecoder output always in {0,1,2,3}", test_random_decoder_output_range)
check("RandomDecoder is actually random", test_random_decoder_is_random)
check("MWPMDecoder matches LookupDecoder on all syndromes", test_mwpm_matches_lookup)
check("LookupDecoder matches SYNDROME_TO_CORRECTION table", test_lookup_decoder_matches_table)
check("All decoders have non-empty name", test_decoders_have_name)


print(HEAD.format("\n[4/6] env.py — Gymnasium environment"))

from qec_rl.env import QECEnv
from qec_rl.syndrome import NUM_ACTIONS


def test_env_reset_returns_valid_obs():
    env = QECEnv(NoiseConfig("bit_flip", 0.1))
    obs, info = env.reset(seed=42)
    assert 0 <= obs <= 3, f"obs {obs} out of range"
    assert "data_bits" in info
    assert "logical_expected" in info
    env.close()


def test_env_step_terminates():
    env = QECEnv(NoiseConfig("bit_flip", 0.1))
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(ACTION_NONE)
    assert terminated is True
    assert truncated is False
    env.close()


def test_env_step_reward_range():
    env = QECEnv(NoiseConfig("bit_flip", 0.1), step_penalty=0.01)
    env.reset(seed=5)
    rewards = []
    for _ in range(50):
        env.reset()
        _, reward, _, _, _ = env.step(ACTION_NONE)
        rewards.append(reward)
    assert all(-1.1 <= r <= 1.0 for r in rewards), f"Reward out of expected range: {rewards}"
    env.close()


def test_env_spaces_correct():
    env = QECEnv()
    assert env.observation_space.n == 4
    assert env.action_space.n == NUM_ACTIONS
    env.close()


def test_env_correct_action_gets_positive_reward():
    env = QECEnv(NoiseConfig("bit_flip", 0.0), step_penalty=0.0)
    successes = 0
    env.reset(seed=99)
    for _ in range(50):
        obs, _ = env.reset()
        assert obs == 0, "With p=0, syndrome should always be 0"
        _, reward, _, _, _ = env.step(ACTION_NONE)
        if reward > 0:
            successes += 1
    assert successes == 50, f"Expected 50/50 successes with p=0, got {successes}"
    env.close()


check("reset() returns obs in [0,3] and info dict", test_env_reset_returns_valid_obs)
check("step() always terminates episode", test_env_step_terminates)
check("reward always in [-1.01, +1.0]", test_env_step_reward_range)
check("observation_space and action_space are Discrete(4)", test_env_spaces_correct)
check("p=0 + ACTION_NONE always gives positive reward", test_env_correct_action_gets_positive_reward)



print(HEAD.format("\n[5/6] evaluate.py — benchmark classical decoders"))

from qec_rl.evaluate import measure_logical_error_rate, run_benchmark, plot_benchmark


def test_measure_logical_error_rate_random_above_lookup():
    # At any error rate, the lookup decoder should beat random.
    config = NoiseConfig("bit_flip", 0.10)
    random_rate = measure_logical_error_rate(RandomDecoder(), config, shots=2000, seed=0)
    lookup_rate = measure_logical_error_rate(LookupDecoder(), config, shots=2000, seed=0)
    assert lookup_rate < random_rate, \
        f"Expected lookup ({lookup_rate:.4f}) < random ({random_rate:.4f})"


def test_lookup_rate_below_physical_at_low_p():
    # At p=0.05, the lookup decoder should have logical rate well below physical.
    config = NoiseConfig("bit_flip", 0.05)
    logical_rate = measure_logical_error_rate(LookupDecoder(), config, shots=3000, seed=42)
    assert logical_rate < 0.05, \
        f"Lookup logical rate {logical_rate:.4f} not below physical rate 0.05"


def test_run_benchmark_returns_correct_shape():
    rates = [0.05, 0.10]
    decoders = [RandomDecoder(seed=0), LookupDecoder()]
    result = run_benchmark(
        decoders, physical_error_rates=rates,
        shots_per_point=500, seed=0, show_progress=False,
    )
    assert result.logical_error_rates.shape == (2, 2), \
        f"Expected shape (2,2), got {result.logical_error_rates.shape}"
    assert list(result.decoder_names) == ["Random", "Lookup"]


def test_benchmark_to_table_runs():
    rates = [0.05, 0.10]
    result = run_benchmark(
        [LookupDecoder()],
        physical_error_rates=rates,
        shots_per_point=200,
        seed=1,
        show_progress=False,
    )
    table = result.to_table()
    assert "Lookup" in table
    assert "0.0500" in table


def test_plot_benchmark_saves_file(tmp_path=None):
    import tempfile, os
    rates = [0.05, 0.10]
    result = run_benchmark(
        [LookupDecoder()],
        physical_error_rates=rates,
        shots_per_point=200,
        seed=2,
        show_progress=False,
    )
    with tempfile.TemporaryDirectory() as tmp:
        save_path = os.path.join(tmp, "test_plot.png")
        plot_benchmark(result, save_path=save_path, show=False)
        assert os.path.exists(save_path), "Plot file was not created"


check("LookupDecoder beats RandomDecoder at p=0.10", test_measure_logical_error_rate_random_above_lookup)
check("LookupDecoder logical rate < physical rate at p=0.05", test_lookup_rate_below_physical_at_low_p)
check("run_benchmark returns correct array shape", test_run_benchmark_returns_correct_shape)
check("to_table() produces readable output", test_benchmark_to_table_runs)
check("plot_benchmark saves PNG to disk", test_plot_benchmark_saves_file)


print(HEAD.format("\n[6/6] agent.py — DQN training smoke test"))

from qec_rl.agent import train_dqn, RLDecoder


def test_train_dqn_saves_model(tmp_path=None):
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmp:
        save_path = os.path.join(tmp, "dqn_test.zip")
        # 500 timesteps — just enough to verify training runs without error.
        train_dqn(
            noise_config=NoiseConfig("bit_flip", 0.1),
            total_timesteps=500,
            save_path=save_path,
            seed=0,
            verbose=0,
        )
        assert os.path.exists(save_path), "Model file was not saved"


def test_rl_decoder_output_range(tmp_path=None):
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmp:
        save_path = os.path.join(tmp, "dqn_test.zip")
        train_dqn(
            noise_config=NoiseConfig("bit_flip", 0.1),
            total_timesteps=500,
            save_path=save_path,
            seed=1,
            verbose=0,
        )
        dec = RLDecoder(save_path)
        for s in range(4):
            action = dec.decode(s)
            assert 0 <= action <= 3, f"RLDecoder returned invalid action {action}"


def test_rl_decoder_missing_model_raises():
    try:
        RLDecoder("nonexistent/path/model.zip")
        raise AssertionError("Should have raised FileNotFoundError")
    except FileNotFoundError:
        pass


check("train_dqn saves model to disk", test_train_dqn_saves_model)
check("RLDecoder output always in {0,1,2,3}", test_rl_decoder_output_range)
check("RLDecoder raises FileNotFoundError for missing model", test_rl_decoder_missing_model_raises)


total_checks = 0
for section in [5, 7, 5, 5, 5, 3]:
    total_checks += section

print(f"\n{'='*50}")
if not failures:
    print(f"\033[92m  ALL CHECKS PASSED\033[0m  ({total_checks} total)")
    print("  Pipeline is healthy end-to-end.\n")
    sys.exit(0)
else:
    print(f"\033[91m  {len(failures)} CHECK(S) FAILED:\033[0m")
    for f in failures:
        print(f"    - {f}")
    print()
    sys.exit(1)