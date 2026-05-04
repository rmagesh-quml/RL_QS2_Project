"""
Unit tests for qec_rl.syndrome

"""

import pytest
import numpy as np

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


class TestConstants:
    def test_num_actions_is_four(self):
        assert NUM_ACTIONS == 4

    def test_action_values_distinct(self):
        actions = {ACTION_NONE, ACTION_FLIP_Q0, ACTION_FLIP_Q1, ACTION_FLIP_Q2}
        assert len(actions) == 4

    def test_action_values_in_range(self):
        for a in [ACTION_NONE, ACTION_FLIP_Q0, ACTION_FLIP_Q1, ACTION_FLIP_Q2]:
            assert 0 <= a < NUM_ACTIONS

    def test_syndrome_table_has_four_entries(self):
        assert len(SYNDROME_TO_CORRECTION) == 4

    def test_syndrome_table_keys_are_zero_to_three(self):
        assert set(SYNDROME_TO_CORRECTION.keys()) == {0, 1, 2, 3}


class TestLookupCorrection:
    @pytest.mark.parametrize("syndrome", [0, 1, 2, 3])
    def test_valid_syndrome_returns_action(self, syndrome):
        action = lookup_correction(syndrome)
        assert 0 <= action < NUM_ACTIONS

    def test_syndrome_zero_returns_none(self):
        assert lookup_correction(0) == ACTION_NONE

    def test_table_is_injective(self):
        actions = [lookup_correction(s) for s in range(4)]
        assert len(set(actions)) == 4

    def test_matches_syndrome_to_correction_dict(self):
        for s in range(4):
            assert lookup_correction(s) == SYNDROME_TO_CORRECTION[s]

    @pytest.mark.parametrize("bad_syndrome", [-1, 4, 100])
    def test_invalid_syndrome_raises_value_error(self, bad_syndrome):
        with pytest.raises(ValueError):
            lookup_correction(bad_syndrome)


class TestApplyCorrection:
    def test_action_none_no_change(self):
        assert apply_correction((0, 0, 0), ACTION_NONE) == (0, 0, 0)
        assert apply_correction((1, 1, 1), ACTION_NONE) == (1, 1, 1)

    def test_flip_q0(self):
        assert apply_correction((0, 0, 0), ACTION_FLIP_Q0) == (1, 0, 0)
        assert apply_correction((1, 0, 0), ACTION_FLIP_Q0) == (0, 0, 0)

    def test_flip_q1(self):
        assert apply_correction((0, 0, 0), ACTION_FLIP_Q1) == (0, 1, 0)
        assert apply_correction((0, 1, 0), ACTION_FLIP_Q1) == (0, 0, 0)

    def test_flip_q2(self):
        assert apply_correction((0, 0, 0), ACTION_FLIP_Q2) == (0, 0, 1)
        assert apply_correction((0, 0, 1), ACTION_FLIP_Q2) == (0, 0, 0)

    def test_double_flip_is_identity(self):
        original = (1, 0, 1)
        after = apply_correction(apply_correction(original, ACTION_FLIP_Q1), ACTION_FLIP_Q1)
        assert after == original

    def test_accepts_numpy_array(self):
        bits = np.array([1, 0, 0], dtype=np.int8)
        result = apply_correction(bits, ACTION_FLIP_Q0)
        assert result == (0, 0, 0)

    def test_invalid_action_raises(self):
        with pytest.raises(ValueError):
            apply_correction((0, 0, 0), 99)

    def test_returns_tuple(self):
        result = apply_correction((0, 0, 0), ACTION_NONE)
        assert isinstance(result, tuple)
        assert len(result) == 3


class TestMajorityVote:
    @pytest.mark.parametrize("bits,expected", [
        ((0, 0, 0), 0),
        ((1, 1, 1), 1),
        ((1, 1, 0), 1),
        ((1, 0, 1), 1),
        ((0, 1, 1), 1),
        ((0, 0, 1), 0),
        ((0, 1, 0), 0),
        ((1, 0, 0), 0),
    ])
    def test_all_eight_inputs(self, bits, expected):
        assert majority_vote(bits) == expected

    def test_accepts_numpy_array(self):
        assert majority_vote(np.array([1, 1, 0])) == 1

    def test_returns_int(self):
        assert isinstance(majority_vote((1, 0, 0)), int)


class TestIsLogicalError:
    def test_no_error_action_none_correct(self):
        assert not is_logical_error((0, 0, 0), ACTION_NONE, logical_expected=0)

    def test_no_error_action_none_correct_logical_1(self):
        assert not is_logical_error((1, 1, 1), ACTION_NONE, logical_expected=1)

    def test_q0_flip_correct_correction(self):
        assert not is_logical_error((1, 0, 0), ACTION_FLIP_Q0, logical_expected=0)

    def test_q1_flip_correct_correction(self):
        assert not is_logical_error((0, 1, 0), ACTION_FLIP_Q1, logical_expected=0)

    def test_q2_flip_correct_correction(self):
        assert not is_logical_error((0, 0, 1), ACTION_FLIP_Q2, logical_expected=0)

    def test_q0_flip_wrong_correction_causes_error(self):
        assert is_logical_error((1, 0, 0), ACTION_FLIP_Q1, logical_expected=0)

    def test_q0_flip_no_action_no_logical_error(self):
        # majority of (1,0,0) = 0 which matches expected=0
        assert not is_logical_error((1, 0, 0), ACTION_NONE, logical_expected=0)

    def test_two_qubit_flip_causes_logical_error(self):
        assert is_logical_error((1, 1, 0), ACTION_NONE, logical_expected=0)

    def test_three_qubit_flip_causes_logical_error(self):
        assert is_logical_error((1, 1, 1), ACTION_NONE, logical_expected=0)

    def test_optimal_policy_never_errors_on_single_qubit_flips(self):
        correct_cases = [
            ((0, 0, 0), ACTION_NONE),
            ((1, 0, 0), ACTION_FLIP_Q0),
            ((0, 1, 0), ACTION_FLIP_Q1),
            ((0, 0, 1), ACTION_FLIP_Q2),
        ]
        for bits, action in correct_cases:
            assert not is_logical_error(bits, action, logical_expected=0)