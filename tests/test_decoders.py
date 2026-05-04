"""
Unit tests for qec_rl.decoders

"""

import pytest
import numpy as np

from qec_rl.decoder import Decoder, RandomDecoder, MWPMDecoder, LookupDecoder
from qec_rl.syndrome import (
    ACTION_NONE, ACTION_FLIP_Q0, ACTION_FLIP_Q1, ACTION_FLIP_Q2,
    NUM_ACTIONS, SYNDROME_TO_CORRECTION,
)


class TestDecoderABC:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            Decoder()

    def test_subclass_without_decode_raises(self):
        class BadDecoder(Decoder):
            @property
            def name(self):
                return "bad"
        with pytest.raises(TypeError):
            BadDecoder()

    def test_subclass_without_name_raises(self):
        class BadDecoder(Decoder):
            def decode(self, syndrome):
                return ACTION_NONE
        with pytest.raises(TypeError):
            BadDecoder()


class TestRandomDecoder:
    def test_output_always_in_range(self):
        dec = RandomDecoder(seed=0)
        for _ in range(200):
            for s in range(4):
                assert 0 <= dec.decode(s) < NUM_ACTIONS

    def test_all_four_actions_appear(self):
        dec = RandomDecoder(seed=42)
        actions = {dec.decode(0) for _ in range(100)}
        assert len(actions) == 4

    def test_reproducible_with_seed(self):
        dec1 = RandomDecoder(seed=7)
        dec2 = RandomDecoder(seed=7)
        assert [dec1.decode(0) for _ in range(20)] == [dec2.decode(0) for _ in range(20)]

    def test_different_seeds_differ(self):
        dec1 = RandomDecoder(seed=0)
        dec2 = RandomDecoder(seed=999)
        assert [dec1.decode(0) for _ in range(30)] != [dec2.decode(0) for _ in range(30)]

    def test_name_is_string(self):
        assert isinstance(RandomDecoder().name, str)

    def test_returns_int(self):
        assert isinstance(RandomDecoder(seed=0).decode(0), int)


class TestMWPMDecoder:
    @pytest.fixture
    def mwpm(self):
        return MWPMDecoder()

    @pytest.mark.parametrize("syndrome", [0, 1, 2, 3])
    def test_output_in_range(self, mwpm, syndrome):
        assert 0 <= mwpm.decode(syndrome) < NUM_ACTIONS

    def test_syndrome_zero_returns_none(self, mwpm):
        assert mwpm.decode(0) == ACTION_NONE

    def test_syndrome_one_flips_q0(self, mwpm):
        assert mwpm.decode(1) == ACTION_FLIP_Q0

    def test_syndrome_three_flips_q1(self, mwpm):
        assert mwpm.decode(3) == ACTION_FLIP_Q1

    def test_syndrome_two_flips_q2(self, mwpm):
        assert mwpm.decode(2) == ACTION_FLIP_Q2

    def test_returns_int(self, mwpm):
        assert isinstance(mwpm.decode(0), int)


class TestLookupDecoder:
    @pytest.fixture
    def lookup(self):
        return LookupDecoder()

    @pytest.mark.parametrize("syndrome", [0, 1, 2, 3])
    def test_matches_syndrome_to_correction_table(self, lookup, syndrome):
        assert lookup.decode(syndrome) == SYNDROME_TO_CORRECTION[syndrome]

    def test_syndrome_zero_returns_none(self, lookup):
        assert lookup.decode(0) == ACTION_NONE

    def test_returns_int(self, lookup):
        assert isinstance(lookup.decode(1), int)


class TestDecoderAgreement:
    def test_mwpm_matches_lookup_on_all_syndromes(self):
        mwpm = MWPMDecoder()
        lookup = LookupDecoder()
        for s in range(4):
            assert mwpm.decode(s) == lookup.decode(s), \
                f"Disagree on syndrome {s}: mwpm={mwpm.decode(s)}, lookup={lookup.decode(s)}"

    def test_all_three_decoders_have_distinct_names(self):
        names = {RandomDecoder().name, MWPMDecoder().name, LookupDecoder().name}
        assert len(names) == 3