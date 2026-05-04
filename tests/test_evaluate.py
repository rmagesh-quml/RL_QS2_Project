"""
Unit tests for qec_rl.evaluate.
"""

import os
import tempfile
import pytest
import numpy as np

from qec_rl.circuit import NoiseConfig
from qec_rl.decoder import RandomDecoder, LookupDecoder
from qec_rl.evaluate import (
    BenchmarkResult,
    measure_logical_error_rate,
    run_benchmark,
    plot_benchmark,
)


class TestBenchmarkResult:
    @pytest.fixture
    def sample_result(self):
        return BenchmarkResult(
            physical_error_rates=np.array([0.05, 0.10]),
            decoder_names=["Random", "Lookup"],
            logical_error_rates=np.array([[0.40, 0.45], [0.03, 0.08]]),
            shots_per_point=1000,
            noise_type="bit_flip",
        )

    def test_construction(self, sample_result):
        assert sample_result is not None

    def test_to_table_contains_decoder_names(self, sample_result):
        table = sample_result.to_table()
        assert "Random" in table and "Lookup" in table

    def test_to_table_contains_rates(self, sample_result):
        table = sample_result.to_table()
        assert "0.0500" in table and "0.1000" in table

    def test_to_table_returns_string(self, sample_result):
        assert isinstance(sample_result.to_table(), str)

    def test_logical_error_rates_shape(self, sample_result):
        assert sample_result.logical_error_rates.shape == (2, 2)

    def test_first_row_is_first_decoder(self, sample_result):
        assert sample_result.logical_error_rates[0, 0] == pytest.approx(0.40)


class TestMeasureLogicalErrorRate:
    def test_output_in_zero_one(self):
        rate = measure_logical_error_rate(
            LookupDecoder(), NoiseConfig("bit_flip", 0.1), shots=200, seed=0
        )
        assert 0.0 <= rate <= 1.0

    def test_zero_noise_lookup_zero_errors(self):
        rate = measure_logical_error_rate(
            LookupDecoder(), NoiseConfig("bit_flip", 0.0), shots=100, seed=0
        )
        assert rate == 0.0

    def test_lookup_beats_random(self):
        cfg = NoiseConfig("bit_flip", 0.1)
        random_rate = measure_logical_error_rate(RandomDecoder(seed=0), cfg, shots=1000, seed=0)
        lookup_rate = measure_logical_error_rate(LookupDecoder(), cfg, shots=1000, seed=0)
        assert lookup_rate < random_rate

    def test_lookup_logical_below_physical(self):
        rate = measure_logical_error_rate(
            LookupDecoder(), NoiseConfig("bit_flip", 0.05), shots=2000, seed=42
        )
        assert rate < 0.05

    def test_reproducible_with_seed(self):
        cfg = NoiseConfig("bit_flip", 0.1)
        r1 = measure_logical_error_rate(LookupDecoder(), cfg, shots=100, seed=7)
        r2 = measure_logical_error_rate(LookupDecoder(), cfg, shots=100, seed=7)
        assert r1 == r2

    def test_error_rate_increases_with_noise(self):
        lookup = LookupDecoder()
        rate_low = measure_logical_error_rate(lookup, NoiseConfig("bit_flip", 0.01), shots=2000, seed=0)
        rate_high = measure_logical_error_rate(lookup, NoiseConfig("bit_flip", 0.15), shots=2000, seed=0)
        assert rate_low < rate_high


class TestRunBenchmark:
    @pytest.fixture(scope="class")
    def small_result(self):
        return run_benchmark(
            decoders=[RandomDecoder(seed=0), LookupDecoder()],
            physical_error_rates=[0.05, 0.10],
            shots_per_point=300,
            seed=0,
            show_progress=False,
        )

    def test_logical_rates_shape(self, small_result):
        assert small_result.logical_error_rates.shape == (2, 2)

    def test_decoder_names_stored(self, small_result):
        assert small_result.decoder_names == ["Random", "Lookup"]

    def test_physical_rates_stored(self, small_result):
        np.testing.assert_array_almost_equal(small_result.physical_error_rates, [0.05, 0.10])

    def test_shots_per_point_stored(self, small_result):
        assert small_result.shots_per_point == 300

    def test_noise_type_stored(self, small_result):
        assert small_result.noise_type == "bit_flip"

    def test_all_rates_in_zero_one(self, small_result):
        assert np.all(small_result.logical_error_rates >= 0.0)
        assert np.all(small_result.logical_error_rates <= 1.0)

    def test_lookup_row_below_random_row(self, small_result):
        assert small_result.logical_error_rates[1].mean() < small_result.logical_error_rates[0].mean()

    def test_accepts_single_decoder(self):
        result = run_benchmark(
            decoders=[LookupDecoder()],
            physical_error_rates=[0.10],
            shots_per_point=100,
            show_progress=False,
        )
        assert result.logical_error_rates.shape == (1, 1)


class TestPlotBenchmark:
    @pytest.fixture(scope="class")
    def result(self):
        return run_benchmark(
            decoders=[LookupDecoder()],
            physical_error_rates=[0.05, 0.10],
            shots_per_point=100,
            show_progress=False,
        )

    def test_returns_figure(self, result):
        import matplotlib.pyplot as plt
        fig = plot_benchmark(result, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_png_to_disk(self, result):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test.png")
            plot_benchmark(result, save_path=path, show=False)
            assert os.path.exists(path) and os.path.getsize(path) > 0

    def test_creates_parent_directory(self, result):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "nested", "plot.png")
            plot_benchmark(result, save_path=path, show=False)
            assert os.path.exists(path)

    def test_custom_title(self, result):
        import matplotlib.pyplot as plt
        fig = plot_benchmark(result, title="My Custom Title", show=False)
        assert fig.axes[0].get_title() == "My Custom Title"
        plt.close(fig)