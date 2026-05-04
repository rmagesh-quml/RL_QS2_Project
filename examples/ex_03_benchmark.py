"""
Example 3: Full benchmark across all four decoders.

Sweeps physical error rates from 0.01 to 0.20 and measures the logical
error rate of each decoder. Prints a comparison table and saves the plot.

Run with:
    python examples/03_benchmark.py
"""

from qec_rl import NoiseConfig, RandomDecoder, MWPMDecoder, LookupDecoder
from qec_rl.agent import RLDecoder
from qec_rl.evaluate import run_benchmark, plot_benchmark
import os

model_path = "models/dqn_qec.zip"
if not os.path.exists(model_path):
    print(f"No trained model found at {model_path}.")
    print("Run examples/02_train_agent.py first.")
    exit(1)

print("Running benchmark — this takes a few minutes...")
result = run_benchmark(
    decoders=[
        RandomDecoder(),
        LookupDecoder(),
        MWPMDecoder(),
        RLDecoder(model_path),
    ],
    shots_per_point=5000,
    seed=0,
)

print()
print(result.to_table())
print()

plot_benchmark(result, save_path="results/comparison.png", show=True)
print("Plot saved to results/comparison.png")