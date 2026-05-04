"""
Example 1: Noisy circuit simulation.

Shows what the raw output of the quantum simulation looks like
syndromes, data bits, and logical measurements — before any decoding.

Run with:
    python examples/01_noisy_circuit.py
"""

from qec_rl.circuit import NoiseConfig, run_noisy_experiment
from qec_rl.syndrome import SYNDROME_TO_CORRECTION

config = NoiseConfig(noise_type="bit_flip", error_rate=0.1)
result = run_noisy_experiment(config, logical_state=0, shots=10, seed=42)

print("3-qubit bit-flip code — 10 noisy shots at p=0.10")
print("Encoded logical state: 0 (should be |000>)")
print()
print(f"{'Shot':>4}  {'Syndrome':>8}  {'Data bits':>10}  {'Logical read':>12}  {'Correct?':>8}")
print("-" * 55)

for i in range(10):
    s = int(result["syndromes"][i])
    d = tuple(result["data_bits"][i])
    lm = int(result["logical_measured"][i])
    correct = "YES" if lm == result["logical_expected"] else "NO"
    correction = SYNDROME_TO_CORRECTION[s]
    print(f"{i+1:>4}  {s:>8}  {str(d):>10}  {lm:>12}  {correct:>8}  (correct action: {correction})")