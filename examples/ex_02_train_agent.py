"""
Example 2: Train the DQN agent from scratch.

Trains a DQN decoder on the 3-qubit bit-flip code and saves the model.
After training, shows what action the agent picks for each syndrome
and compares it to the lookup table.

Run with:
    python examples/02_train_agent.py
"""

from qec_rl.circuit import NoiseConfig
from qec_rl.agent import train_dqn, RLDecoder
from qec_rl.syndrome import SYNDROME_TO_CORRECTION, ACTION_NONE

print("Training DQN decoder for 20,000 timesteps...")
train_dqn(
    noise_config=NoiseConfig("bit_flip", 0.1),
    total_timesteps=20_000,
    save_path="models/dqn_qec.zip",
    seed=0,
    verbose=0,
)
print("Training complete. Model saved to models/dqn_qec.zip")
print()

dec = RLDecoder("models/dqn_qec.zip")

print("Learned policy vs canonical lookup table:")
print(f"{'Syndrome':>8}  {'RL action':>10}  {'Lookup action':>14}  {'Match?':>7}")
print("-" * 45)

action_names = {0: "NONE", 1: "FLIP_Q0", 2: "FLIP_Q1", 3: "FLIP_Q2"}
all_match = True
for syndrome in range(4):
    rl_action = dec.decode(syndrome)
    lookup_action = SYNDROME_TO_CORRECTION[syndrome]
    match = rl_action == lookup_action
    if not match:
        all_match = False
    print(f"{syndrome:>8}  {action_names[rl_action]:>10}  {action_names[lookup_action]:>14}  {'YES' if match else 'NO':>7}")

print()
if all_match:
    print("Agent learned the optimal policy on all 4 syndromes.")
else:
    print("Agent did not fully converge — try training longer.")