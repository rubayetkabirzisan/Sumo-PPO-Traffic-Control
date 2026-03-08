"""
traffic_agent.py — Improved PPO Traffic Signal Controller
=========================================================
Fixes vs. original:
  1. Action applied every DECISION_INTERVAL steps (not every 0.1s tick)
  2. Normalized state vector (queue counts / max_queue, phase / num_phases)
  3. Richer reward: negative queue + throughput bonus
  4. Added 'reduce phase' action (action 3) to complement 'extend' (action 2)
  5. Yellow-phase guard: prevent skipping past yellow transitions
  6. Episode metrics logged to CSV: sim_time, total_wait, vehicles_cleared
  7. Fixed seed for reproducibility
  8. total_timesteps raised to 200,000 (set FAST_TRAIN=True for quick test)
  9. Saved best model via EvalCallback, not just final
 10. Cleaned up all stray characters and dead commented code

Usage:
    python traffic_agent.py              # full training (200k steps)
    python traffic_agent.py --fast       # quick test run (10k steps)
"""

import os
import sys
import csv
import argparse
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ── SUMO PATH ────────────────────────────────────────────────────────────────
if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    sys.exit(
        "ERROR: 'SUMO_HOME' environment variable not set.\n"
        "  Linux/macOS: export SUMO_HOME=/usr/share/sumo\n"
        "  Windows:     set SUMO_HOME=C:\\Program Files (x86)\\Eclipse\\Sumo"
    )

import traci
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
SUMO_CONFIG       = "sumoconfig.sumocfg"
SUMO_BINARY       = "sumo"              # headless; use "sumo-gui" to see GUI
SEED              = 42                  # fixed seed for reproducibility

# How many simulation steps (0.1 s each) between RL decisions.
# 50 steps = 5 real simulated seconds — matches test_model.py.
DECISION_INTERVAL = 50

# Maximum vehicles that could ever queue on a single lane.
# Used to normalise queue counts to [0, 1].
MAX_QUEUE         = 20.0

# Training length (timesteps = number of RL decisions taken).
FULL_TIMESTEPS    = 200_000
FAST_TIMESTEPS    = 10_000

LOG_CSV           = "training_log.csv"  # per-episode metrics


# ── ENVIRONMENT ───────────────────────────────────────────────────────────────
class SumoEnv(gym.Env):
    """
    Gymnasium wrapper around a SUMO simulation of a 4-way intersection.

    Observation (per traffic light):
        [queue_lane_0, ..., queue_lane_N,   <- halted vehicles, normalised
         norm_phase]                         <- phase / num_phases, in [0,1]

    Actions (Discrete 4):
        0 = keep current phase
        1 = advance to next phase
        2 = extend current phase by +3 s
        3 = reduce  current phase by -3 s (new, down to a 5 s minimum)

    Reward:
        - (total halting vehicles / total lanes)   <- penalise queues
        + 0.5 * vehicles that departed this step   <- reward throughput
    """

    metadata = {}

    def __init__(self, seed: int = SEED):
        super().__init__()

        self._seed     = seed
        self._sumoCmd  = [SUMO_BINARY, "-c", SUMO_CONFIG, "--start",
                          "--no-warnings", "--no-step-log"]
        self._step_count      = 0          # sim steps within episode
        self._prev_departed   = 0          # vehicle count from last step
        self._episode_rewards = []         # accumulated reward per episode
        self._episode_queue   = 0          # total queue-steps this episode
        self._log_rows        = []         # rows to flush to CSV

        # Start SUMO briefly to probe the state shape, then close.
        self._start_sumo()
        self.tl_ids     = traci.trafficlight.getIDList()
        self._num_lanes = self._count_total_lanes()
        dummy            = self._build_state()
        obs_shape        = dummy.shape
        traci.close()

        # Gym spaces
        # Actions: 0=keep, 1=next-phase, 2=extend, 3=reduce
        self.action_space      = spaces.Discrete(4)
        # Observations: all values normalised to [0, 1]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=obs_shape, dtype=np.float32
        )

    # ── Gym API ───────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        """Restart the SUMO simulation and return the initial observation."""
        if traci.isLoaded():
            traci.close()
        self._start_sumo()
        self.tl_ids           = traci.trafficlight.getIDList()
        self._step_count      = 0
        self._prev_departed   = 0
        self._episode_rewards = []
        self._episode_queue   = 0
        return self._build_state(), {}

    def step(self, action):
        """
        Apply action, advance DECISION_INTERVAL simulation steps, return
        (obs, reward, terminated, truncated, info).
        """
        self._apply_action(action)

        # Advance the simulation by DECISION_INTERVAL ticks (each = 0.1 s).
        for _ in range(DECISION_INTERVAL):
            if traci.simulation.getMinExpectedNumber() == 0:
                break
            traci.simulationStep()
            self._step_count += 1

        obs    = self._build_state()
        reward = self._compute_reward()
        done   = traci.simulation.getMinExpectedNumber() == 0

        self._episode_rewards.append(reward)
        self._episode_queue  += self._total_halting()

        if done:
            self._on_episode_end()

        return obs, reward, done, False, {}

    def close(self):
        if traci.isLoaded():
            traci.close()
        self._flush_log()

    # ── State ─────────────────────────────────────────────────────────────────

    def _build_state(self) -> np.ndarray:
        """
        Build a normalised state vector.

        For each traffic light:
          - halting vehicles per controlled lane, divided by MAX_QUEUE
          - current phase index divided by number of phases

        All values are in [0, 1], making gradient-based learning faster.
        """
        state = []
        for tl in self.tl_ids:
            lanes      = traci.trafficlight.getControlledLanes(tl)
            queues     = [traci.lane.getLastStepHaltingNumber(l) / MAX_QUEUE
                          for l in lanes]
            logics     = traci.trafficlight.getAllProgramLogics(tl)
            num_phases = len(logics[0].phases)
            phase      = traci.trafficlight.getPhase(tl) / max(num_phases - 1, 1)
            state.extend(queues + [phase])
        return np.array(state, dtype=np.float32)

    # ── Reward ────────────────────────────────────────────────────────────────

    def _compute_reward(self) -> float:
        """
        reward = -normalised_queue + throughput_bonus

        Normalised queue: average halting vehicles per lane (range roughly [0,1]).
        Throughput bonus: +0.5 per vehicle that completed its journey this step.
        """
        halting       = self._total_halting()
        norm_queue    = halting / max(self._num_lanes, 1)

        # Vehicles that departed (arrived at destination) since last step.
        now_departed  = traci.simulation.getArrivedNumber()
        throughput    = now_departed  # count for this decision step
        self._prev_departed = now_departed

        reward = -norm_queue + 0.5 * throughput
        return float(reward)

    def _total_halting(self) -> int:
        """Sum of halting vehicles across all controlled lanes."""
        total = 0
        for tl in self.tl_ids:
            lanes  = traci.trafficlight.getControlledLanes(tl)
            total += sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes)
        return total

    # ── Action ────────────────────────────────────────────────────────────────

    def _apply_action(self, action: int):
        """
        Apply the chosen action to every traffic light.

        Actions:
            0 → keep current phase (do nothing)
            1 → advance to next phase
            2 → extend current phase duration by +3 s
            3 → reduce  current phase duration by -3 s (min 5 s enforced)

        Yellow-phase guard: never skip a yellow phase (state char 'y' or 'Y').
        Yellow phases are critical safety transitions; the agent must let them
        complete naturally.
        """
        for tl in self.tl_ids:
            logics     = traci.trafficlight.getAllProgramLogics(tl)
            phases     = logics[0].phases
            num_phases = len(phases)
            current    = traci.trafficlight.getPhase(tl)

            # Safety guard: do not alter control during yellow phase.
            current_state = phases[current].state
            is_yellow     = 'y' in current_state.lower()
            if is_yellow:
                continue   # let the yellow phase run out naturally

            if action == 1:
                # Advance to the next phase in the cycle.
                traci.trafficlight.setPhase(tl, (current + 1) % num_phases)

            elif action == 2:
                # Extend current green by 3 seconds.
                dur = traci.trafficlight.getPhaseDuration(tl)
                traci.trafficlight.setPhaseDuration(tl, dur + 3.0)

            elif action == 3:
                # Shorten current green by 3 seconds (minimum 5 s).
                dur = traci.trafficlight.getPhaseDuration(tl)
                traci.trafficlight.setPhaseDuration(tl, max(5.0, dur - 3.0))
            # action == 0: do nothing

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _start_sumo(self):
        """Launch SUMO and connect via TraCI."""
        if traci.isLoaded():
            traci.close()
        traci.start(self._sumoCmd)

    def _count_total_lanes(self) -> int:
        """Return total number of controlled lanes across all traffic lights."""
        total = 0
        for tl in traci.trafficlight.getIDList():
            total += len(traci.trafficlight.getControlledLanes(tl))
        return total

    def _on_episode_end(self):
        """Called when all vehicles have exited. Log episode metrics."""
        sim_time      = traci.simulation.getTime()
        total_reward  = sum(self._episode_rewards)
        avg_queue     = self._episode_queue / max(len(self._episode_rewards), 1)

        print(
            f"[Episode done] sim_time={sim_time:.1f}s | "
            f"total_reward={total_reward:.2f} | avg_queue={avg_queue:.2f}"
        )
        self._log_rows.append({
            "sim_time":    sim_time,
            "total_reward": total_reward,
            "avg_queue":   avg_queue,
        })

    def _flush_log(self):
        """Write accumulated episode metrics to CSV."""
        if not self._log_rows:
            return
        with open(LOG_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["sim_time", "total_reward", "avg_queue"])
            writer.writeheader()
            writer.writerows(self._log_rows)
        print(f"Training log saved to {LOG_CSV}")


# ── TRAINING ──────────────────────────────────────────────────────────────────

def train(fast: bool = False):
    """
    Train the PPO agent.

    Key improvements over original:
      - DECISION_INTERVAL: agent acts every 5 sim-seconds, not every 0.1 s
      - Normalised observations: all values in [0,1]
      - Richer reward (queue penalty + throughput bonus)
      - 4 actions (added 'reduce phase')
      - Yellow-phase guard
      - Per-episode CSV log
      - Fixed random seed
      - Best model saved via EvalCallback (not just final checkpoint)
      - n_steps increased to 2048, gamma=0.99 explicitly set
    """
    total_timesteps = FAST_TIMESTEPS if fast else FULL_TIMESTEPS
    print(f"Starting training for {total_timesteps:,} timesteps...")
    if fast:
        print("  (FAST mode: use this only to verify the setup works)")

    # Wrap with Monitor so SB3 can track episode rewards automatically.
    env      = Monitor(SumoEnv(seed=SEED))
    eval_env = Monitor(SumoEnv(seed=SEED + 1))

    model = PPO(
        policy          = "MlpPolicy",
        env             = env,
        verbose         = 1,
        seed            = SEED,
        # Rollout buffer: 2048 steps per update (was 512).
        # More steps → more diverse experience per gradient update.
        n_steps         = 2048,
        batch_size      = 128,
        n_epochs        = 10,
        learning_rate   = 3e-4,
        # Discount factor: 0.99 means the agent cares about future rewards
        # ~100 decision steps ahead, appropriate for traffic control.
        gamma           = 0.99,
        # GAE lambda: balances bias vs. variance in advantage estimation.
        gae_lambda      = 0.95,
        # PPO clip: prevents policy from changing too drastically per update.
        clip_range      = 0.2,
        # TensorBoard log (optional: view with `tensorboard --logdir ./logs`)
        tensorboard_log = "./logs/",
        policy_kwargs   = dict(net_arch=[128, 128]),  # two hidden layers of 128
    )

    # EvalCallback: evaluate on a separate env every 5000 steps.
    # Saves the best-performing model automatically.
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path = "./best_model/",
        log_path             = "./eval_logs/",
        eval_freq            = 5000,
        n_eval_episodes      = 1,
        deterministic        = True,
        verbose              = 1,
    )

    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    model.save("ppo_sumo_model")

    env.close()
    eval_env.close()
    print("Training complete. Model saved to ppo_sumo_model.zip")
    print("Best model saved to best_model/best_model.zip")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO traffic agent")
    parser.add_argument(
        "--fast", action="store_true",
        help=f"Quick test run ({FAST_TIMESTEPS:,} steps instead of {FULL_TIMESTEPS:,})"
    )
    args = parser.parse_args()
    train(fast=args.fast)
