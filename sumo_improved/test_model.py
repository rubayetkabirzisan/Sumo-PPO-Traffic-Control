"""
test_model.py — Evaluate a trained PPO Traffic Signal Controller
================================================================
Improvements vs. original:
  - Consistent with training: uses DECISION_INTERVAL (every 5 sim-seconds)
  - Headless mode by default (--gui flag to open SUMO-GUI)
  - Loads best_model if available, falls back to ppo_sumo_model.zip
  - Prints live throughput and queue stats every 100 decisions
  - Final summary: sim time, average queue, total wait time
  - Prints comparison vs. fixed-time baseline (1999 s)

Usage:
    python test_model.py            # headless SUMO (no window)
    python test_model.py --gui      # open SUMO-GUI to watch visually
    python test_model.py --model path/to/my_model.zip
"""

import os
import sys
import argparse
import numpy as np

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

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
SUMO_CONFIG       = "sumoconfig.sumocfg"
DECISION_INTERVAL = 50      # must match training (50 steps = 5 sim-seconds)
MAX_QUEUE         = 20.0    # must match training
MAX_STEPS         = 100_000 # safety cap (10,000 sim-seconds)
BASELINE_TIME     = 1999    # fixed-signal baseline from README

# ── HELPERS ───────────────────────────────────────────────────────────────────

def get_state(tl_ids) -> np.ndarray:
    """Build normalised observation — identical to SumoEnv._build_state()."""
    state = []
    for tl in tl_ids:
        lanes      = traci.trafficlight.getControlledLanes(tl)
        queues     = [traci.lane.getLastStepHaltingNumber(l) / MAX_QUEUE
                      for l in lanes]
        logics     = traci.trafficlight.getAllProgramLogics(tl)
        num_phases = len(logics[0].phases)
        phase      = traci.trafficlight.getPhase(tl) / max(num_phases - 1, 1)
        state.extend(queues + [phase])
    return np.array(state, dtype=np.float32)


def apply_action(action: int, tl_ids):
    """Apply action — identical to SumoEnv._apply_action()."""
    for tl in tl_ids:
        logics     = traci.trafficlight.getAllProgramLogics(tl)
        phases     = logics[0].phases
        num_phases = len(phases)
        current    = traci.trafficlight.getPhase(tl)

        # Yellow-phase guard
        if 'y' in phases[current].state.lower():
            continue

        if action == 1:
            traci.trafficlight.setPhase(tl, (current + 1) % num_phases)
        elif action == 2:
            dur = traci.trafficlight.getPhaseDuration(tl)
            traci.trafficlight.setPhaseDuration(tl, dur + 3.0)
        elif action == 3:
            dur = traci.trafficlight.getPhaseDuration(tl)
            traci.trafficlight.setPhaseDuration(tl, max(5.0, dur - 3.0))
        # action == 0: do nothing


def total_halting(tl_ids) -> int:
    total = 0
    for tl in tl_ids:
        lanes  = traci.trafficlight.getControlledLanes(tl)
        total += sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes)
    return total


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test trained PPO traffic agent")
    parser.add_argument("--gui",   action="store_true", help="Open SUMO-GUI window")
    parser.add_argument("--model", default=None,        help="Path to model .zip file")
    args = parser.parse_args()

    # ── Load model ───────────────────────────────────────────────────────────
    if args.model:
        model_path = args.model
    elif os.path.exists("best_model/best_model.zip"):
        model_path = "best_model/best_model.zip"
        print("Loading best_model/best_model.zip (from EvalCallback)")
    elif os.path.exists("ppo_sumo_model.zip"):
        model_path = "ppo_sumo_model.zip"
        print("Loading ppo_sumo_model.zip")
    else:
        sys.exit("No model found. Run traffic_agent.py first, or pass --model <path>.")

    print(f"Model: {model_path}")
    model = PPO.load(model_path)

    # ── Start SUMO ───────────────────────────────────────────────────────────
    binary   = "sumo-gui" if args.gui else "sumo"
    sumoCmd  = [binary, "-c", SUMO_CONFIG, "--start", "--no-warnings", "--no-step-log"]
    traci.start(sumoCmd)
    tl_ids   = traci.trafficlight.getIDList()
    num_lanes = sum(
        len(traci.trafficlight.getControlledLanes(tl)) for tl in tl_ids
    )

    # ── Simulation loop ───────────────────────────────────────────────────────
    state         = get_state(tl_ids)
    sim_step      = 0
    decision_step = 0
    total_queue   = 0.0
    total_arrived = 0

    action_counts = {0: 0, 1: 0, 2: 0, 3: 0}   # track what the agent chose

    print("\n── Running simulation ──────────────────────────────────")
    print(f"  Decision every {DECISION_INTERVAL} sim-steps ({DECISION_INTERVAL * 0.1:.1f} s)")
    print(f"  Traffic lights: {list(tl_ids)}")
    print()

    while traci.simulation.getMinExpectedNumber() > 0 and sim_step < MAX_STEPS:

        # RL decision every DECISION_INTERVAL sim-steps
        if sim_step % DECISION_INTERVAL == 0:
            action, _ = model.predict(state, deterministic=True)
            action    = int(action)
            apply_action(action, tl_ids)
            action_counts[action] = action_counts.get(action, 0) + 1
            decision_step += 1

        traci.simulationStep()
        sim_step     += 1
        total_queue  += total_halting(tl_ids)
        total_arrived = traci.simulation.getArrivedNumber()

        # Progress update every 100 decisions
        if decision_step > 0 and decision_step % 100 == 0 and sim_step % DECISION_INTERVAL == 0:
            sim_time   = traci.simulation.getTime()
            avg_q      = total_queue / sim_step
            remaining  = traci.simulation.getMinExpectedNumber()
            print(
                f"  t={sim_time:6.1f}s | decisions={decision_step:4d} | "
                f"avg_queue={avg_q:.2f} | remaining={remaining}"
            )

    sim_time  = traci.simulation.getTime()
    avg_queue = total_queue / max(sim_step, 1)
    traci.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n── RESULTS ──────────────────────────────────────────────")
    print(f"  Simulation time:      {sim_time:.1f} s")
    print(f"  Vehicles cleared:     {total_arrived}")
    print(f"  Avg queue (halting):  {avg_queue:.2f} vehicles/lane-step")
    print(f"  Total decisions made: {decision_step}")
    print(f"  Action distribution:  keep={action_counts[0]}  "
          f"next={action_counts[1]}  extend={action_counts[2]}  "
          f"reduce={action_counts[3]}")

    if sim_time > 0:
        improvement = (BASELINE_TIME - sim_time) / BASELINE_TIME * 100
        sign        = "+" if improvement >= 0 else ""
        print(f"\n  vs. fixed-time baseline ({BASELINE_TIME} s): "
              f"{sign}{improvement:.1f}%  ({sim_time:.0f} s)")
    print("─────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
