# SUMO-PPO Traffic Control — Improved Version

Reinforcement learning traffic signal controller using **PPO** (Proximal Policy
Optimization) and the **SUMO** traffic simulator.

---

## What Changed from the Original

| Area | Original | Improved |
|------|----------|----------|
| Routes | 4 U-turn routes + 13 duplicates | 12 unique, valid through-routes |
| Decision interval | Every 0.1 s (every tick) | Every 5 s (50 ticks) — matches reality |
| State normalisation | Raw counts mixed with phase index | All values normalised to [0,1] |
| Actions | 3 (keep / next / extend) | 4 (+ reduce phase) |
| Yellow-phase guard | None | Cannot alter yellow phases |
| Reward | –queue only | –queue + throughput bonus |
| Training steps | 20,000 | 200,000 (use `--fast` for quick test) |
| Best model saving | Only final model | EvalCallback saves best during training |
| Metrics / logging | None | Per-episode CSV + live console output |
| Reproducibility | No seed | Fixed seed=42 |

---

## Quick Start

### 1. Install SUMO

```bash
# Ubuntu/Debian
sudo add-apt-repository ppa:sumo/stable
sudo apt-get update && sudo apt-get install sumo sumo-tools

# macOS
brew tap dlr-ts/sumo && brew install sumo
```

Set environment variable (add to `~/.bashrc`):

```bash
export SUMO_HOME=/usr/share/sumo
export PATH=$PATH:$SUMO_HOME/bin
```

Verify: `sumo --version`

### 2. Install Python packages

```bash
pip install -r requirements.txt
```

> **Note:** `traci` and `sumolib` are bundled with SUMO — not pip packages.
> They are loaded automatically via `SUMO_HOME`.

### 3. Train

```bash
# Full training (200,000 steps, ~30-60 min depending on CPU)
python traffic_agent.py

# Quick test to verify everything works (10,000 steps, ~2 min)
python traffic_agent.py --fast
```

Training outputs:
- `ppo_sumo_model.zip` — final model
- `best_model/best_model.zip` — best model during training (use this for testing)
- `training_log.csv` — per-episode metrics
- `logs/` — TensorBoard logs (view with `tensorboard --logdir ./logs`)

### 4. Test

```bash
# Headless (no window, prints stats)
python test_model.py

# With SUMO-GUI (visual, requires display)
python test_model.py --gui

# Use a specific model file
python test_model.py --model best_model/best_model.zip
```

---

## Project Files

```
├── traffic_agent.py     # PPO training script (main entry point)
├── test_model.py        # Evaluation script
├── requirements.txt     # Python dependencies
├── routes.rou.xml       # Fixed: 12 unique valid routes, 500 vehicles
├── nodes.nod.xml        # 5 nodes (4 endpoints + 1 intersection)
├── edges.edg.xml        # 8 bidirectional road edges
├── type.type.xml        # Road type: 2-lane, 50 km/h
├── net.net.xml          # Generated SUMO network (do not edit)
├── sumoconfig.sumocfg   # SUMO simulation config
└── sumonet.netccfg      # Netconvert config (to regenerate net.net.xml)
```

---

## RL Design

### State (observation)
For each traffic light:
- Normalised halting vehicles per controlled lane: `count / 20.0`  
- Normalised current phase index: `phase / (num_phases - 1)`

### Actions
| Action | Effect |
|--------|--------|
| 0 | Keep current phase |
| 1 | Advance to next phase |
| 2 | Extend current phase by +3 s |
| 3 | Reduce current phase by –3 s (min 5 s) |

Yellow phases are protected — the agent cannot change signal state during a
yellow transition.

### Reward
```
reward = −(total_halting / num_lanes)  +  0.5 × vehicles_departed_this_step
```

---

## Regenerate Network (only if you change nodes/edges)

```bash
netconvert --node-files=nodes.nod.xml \
           --edge-files=edges.edg.xml \
           --type-files=type.type.xml \
           --output-file=net.net.xml
```

---
