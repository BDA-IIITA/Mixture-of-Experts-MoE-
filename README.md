# Adaptive MoE Quantization

An RL-based framework that dynamically assigns per-expert quantization tiers (FP16 / FP8 / FP4) in Mixture-of-Experts models, adapting in real time to device hardware state (battery, RAM, thermal load). Built using [OLMoE](https://huggingface.co/OLMoE) routing data and trained with PPO via [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3).

## Overview

Large Mixture-of-Experts (MoE) models like OLMoE activate only a subset of experts per token, but still require **all** experts to reside in memory. On resource-constrained devices, static quantization (applying the same precision to every expert) leaves performance on the table: high-importance experts get degraded unnecessarily, while low-importance ones waste resources at full precision.

This project frames **per-expert quantization assignment** as a reinforcement learning problem. A PPO agent observes per-expert routing importance and hardware telemetry at each inference step, then outputs a 64-dimensional action — one quantization tier per expert — to jointly optimize:
- **Memory savings** — reduce the total memory footprint
- **Energy savings** — extend battery life on mobile / edge devices
- **Quality preservation** — protect important experts from aggressive quantization

The agent learns to **compress more aggressively as battery drains**, smoothly trading off quality for survival, while always routing heavier quantization toward less-important experts.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Observation Space (68-dim)                │
│  ┌──────────────────────┐  ┌──────────────────────────────┐ │
│  │  Expert Importance   │  │   Hardware State (4-dim)     │ │
│  │  (64-dim, softmax-   │  │   • battery / 10            │ │
│  │   normalized, scaled │  │   • ram_free / 8             │ │
│  │   by N_EXPERTS)      │  │   • drain_rate               │ │
│  │                      │  │   • thermal flag              │ │
│  └──────────────────────┘  └──────────────────────────────┘ │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
               ┌────────────────────────┐
               │   PPO Agent (MLP)      │
               │   384 → 192 (ReLU)     │
               └────────────┬───────────┘
                            │
                            ▼
               ┌────────────────────────┐
               │  Action: MultiDiscrete │
               │  [3]^64 — one tier per │
               │  expert (0=FP16,       │
               │  1=FP8, 2=FP4)         │
               └────────────────────────┘
```

## Key Components

### Custom Gymnasium Environment (`AdaptiveMoEEnv`)

A full simulation environment that models a multi-turn "conversation session" on a device:

| Component | Description |
|---|---|
| **Expert Importance** | Derived from OLMoE routing logits via temperature-scaled softmax (`τ = 1.0`) |
| **Hardware Simulator** | Simulates battery drain (uniform 1–100% start), fluctuating RAM (2–8 GB), and thermal state. Battery depletes each step as a function of the agent's energy consumption. |
| **Quantization Tiers** | 3 tiers with distinct memory cost (0.012 / 0.006 / 0.003 GB per expert), energy cost (1.0 / 0.6 / 0.3 relative units), and quality penalty (0.0 / 0.18 / 1.0, derived from actual OLMoE perplexity deltas) |
| **Reward Function** | Weighted sum of memory savings, energy savings (weight increases as battery drops), dynamic quality penalty (quality protection fades as battery drops, shaped by a `quality_curve_k` parameter), survival bonus, and terminal death penalty |
| **Termination** | Episode ends when battery < 2% or free RAM < 0.3 GB (terminal failure) or after `max_turns = 40` steps |

### Routing Dataset

- **2,264 prompts** across diverse task types (chat, math, code, reasoning, etc.)
- Extracted from **OLMoE-1B-7B** (64 experts, top-2 routing)
- Each sample contains per-layer routing logits (16 layers × 64 experts), the mean routing logits across layers, and top-2 expert selections
- Stored at `data/routing_dataset_final.json` (~52 MB)

### PPO Training Configuration

| Hyperparameter | Value |
|---|---|
| Policy | MlpPolicy (384 → 192, ReLU) |
| Learning rate | 1.5e-4 |
| Steps per rollout | 2,048 |
| Batch size | 1,024 |
| Epochs per update | 5 |
| Gamma (discount) | 0.95 |
| GAE Lambda | 0.95 |
| Clip range | 0.2 |
| Entropy coeff | 0.0 |
| Target KL | 0.02 |
| Total timesteps | 200,000 |
| Parallel envs | 4 (VecNormalize with reward normalization) |
| Eval frequency | Every 2,500 steps, 100 episodes |

## Results

### Adaptive Behavior Across Battery Levels

The trained agent demonstrates smooth, monotonic adaptation — compressing more as battery drains:

| Battery | Perplexity | Mem Saved | Energy Saved | FP16% | FP8% | FP4% |
|---------|------------|-----------|--------------|-------|------|------|
| 100%    | 8.9801     | 35.2%     | 29.2%        | 34.9% | 54.4% | 10.7% |
| 75%     | 8.9829     | 37.5%     | 31.5%        | 32.1% | 53.6% | 14.3% |
| 50%     | 8.9880     | 41.0%     | 34.9%        | 28.4% | 50.5% | 21.0% |
| 25%     | 8.9973     | 46.2%     | 40.1%        | 23.3% | 45.2% | 31.5% |
| 10%     | 9.0041     | 49.1%     | 43.0%        | 20.6% | 41.9% | 37.5% |

### Importance-Aware Routing

The agent achieves **59.4% captured alignment** at 50% battery (0% = random, 100% = oracle-optimal), confirming it systematically routes heavier quantization toward less-important experts. Spearman rank correlation between expert importance and assigned precision: **~0.28–0.31** across all battery levels.

### Pareto Frontier

The RL agent traces a curve between the static FP16 and FP4 baselines, achieving the Pareto-optimal quality–efficiency tradeoff inaccessible to any single static quantization scheme.

<p align="center">
  <img src="results/fig2_pareto_frontier (1).png" alt="Pareto Frontier" width="600"/>
</p>

### Agent Behavior Dashboard

<p align="center">
  <img src="results/fig3_stacked_behavior.png" alt="Stacked Behavior" width="600"/>
</p>

## Repository Structure

```
Adaptive-MoE-Quantization/
├── adapmoe-ipynb.ipynb              # Main notebook (data collection, environment,
│                                    #   training, evaluation, and visualization)
├── data/
│   └── routing_dataset_final.json   # 2,264 prompts with OLMoE routing logits
├── model/
│   ├── agent.zip                    # Final trained PPO agent
│   ├── best_model.zip               # Best checkpoint (by eval reward)
│   └── routing_predictor_best (1).pt # Routing predictor model checkpoint
├── results/
│   ├── fig2_pareto_frontier (1).png # Pareto frontier plot
│   └── fig3_stacked_behavior.png    # Multi-panel behavior dashboard
├── .gitignore
└── README.md
```

## Requirements

- Python 3.12+
- PyTorch ≥ 2.8
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) ≥ 2.9
- Gymnasium
- NumPy, SciPy, Pandas, Matplotlib

Install the key dependency:
```bash
pip install stable-baselines3
```

## Usage

### Training

The entire pipeline runs in a single Kaggle notebook (`adapmoe-ipynb.ipynb`) with a T4 GPU:

1. Load the routing dataset
2. Instantiate the `AdaptiveMoEEnv` with the `HardwareSimulator`
3. Train a PPO agent for 200K timesteps (~50–70 minutes)
4. Evaluate across battery levels and generate plots

### Inference

```python
from stable_baselines3 import PPO
import numpy as np

agent = PPO.load("model/best_model")

# Build observation from routing logits + hardware state
routing_logits = np.array(sample["mean_routing_logits"], dtype=np.float32)
z = (routing_logits - np.mean(routing_logits)) / (np.std(routing_logits) + 1e-8)
e = np.exp((z - np.max(z)) / 1.0)
importance = e / np.sum(e)
routing_features = (importance * 64).astype(np.float32)

hw_vec = np.array([battery / 10.0, ram_free / 8.0, drain_rate, thermal], dtype=np.float32)
obs = np.concatenate([routing_features, hw_vec])

# Get per-expert quantization assignments
action, _ = agent.predict(obs, deterministic=True)
# action[i] ∈ {0: FP16, 1: FP8, 2: FP4} for each expert i
```

## Citation

If you use this work, please cite:

```
@misc{adaptive-moe-quantization,
  title={Adaptive Per-Expert Quantization for Mixture-of-Experts Models via Reinforcement Learning},
  year={2026},
  url={https://github.com/your-username/Adaptive-MoE-Quantization}
}
```

## License

This project is for academic research purposes.
