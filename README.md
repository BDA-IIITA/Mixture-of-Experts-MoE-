# Adaptive MoE Quantization via RL

> **Resource-aware per-expert quantization for Mixture-of-Experts LLMs, guided by prompt-conditioned routing signals and on-device hardware constraints.**

---

## Overview

Standard quantization (fp16/fp8/fp4) applies the same precision to every expert in an MoE model. This wastes memory when most experts are irrelevant for a given prompt, and degrades quality when important experts are over-compressed.

This project trains an RL agent that:
1. **Predicts** which experts matter for the current prompt (routing predictor)
2. **Observes** real device constraints (battery, RAM, thermal)
3. **Decides** per-expert precision tiers (fp16 / fp8 / fp4 / drop) dynamically

Compared to DynaExq (traffic heuristics) and static quantization, this approach is:
- **Prompt-conditioned** — not averaged over history
- **Hardware-aware** — adapts to battery and RAM in real time
- **Learned** — not hand-tuned thresholds

---

## Key Results

| Policy | Perplexity | Memory Saved |
|---|---|---|
| FP16 (baseline) | lowest | 0% |
| FP8 (static) | +small delta | 50% |
| FP4 (static) | +large delta | 75% |
| **RL Agent (ours)** | **between FP8–FP4** | **adaptive (50–72%)** |

Agent becomes more aggressive as battery drops — saves more memory with acceptable quality loss.

---

## Project Structure

```
adaptive-moe-quant/
│
├── configs/
│   └── config.yaml              # all hyperparameters in one place
│
├── src/
│   ├── collector/
│   │   ├── load_model.py        # load OLMoE with hooks
│   │   ├── hook_manager.py      # register/remove routing hooks
│   │   ├── dataset_builder.py   # load HF datasets, collect prompts
│   │   └── collect_logits.py    # main collection loop
│   │
│   ├── predictor/
│   │   ├── dataset.py           # PyTorch dataset for predictor training
│   │   ├── model.py             # routing predictor MLP
│   │   └── train.py             # training loop + evaluation
│   │
│   ├── environment/
│   │   ├── hardware_sim.py      # battery/RAM/thermal simulator
│   │   ├── constants.py         # tier definitions, memory/energy costs
│   │   └── moe_env.py           # Gym environment
│   │
│   ├── agent/
│   │   └── train_agent.py       # PPO training + callbacks
│   │
│   └── evaluation/
│       ├── evaluate_agent.py    # per-battery evaluation
│       ├── baselines.py         # fp16/fp8/fp4 perplexity measurement
│       └── visualize.py         # all plots
│
├── scripts/
│   ├── run_collection.py        # end-to-end data collection
│   ├── run_predictor.py         # train predictor
│   ├── run_agent.py             # train RL agent
│   └── run_evaluation.py        # full evaluation pipeline
│
├── notebooks/
│   └── full_pipeline.ipynb      # Kaggle-ready notebook (all steps)
│
├── data/                        # gitignored — populated after collection
│   └── .gitkeep
│
├── models/                      # gitignored — populated after training
│   └── .gitkeep
│
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Install

```bash
git clone https://github.com/yourusername/adaptive-moe-quant
cd adaptive-moe-quant
pip install -r requirements.txt
```

### 2. Collect routing logits (run on Kaggle / Colab)

```bash
python scripts/run_collection.py --output_dir data/ --n_prompts 2200
```

### 3. Train routing predictor

```bash
python scripts/run_predictor.py --data_path data/routing_dataset_final.json
```

### 4. Train RL agent

```bash
python scripts/run_agent.py --data_path data/routing_dataset_final.json
```

### 5. Evaluate

```bash
python scripts/run_evaluation.py --agent_path models/best_model
```

---

## How It Works

### Stage 1 — Routing Logit Collection
Forward hooks on `mlp.gate` (OlmoeTopKRouter) capture raw logits over all 64 experts before top-k selection. Mean-pooled over sequence length → one `(64,)` vector per prompt per layer → `(16, 64)` matrix per prompt.

### Stage 2 — Routing Predictor MLP
Tiny MLP trained to predict mean routing logits from prompt embeddings. Enables importance estimation without a full forward pass.

### Stage 3 — RL Environment
Custom Gym env. State = routing logits + hardware. Action = tier per expert. Reward = memory saved + energy saved − quality loss (importance-weighted).

### Stage 4 — PPO Agent
Small 2-layer MLP policy trained with PPO. Battery-conditioned quality ceiling in reward ensures graceful degradation.

### Stage 5 — Real Perplexity Comparison
Agent's effective perplexity computed as importance-weighted average of real fp16/fp8/fp4 perplexity measurements.

---

## Citation

```bibtex
@misc{adaptive-moe-quant-2025,
  title   = {Resource-Adaptive MoE Inference via Prompt-Conditioned Expert Quantization},
  year    = {2025},
  note    = {GitHub repository}
}
```