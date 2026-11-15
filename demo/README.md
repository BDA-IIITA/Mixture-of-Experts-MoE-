# Adaptive MoE demo scaffold

This standalone package preserves both research notebooks. It adds a Gradio shell, request/event/record contracts, JSON/CSV export, checkpoint-specific PPO preprocessing, and a one-expert CUDA compatibility command. Full-model inference, routing visualization, budget enforcement, sequential comparisons, timed replay, and the frozen task suite are still pending. The live UI explicitly reports this status.

## Run the interface

Use Python 3.11 or 3.12 in a virtual environment. Install `demo/requirements-replay.txt`, then run `python -m demo.app` from the repository root. No CUDA packages are needed for viewing a stored record. The File control accepts a single RunRecord object or a session export containing exactly one record. No GPU recordings are bundled. Replay currently displays the final output and trace, without reproducing event timing.

For a remote host, use `ssh -L 7860:127.0.0.1:7860 user@gpu-host`. The launch scripts install requirements into the active Python environment. `launch-gpu.sh` installs provisional GPU dependencies but the UI still refuses live inference until full-model support is implemented and validated.

## Policy compatibility

The default checkpoint is `model/agent.zip`, SHA256 `399cc0497f43854fad6367a29c062038b17644f68c6178672aed81d0d3d3dcc5`, saved with Stable-Baselines3 **2.9.0**. Its metadata describes a 384/192 ReLU policy. `demo.policy.load_policy` verifies the hash and checks 68 observations and 64 three-tier actions on CPU.

`demo.policy.observation` follows `AdaptiveMoEEnv._obs` in `adapmoe-ipynb.ipynb`: standardize 64 logits, stable softmax at temperature 1, multiply importance by 64, append `[battery/10, ram_GB/8, drain_rate, thermal]`. Defaults are battery 100, RAM 4 GB, drain 0.10, thermal zero. One older evaluation cell in that notebook feeds standardized logits directly; it does not match `_obs`. The real benchmark's final comparison cell uses the correct importance features but reuses the first dataset record for every prompt. The intended demo must instead profile each entered prompt with NF4 and evaluate that feature distribution freshly.

Tiers map to FP16=0, INT8=1, NF4=2. The training quality penalties are `[0, (8.991-8.962)/(9.120-8.962), 1]`; these are training inputs, not measured demo quality.

## GPU gate

Install `demo/requirements-live.txt`. Resolve and record an immutable model commit SHA, then run:

```bash
python -m demo.expert_gate --revision <40-character-model-commit> --output expert-gate.json
```

The command downloads the checkpoint index and the shard(s) containing layer zero/expert zero. It constructs only one CPU expert, then tests its three projections at FP16, packed INT8, and packed NF4 in sequence. It checks finite forward outputs, post-forward weight dtypes, and measured allocation ordering, recording the GPU, versions, and errors against FP16. It can download a whole shard despite selecting only one expert; ensure sufficient disk space. Dependencies are provisional until validated on the target GPU. This gate does not enable the unfinished full-model backend.

## Why a separate packed backend is required

In `adaptive-moe-real-benchmark.ipynb`, `quantize_slice` rounds INT8 values then returns a tensor in the source dtype, and NF4 calls `dequantize_4bit` immediately. `apply_mixed_precision` copies those tensors into existing FP16 fused parameters. This evaluates quantization error but does not retain packed expert storage. `expert_memory_gb` estimates bits per tier using hard-coded shapes and a default layer count of 32; it does not measure live GPU allocations. Its figures therefore cannot serve as recorded demo memory evidence or prove a configuration fits on 16 GB.

The intended full backend uses Transformers 4.57.1 individual experts, retains original CPU weights, preserves model-configured routing, and keeps quantized parameters packed. Read layer count, expert count, top-k, and dimensions from model config; the README's top-2 description must not be hard-coded. Remaining validation includes FP16 agreement, routing counts, switching/generation peaks, KV/workspace reservations, budget rejection, cancellation, OOM recovery, and repeated-request stability.
