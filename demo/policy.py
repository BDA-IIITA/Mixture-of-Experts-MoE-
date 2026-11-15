"""Observation adapter matching AdaptiveMoEEnv._obs in this repository."""
from pathlib import Path
import hashlib

CHECKPOINT = Path(__file__).resolve().parents[1] / 'model' / 'agent.zip'
CHECKPOINT_SHA256 = '399cc0497f43854fad6367a29c062038b17644f68c6178672aed81d0d3d3dcc5'
QUALITY_PENALTY = (0.0, (8.991 - 8.962) / (9.120 - 8.962), 1.0)


def observation(logits, battery=100, ram_gb=4.0, drain=0.10, thermal=0.0):
    import numpy as np
    x = np.asarray(logits, dtype=np.float32)
    if x.shape != (64,) or not np.isfinite(x).all():
        raise ValueError('Expected 64 finite mean routing logits')
    if not 0 <= battery <= 100:
        raise ValueError('Battery must be between 0 and 100')
    z = (x - np.mean(x)) / (np.std(x) + 1e-8)
    e = np.exp(z - np.max(z))
    importance = e / np.sum(e)
    return np.concatenate([(importance * 64).astype(np.float32),
                           np.array([battery / 10.0, ram_gb / 8.0, drain, thermal], dtype=np.float32)])


def load_policy(path=CHECKPOINT):
    import numpy as np
    from stable_baselines3 import PPO
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != CHECKPOINT_SHA256:
        raise ValueError('Checkpoint hash differs from the inspected model/agent.zip')
    policy = PPO.load(path, device='cpu')
    if policy.observation_space.shape != (68,) or not np.array_equal(policy.action_space.nvec, [3] * 64):
        raise ValueError('Expected 68 observations and MultiDiscrete([3] * 64) actions')
    return policy
