#!/bin/bash
################################################################################
# SINGLE INSTANCE TEST - Verify parallel training works before scaling
################################################################################

#SBATCH --job-name=bimanual_test_single
#SBATCH --partition=gpu-a100,gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/test_single_%j.log

export BASE_DIR="$HOME/yash"
export TRAINRL_DIR="${BASE_DIR}/trainRL"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export ACCEPT_EULA=Y
export ISAACSIM_ACCEPT_EULA=Y
export OMNI_KIT_ACCEPT_EULA=Y
export PRIVACY_CONSENT=Y

source "${BASE_DIR}/miniforge3/bin/activate"
conda activate isaac_fresh

cd "${TRAINRL_DIR}"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  BIMANUAL RL - SINGLE INSTANCE TEST                           ║"
echo "║  Testing with: 4 environments, 10k steps                      ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

python << 'PYEOF'
import sys
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Boot Isaac
from isaaclab.app import AppLauncher
launcher = AppLauncher({"headless": True})
sim = launcher.app

# Setup assets
import isaaclab.utils.assets as assets
assets.NVIDIA_NUCLEUS_DIR = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets"
assets.ISAAC_NUCLEUS_DIR = f"{assets.NVIDIA_NUCLEUS_DIR}/Isaac/4.5"
assets.ISAACLAB_NUCLEUS_DIR = f"{assets.ISAAC_NUCLEUS_DIR}/IsaacLab"

# Import RL
import torch
import gymnasium as gym
from bimanual_vx300s_env.vx300s_env_cfg import BimanualVX300sEnvCfg
from isaaclab.envs import make_rl_env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

print("\n[INFO] Creating environment (4 parallel)...")
env_cfg = BimanualVX300sEnvCfg()
env_cfg.scene.num_envs = 4
env_cfg.seed = 42

env = make_rl_env(cfg=env_cfg, num_envs=4)
env = VecNormalize(env, norm_obs=True, norm_reward=True)

print("[INFO] Creating PPO model...")
model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,
    n_steps=512,
    batch_size=64,
    n_epochs=5,
    verbose=1,
)

print("[INFO] Training for 10,000 steps (test run)...")
print("="*60)
model.learn(total_timesteps=10000)
print("="*60)

print("\n✓ SUCCESS! Parallel training environment works!")
print("Ready to scale to 4 instances.\n")

env.close()
sim.close()
PYEOF

EXIT_CODE=$?

echo ""
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "✓ Test passed! You can now run:"
    echo "  sbatch launch_parallel_training.sh"
else
    echo "✗ Test failed with exit code $EXIT_CODE"
fi
echo ""
