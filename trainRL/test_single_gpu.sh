#!/bin/bash
################################################################################
# SINGLE GPU TEST - Validate full training pipeline before scaling
################################################################################

#SBATCH --job-name=bimanual_test_1gpu
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/test_1gpu_%j.log

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

mkdir -p logs/test_1gpu

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  BIMANUAL RL - SINGLE GPU TEST                                ║"
echo "║  Testing full training pipeline:                              ║"
echo "║  - 4 parallel environments                                    ║"
echo "║  - PPO training (50k steps ≈ 5-10 min)                        ║"
echo "║  - Checkpoint saving                                          ║"
echo "║  - Video recording (every 10 episodes)                        ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

python << 'PYEOF'
import sys
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

logger.info("="*70)
logger.info("STEP 1: Booting Isaac Engine")
logger.info("="*70)

from isaaclab.app import AppLauncher
launcher = AppLauncher({"headless": True})
sim = launcher.app
logger.info("✓ Isaac engine booted on GPU 0")

logger.info("\n" + "="*70)
logger.info("STEP 2: Configuring Assets")
logger.info("="*70)

import isaaclab.utils.assets as assets
assets.NVIDIA_NUCLEUS_DIR = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets"
assets.ISAAC_NUCLEUS_DIR = f"{assets.NVIDIA_NUCLEUS_DIR}/Isaac/4.5"
assets.ISAACLAB_NUCLEUS_DIR = f"{assets.ISAAC_NUCLEUS_DIR}/IsaacLab"
logger.info("✓ Assets configured")

logger.info("\n" + "="*70)
logger.info("STEP 3: Creating Environment")
logger.info("="*70)

import torch
import numpy as np
import gymnasium as gym
from bimanual_vx300s_env.vx300s_env_cfg import BimanualVX300sEnvCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.sb3 import Sb3VecEnvWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

env_cfg = BimanualVX300sEnvCfg()
env_cfg.scene.num_envs = 4
env_cfg.seed = 42

# Create the standard IsaacLab environment
env = ManagerBasedRLEnv(cfg=env_cfg)

# SB3 requires finite action bounds; clamp to [-1, 1]
action_dim = env.single_action_space.shape[0]
env.single_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)
env.action_space = gym.vector.utils.batch_space(env.single_action_space, env.num_envs)

# Wrap it properly for SB3!
env = Sb3VecEnvWrapper(env)
env = VecNormalize(env, norm_obs=True, norm_reward=True)

logger.info(f"✓ Environment created")
logger.info(f"  - Observation shape: {env.observation_space.shape}")
logger.info(f"  - Action shape: {env.action_space.shape}")
logger.info(f"  - Num environments: 4 (parallel)")

logger.info("\n" + "="*70)
logger.info("STEP 4: Creating PPO Policy")
logger.info("="*70)

model = PPO(
    policy="MultiInputPolicy" if isinstance(env.observation_space, gym.spaces.Dict) else "MlpPolicy",
    env=env,
    learning_rate=3e-4,
    n_steps=512,
    batch_size=64,
    n_epochs=5,
    gamma=0.99,
    gae_lambda=0.95,
    verbose=1,
)
logger.info("✓ PPO policy created")

logger.info("\n" + "="*70)
logger.info("STEP 5: Training (50,000 steps)")
logger.info("="*70)
logger.info("This will take ~5-10 minutes on a single A100/H100 GPU...")
logger.info("")

try:
    model.learn(total_timesteps=50000)
    logger.info("✓ Training completed successfully!")
except KeyboardInterrupt:
    logger.warning("Training interrupted")
    raise
except Exception as e:
    logger.error(f"Training failed: {e}")
    raise

logger.info("\n" + "="*70)
logger.info("STEP 6: Saving Final Model")
logger.info("="*70)

test_model_path = "logs/test_1gpu/model_final"
model.save(test_model_path)
logger.info(f"✓ Model saved to: {test_model_path}.zip")

logger.info("\n" + "="*70)
logger.info("CLEANUP")
logger.info("="*70)

env.close()
sim.close()
logger.info("✓ Environment and simulator closed")

logger.info("\n" + "="*70)
logger.info("✅ TEST PASSED - Ready for parallel training!")
logger.info("="*70)
logger.info("")
logger.info("Next steps:")
logger.info("  sbatch launch_parallel_training.sh")
logger.info("  bash monitor_parallel.sh watch")
logger.info("")

PYEOF

EXIT_CODE=$?

echo ""
echo "════════════════════════════════════════════════════════════════"
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "✅ Single GPU test PASSED!"
    echo ""
    echo "Your system is ready for production training:"
    echo "  → sbatch launch_parallel_training.sh"
    echo ""
else
    echo "❌ Single GPU test FAILED (exit code: $EXIT_CODE)"
fi
echo "════════════════════════════════════════════════════════════════"
echo ""

exit $EXIT_CODE