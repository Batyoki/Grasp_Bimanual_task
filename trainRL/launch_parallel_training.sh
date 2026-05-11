#!/bin/bash
#SBATCH --job-name=bimanual_rl_train
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j.err

set -e

################################################################################
# PARSE ARGUMENTS IMMEDIATELY - BEFORE ANY CONDA COMMANDS
################################################################################
TOTAL_TIMESTEPS="${1:-1000000}"
NUM_ENVS="${2:-4}"

# Strip any trailing whitespace
TOTAL_TIMESTEPS="$(echo "$TOTAL_TIMESTEPS" | xargs)"
NUM_ENVS="$(echo "$NUM_ENVS" | xargs)"

################################################################################
# PART 1: HPC SURVIVAL SHIELDS & ENVIRONMENT SETUP
################################################################################

export BASE_DIR="${HOME}/yash"
export TRAINRL_DIR="${BASE_DIR}/trainRL"

# Capture the SLURM Job ID (Fallback to "local" if running bash directly)
JOB_ID=${SLURM_JOB_ID:-local_$(date +%s)}

# Create a UNIQUE directory for this specific run!
export RUN_DIR="${TRAINRL_DIR}/logs/run_${JOB_ID}"
mkdir -p "${RUN_DIR}"

export PYTHONUNBUFFERED=1
export ACCEPT_EULA=Y
export ISAACSIM_ACCEPT_EULA=Y
export OMNI_KIT_ACCEPT_EULA=Y
export PRIVACY_CONSENT=Y
export PYTHONNOUSERSITE=1

# Use direct path to conda/python instead of source/activate to avoid issues
export PATH="${BASE_DIR}/miniforge3/envs/isaac_fresh/bin:${BASE_DIR}/miniforge3/bin:${PATH}"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  BIMANUAL RL - SINGLE GPU TRAINING                             ║"
echo "║  Job ID: ${JOB_ID}                                             ║"
echo "║  GPU: 0 (H100)                                                 ║"
echo "║  Environments: ${NUM_ENVS}                                            ║"
echo "║  Timesteps: ${TOTAL_TIMESTEPS}                                   ║"
echo "║  Output Dir: ${RUN_DIR}                                        ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

################################################################################
# PART 2: GENERATE & EXECUTE TRAINING SCRIPT
################################################################################

# Save the python script directly into the unique run folder
cat > "${RUN_DIR}/train.py" << 'PYEOF'
#!/usr/bin/env python3
"""
Single-GPU training script for Bimanual RL.
Trains on 1 GPU with independent checkpoint management.
"""

import sys
import os
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[Train] %(asctime)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get parameters from environment
GPU_ID = int(os.environ.get('GPU_ID', 0))
NUM_ENVS = int(os.environ.get('NUM_ENVS', 4))
TOTAL_TIMESTEPS = int(os.environ.get('TOTAL_TIMESTEPS', 1000000))
TRAINRL_DIR = os.environ.get('TRAINRL_DIR', './trainRL')
# We read RUN_DIR from bash, which has our unique SLURM ID
RUN_DIR = os.environ.get('RUN_DIR', './logs')

# Add trainRL to path
sys.path.insert(0, TRAINRL_DIR)

logger.info("="*70)
logger.info("BIMANUAL RL - SINGLE GPU TRAINING")
logger.info("="*70)

try:
    # ─────────────────────────────────────────────────────────────
    # A. BOOT ISAAC ENGINE
    # ─────────────────────────────────────────────────────────────
    import torch
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)
    torch.cuda.set_device(GPU_ID)
    
    from isaaclab.app import AppLauncher
    # enable_cameras=True is required for RecordVideo to work!
    launcher = AppLauncher({"headless": True, "enable_cameras": True})
    sim = launcher.app

    # ─────────────────────────────────────────────────────────────
    # B. ASSET CONFIGURATION
    # ─────────────────────────────────────────────────────────────
    import isaaclab.utils.assets as assets
    assets.NVIDIA_NUCLEUS_DIR = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets"
    assets.ISAAC_NUCLEUS_DIR = f"{assets.NVIDIA_NUCLEUS_DIR}/Isaac/4.5"
    assets.ISAACLAB_NUCLEUS_DIR = f"{assets.ISAAC_NUCLEUS_DIR}/IsaacLab"

    # ─────────────────────────────────────────────────────────────
    # C. CREATE ENVIRONMENT & VIDEO SETUP
    # ─────────────────────────────────────────────────────────────
    import numpy as np
    import gymnasium as gym
    from gymnasium.wrappers import RecordVideo
    from bimanual_vx300s_env.vx300s_env_cfg import BimanualVX300sEnvCfg
    from isaaclab.envs import ManagerBasedRLEnv
    from checkpoint_manager import CheckpointManager
    from stable_baselines3 import PPO
    from isaaclab_rl.sb3 import Sb3VecEnvWrapper
    
    # 1. Setup Directories using the unique RUN_DIR
    checkpoint_dir = Path(RUN_DIR) / "checkpoints"
    video_dir = Path(RUN_DIR) / "videos"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Setup Environment
    env_cfg = BimanualVX300sEnvCfg()
    env_cfg.scene.num_envs = NUM_ENVS
    env_cfg.seed = 42
    
    # render_mode is required for RecordVideo!
    env = ManagerBasedRLEnv(cfg=env_cfg, render_mode="rgb_array")
    
    # 3. Action Bounds for SB3
    action_dim = env.single_action_space.shape[0]
    env.single_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)
    env.action_space = gym.vector.utils.batch_space(env.single_action_space, env.num_envs)

    # 4. ADD VIDEO RECORDER (Record every 500 steps)
    env = RecordVideo(
        env, 
        video_folder=str(video_dir), 
        step_trigger=lambda step: step % 500 == 0,
        disable_logger=True
    )

    # 5. Wrap for Stable Baselines 3
    env = Sb3VecEnvWrapper(env)
    
    checkpoint_manager = CheckpointManager(checkpoint_dir=str(checkpoint_dir))

    # ─────────────────────────────────────────────────────────────
    # E. CREATE PPO POLICY
    # ─────────────────────────────────────────────────────────────
    model = PPO(
        policy="MultiInputPolicy" if isinstance(env.observation_space, gym.spaces.Dict) else "MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
    )

    # ─────────────────────────────────────────────────────────────
    # F. TRAINING LOOP
    # ─────────────────────────────────────────────────────────────
    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    # ─────────────────────────────────────────────────────────────
    # G. SAVE FINAL CHECKPOINT
    # ─────────────────────────────────────────────────────────────
    final_metrics = {'total_timesteps': TOTAL_TIMESTEPS, 'num_envs': NUM_ENVS}
    checkpoint_manager.save_checkpoint(
        step=TOTAL_TIMESTEPS,
        model=model,
        optimizer=model.policy.optimizer if hasattr(model.policy, 'optimizer') else None,
        metrics=final_metrics,
        config={'num_envs': NUM_ENVS},
    )

    env.close()
    sim.close()
    
except Exception as e:
    logger.error(f"\n❌ Training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

PYEOF

chmod +x "${RUN_DIR}/train.py"

export GPU_ID=0
export NUM_ENVS=${NUM_ENVS}
export TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS}
export TRAINRL_DIR=${TRAINRL_DIR}
export RUN_DIR=${RUN_DIR}

# Run training, routing logs into our unique folder
python "${RUN_DIR}/train.py" 2>&1 | tee "${RUN_DIR}/training.log"

TRAIN_EXIT_CODE=$?

echo ""
echo "════════════════════════════════════════════════════════════════"
if [[ $TRAIN_EXIT_CODE -eq 0 ]]; then
    echo "✅ Training completed successfully!"
    echo "Results saved to: ${RUN_DIR}/"
else
    echo "❌ Training failed (exit code: $TRAIN_EXIT_CODE)"
fi
echo "════════════════════════════════════════════════════════════════"
exit ${TRAIN_EXIT_CODE}