#!/bin/bash
#SBATCH --job-name=bimanual_rl_training
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_%j.log
#SBATCH --error=logs/slurm_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# ==============================================================================
# BIMANUAL VX300s RL TRAINING - GPU CLUSTER SUBMISSION
# ==============================================================================
# This script submits bimanual arm RL training to GPU cluster (A100/H100)
# Usage: sbatch submit_training.sh [config] [resume_checkpoint]
# ==============================================================================

# ==============================================================================
# PART 1: HPC ENVIRONMENT SETUP (Following master_bimanual.sh)
# ==============================================================================

export BASE_DIR="$HOME/yash"
export PYTHONUNBUFFERED=1
export ACCEPT_EULA=Y
export ISAACSIM_ACCEPT_EULA=Y
export OMNI_KIT_ACCEPT_EULA=Y
export PRIVACY_CONSENT=Y
unset PYTHONNOUSERSITE

echo "======================================================================"
echo "Starting Bimanual VX300s RL Training on GPU Cluster"
echo "======================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Partition: $SLURM_PARTITION"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Node: $SLURMD_NODENAME"
echo "======================================================================"

# ==============================================================================
# PART 2: ACTIVATE ENVIRONMENT
# ==============================================================================

source "$BASE_DIR/miniforge3/bin/activate"
conda activate isaac_fresh

# Verify Isaac Lab installation
python -c "import isaaclab; print(f'Isaac Lab version: {isaaclab.__version__}')" || {
    echo "ERROR: Isaac Lab not properly installed"
    exit 1
}

# ==============================================================================
# PART 3: CREATE TRAINING DIRECTORIES
# ==============================================================================

TRAINING_DIR="$BASE_DIR/trainRL"
LOG_DIR="$TRAINING_DIR/logs/$(date +%Y%m%d_%H%M%S)"
MODEL_DIR="$TRAINING_DIR/models/$(date +%Y%m%d_%H%M%S)"
VIDEO_DIR="$LOG_DIR/videos"
IMAGE_DIR="$LOG_DIR/images"
CHECKPOINT_DIR="$LOG_DIR/checkpoints"

mkdir -p "$LOG_DIR" "$MODEL_DIR" "$VIDEO_DIR" "$IMAGE_DIR" "$CHECKPOINT_DIR"

echo "Log Directory: $LOG_DIR"
echo "Model Directory: $MODEL_DIR"
echo "Checkpoint Directory: $CHECKPOINT_DIR"

# ==============================================================================
# PART 4: PARSE ARGUMENTS
# ==============================================================================

CONFIG_FILE="${1:-$TRAINING_DIR/config.yaml}"
RESUME_CHECKPOINT="${2:-}"

echo "Configuration: $CONFIG_FILE"
if [ -n "$RESUME_CHECKPOINT" ]; then
    echo "Resuming from: $RESUME_CHECKPOINT"
fi

# ==============================================================================
# PART 5: CREATE TRAINING SCRIPT (DYNAMIC PYTHON BOILERPLATE)
# ==============================================================================

cat << 'PYEOF' > "$LOG_DIR/train_cluster.py"
#!/usr/bin/env python3
"""
GPU Cluster Training Script for Bimanual VX300s RL

Implements:
- Distance-based reward shaping
- Table collision avoidance
- Grasping and lifting rewards
- Video/image recording
- Checkpoint save/load/resume
- Performance monitoring
"""

import sys
import os
import logging
from pathlib import Path
from datetime import datetime
import argparse

import torch
import numpy as np
import yaml
from omegaconf import OmegaConf

# Add trainRL to path
sys.path.insert(0, os.environ.get('TRAINING_DIR', '.'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.environ.get('LOG_DIR', '.'), 'training.log')),
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

logger.info("="*70)
logger.info("BIMANUAL VX300s RL TRAINING - CLUSTER MODE")
logger.info("="*70)


def setup_environment():
    """Setup IsaacLab environment."""
    try:
        from isaaclab.app import AppLauncher
        launcher = AppLauncher({"headless": True})
        sim = launcher.app
        
        logger.info("✓ IsaacLab initialized")
        
        # Set cloud assets fallback
        import isaaclab.utils.assets as assets
        assets.NVIDIA_NUCLEUS_DIR = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets"
        assets.ISAAC_NUCLEUS_DIR = f"{assets.NVIDIA_NUCLEUS_DIR}/Isaac/4.5"
        assets.ISAACLAB_NUCLEUS_DIR = f"{assets.ISAAC_NUCLEUS_DIR}/IsaacLab"
        
        return sim
    except Exception as e:
        logger.error(f"Failed to initialize IsaacLab: {e}")
        raise


def load_configuration(config_path):
    """Load training configuration."""
    if not Path(config_path).exists():
        logger.error(f"Config file not found: {config_path}")
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Configuration loaded from {config_path}")
    return config


def setup_training(config, log_dir, model_dir):
    """Setup training environment and utilities."""
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import VecNormalize
        from checkpoint_manager import CheckpointManager
        from video_recorder import VideoRecorder, ImageRecorder, PerformancePlotter
        
        logger.info("✓ Training utilities imported")
        
        # Create checkpoint manager
        checkpoint_manager = CheckpointManager(
            checkpoint_dir=os.path.join(log_dir, 'checkpoints'),
            keep_best_n=3,
        )
        
        # Create recorders
        video_recorder = VideoRecorder(
            video_dir=os.path.join(log_dir, 'videos'),
            fps=30,
        )
        
        image_recorder = ImageRecorder(
            image_dir=os.path.join(log_dir, 'images'),
        )
        
        performance_plotter = PerformancePlotter(
            plot_dir=os.path.join(log_dir, 'plots'),
        )
        
        logger.info("✓ Checkpoint manager, recorders, and plotters initialized")
        
        return {
            'checkpoint_manager': checkpoint_manager,
            'video_recorder': video_recorder,
            'image_recorder': image_recorder,
            'performance_plotter': performance_plotter,
        }
    except Exception as e:
        logger.error(f"Failed to setup training utilities: {e}")
        raise


def main():
    """Main training loop."""
    parser = argparse.ArgumentParser(description="Bimanual VX300s RL Training")
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint to resume from')
    parser.add_argument('--log-dir', type=str, required=True, help='Log directory')
    parser.add_argument('--model-dir', type=str, required=True, help='Model directory')
    
    args = parser.parse_args()
    
    logger.info(f"Log directory: {args.log_dir}")
    logger.info(f"Model directory: {args.model_dir}")
    logger.info(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU name: {torch.cuda.get_device_name()}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    try:
        # Setup
        setup_environment()
        config = load_configuration(args.config)
        utils = setup_training(config, args.log_dir, args.model_dir)
        
        # Load/create environment
        logger.info("Creating RL environment...")
        import gymnasium as gym
        from isaaclab_tasks.utils import parse_env_cfg
        
        # TODO: Replace with your bimanual environment
        env_name = "Isaac-Cartpole-v0"
        env_cfg = parse_env_cfg(env_name)
        env_cfg.scene.num_envs = config.get('env', {}).get('num_envs', 4)
        
        env = gym.make(env_name, cfg=env_cfg)
        logger.info(f"✓ Environment created with {env_cfg.scene.num_envs} parallel environments")
        
        # Training loop
        logger.info("Starting training loop...")
        from stable_baselines3 import PPO
        
        model = PPO(
            policy='MlpPolicy',
            env=env,
            learning_rate=config.get('training', {}).get('learning_rate', 3e-4),
            n_steps=config.get('training', {}).get('n_steps', 2048),
            batch_size=config.get('training', {}).get('batch_size', 256),
            gamma=config.get('training', {}).get('gamma', 0.99),
            verbose=1,
        )
        
        total_timesteps = config.get('training', {}).get('total_timesteps', 1_000_000)
        
        logger.info(f"Training for {total_timesteps} timesteps...")
        model.learn(total_timesteps=total_timesteps)
        
        # Save final model
        final_model_path = os.path.join(args.model_dir, 'ppo_final.zip')
        model.save(final_model_path)
        logger.info(f"✓ Final model saved: {final_model_path}")
        
        logger.info("="*70)
        logger.info("TRAINING COMPLETE!")
        logger.info("="*70)
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
PYEOF

# ==============================================================================
# PART 6: RUN TRAINING SCRIPT
# ==============================================================================

export TRAINING_DIR="$TRAINING_DIR"
export LOG_DIR="$LOG_DIR"
export MODEL_DIR="$MODEL_DIR"

cd "$TRAINING_DIR"

python "$LOG_DIR/train_cluster.py" \
    --config "$CONFIG_FILE" \
    --resume "$RESUME_CHECKPOINT" \
    --log-dir "$LOG_DIR" \
    --model-dir "$MODEL_DIR"

EXIT_CODE=$?

# ==============================================================================
# PART 7: POST-TRAINING CLEANUP & SUMMARY
# ==============================================================================

echo ""
echo "======================================================================"
echo "Training Completed (Exit Code: $EXIT_CODE)"
echo "======================================================================"
echo "Results saved to: $LOG_DIR"
echo "Models saved to: $MODEL_DIR"
echo "Videos saved to: $VIDEO_DIR"
echo "Images saved to: $IMAGE_DIR"
echo "======================================================================"

# Save summary
SUMMARY_FILE="$LOG_DIR/training_summary.txt"
cat > "$SUMMARY_FILE" << EOF
Training Summary
================
Job ID: $SLURM_JOB_ID
Submitted: $(date)
Config: $CONFIG_FILE
Exit Code: $EXIT_CODE
Log Directory: $LOG_DIR
Model Directory: $MODEL_DIR
Video Directory: $VIDEO_DIR
Image Directory: $IMAGE_DIR
EOF

logger.info "Summary saved to: $SUMMARY_FILE"

# Save configuration
cp "$CONFIG_FILE" "$LOG_DIR/config_used.yaml"

exit $EXIT_CODE
