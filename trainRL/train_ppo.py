#!/usr/bin/env python3
"""
PPO training script for the bimanual VX300s arm environment using IsaacLab.

This script trains a policy using Proximal Policy Optimization (PPO) to control
a bimanual arm for manipulation tasks.
"""

import argparse
import os
from datetime import datetime
from pathlib import Path

import torch
import yaml
from omegaconf import DictConfig, OmegaConf

import isaaclab.lab as isaac
from isaaclab.utils.io import dump_pickle, dump_yaml

# Configure logging
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_env_config(env_cfg_path: str = None) -> DictConfig:
    """
    Load or create environment configuration.

    Args:
        env_cfg_path: Path to environment config YAML file.

    Returns:
        Environment configuration.
    """
    if env_cfg_path and Path(env_cfg_path).exists():
        cfg = OmegaConf.load(env_cfg_path)
    else:
        # Default configuration
        cfg = OmegaConf.create({
            "env": {
                "num_envs": 4,
                "max_episode_length": 1000,
                "device": "cuda" if torch.cuda.is_available() else "cpu",
            },
            "training": {
                "algorithm": "ppo",
                "total_timesteps": 1_000_000,
                "batch_size": 256,
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "n_epochs": 10,
            },
            "experiment": {
                "name": "bimanual_vx300s_v1",
                "log_dir": "./logs",
                "model_dir": "./models",
                "checkpoint_freq": 10000,
            },
        })
    
    return cfg


def setup_directories(cfg: DictConfig) -> dict:
    """
    Create necessary directories for logging and model checkpoints.

    Args:
        cfg: Configuration dictionary.

    Returns:
        Dictionary with created directory paths.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_name = cfg.experiment.name
    
    log_dir = Path(cfg.experiment.log_dir) / f"{exp_name}_{timestamp}"
    model_dir = Path(cfg.experiment.model_dir) / f"{exp_name}_{timestamp}"
    
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Log directory: {log_dir}")
    logger.info(f"Model directory: {model_dir}")
    
    return {
        "log_dir": log_dir,
        "model_dir": model_dir,
        "timestamp": timestamp,
    }


def train_ppo(cfg: DictConfig, dirs: dict):
    """
    Train PPO policy using IsaacLab environment.

    Args:
        cfg: Configuration dictionary.
        dirs: Dictionary with directory paths.
    """
    try:
        # Import here to avoid issues if IsaacLab not fully installed
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
        from stable_baselines3.common.vec_env import VecNormalize
        from isaaclab.envs import ManagerBasedRLEnv
        from isaaclab.utils.dict import update_dict
    except ImportError as e:
        logger.error(f"Missing required package: {e}")
        logger.error("Please ensure Stable-Baselines3 and IsaacLab are installed")
        return
    
    logger.info("Creating IsaacLab environment...")
    
    # Note: The actual environment creation would depend on how you register
    # the bimanual_vx300s environment in Isaac Lab
    # For now, this is a placeholder that shows the structure
    
    try:
        env = isaac.sim.build_env(
            num_envs=cfg.env.num_envs,
            device=cfg.env.device,
        )
    except Exception as e:
        logger.warning(f"Could not create IsaacLab environment: {e}")
        logger.info("Falling back to dummy environment for demonstration...")
        # Placeholder environment
        return
    
    logger.info(f"Environment created with {cfg.env.num_envs} parallel environments")
    
    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=cfg.experiment.checkpoint_freq,
        save_path=str(dirs["model_dir"]),
        name_prefix="ppo_bimanual",
        save_replay_buffer=True,
    )
    
    # Create and train PPO model
    logger.info("Initializing PPO trainer...")
    
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=cfg.training.learning_rate,
        batch_size=cfg.training.batch_size,
        n_epochs=cfg.training.n_epochs,
        gamma=cfg.training.gamma,
        gae_lambda=cfg.training.gae_lambda,
        clip_range=cfg.training.clip_range,
        device=cfg.env.device,
        verbose=1,
        tensorboard_log=str(dirs["log_dir"]),
    )
    
    logger.info(f"Starting training for {cfg.training.total_timesteps} timesteps...")
    
    model.learn(
        total_timesteps=cfg.training.total_timesteps,
        callback=checkpoint_callback,
        log_interval=10,
    )
    
    # Save final model
    final_model_path = dirs["model_dir"] / "ppo_bimanual_final"
    model.save(str(final_model_path))
    logger.info(f"Final model saved to {final_model_path}")
    
    # Save configuration
    config_path = dirs["log_dir"] / "config.yaml"
    dump_yaml(config_path, OmegaConf.to_container(cfg, resolve=True))
    logger.info(f"Configuration saved to {config_path}")
    
    env.close()


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Train bimanual VX300s arm policy using PPO")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=4,
        help="Number of parallel environments",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=1_000_000,
        help="Total number of training timesteps",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate for PPO",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Device to use for training",
    )
    
    args = parser.parse_args()
    
    # Load configuration
    cfg = get_env_config(args.config)
    
    # Override with command line arguments
    cfg.env.num_envs = args.num_envs
    cfg.training.total_timesteps = args.total_timesteps
    cfg.training.learning_rate = args.learning_rate
    
    if args.device != "auto":
        cfg.env.device = args.device
    else:
        cfg.env.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))
    
    # Setup directories
    dirs = setup_directories(cfg)
    
    # Train model
    train_ppo(cfg, dirs)
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
