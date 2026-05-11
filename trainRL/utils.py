#!/usr/bin/env python3
"""
Utilities for the bimanual VX300s RL environment.

This module provides helper functions for environment setup, logging,
configuration management, and visualization.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf, DictConfig

logger = logging.getLogger(__name__)


class ConfigManager:
    """Manages experiment configurations."""
    
    @staticmethod
    def create_default_config() -> DictConfig:
        """
        Create a default configuration for the environment.
        
        Returns:
            Default OmegaConf configuration.
        """
        cfg = OmegaConf.create({
            # Environment settings
            "env": {
                "name": "BimanualVX300s",
                "num_envs": 4,
                "max_episode_length": 1000,
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "render": False,
                "render_freq": 10,
            },
            # Observation settings
            "observation": {
                "include_images": False,
                "image_size": (64, 64),
                "camera_names": ["cam_high", "cam_low", "cam_left_wrist"],
                "normalize_obs": True,
            },
            # Action settings
            "action": {
                "action_type": "joint_position",
                "action_scale": 1.0,
                "use_relative_actions": True,
            },
            # Reward settings
            "reward": {
                "reach_weight": 1.0,
                "action_penalty_weight": 0.01,
                "gripper_penalty_weight": 0.001,
                "success_bonus": 10.0,
                "success_threshold": 0.05,
            },
            # Training settings
            "training": {
                "algorithm": "ppo",
                "total_timesteps": 1_000_000,
                "num_epochs": 10,
                "batch_size": 256,
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.0,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5,
            },
            # Experiment settings
            "experiment": {
                "name": "bimanual_vx300s_v1",
                "log_dir": "./logs",
                "model_dir": "./models",
                "seed": 42,
                "checkpoint_freq": 10000,
                "eval_freq": 10000,
                "eval_episodes": 5,
            },
        })
        
        return cfg
    
    @staticmethod
    def load_config(config_path: str) -> DictConfig:
        """
        Load configuration from a YAML file.
        
        Args:
            config_path: Path to the YAML configuration file.
            
        Returns:
            Loaded OmegaConf configuration.
        """
        if not Path(config_path).exists():
            logger.warning(f"Config file not found: {config_path}")
            logger.info("Using default configuration")
            return ConfigManager.create_default_config()
        
        cfg = OmegaConf.load(config_path)
        logger.info(f"Loaded configuration from {config_path}")
        
        return cfg
    
    @staticmethod
    def save_config(cfg: DictConfig, output_path: str) -> None:
        """
        Save configuration to a YAML file.
        
        Args:
            cfg: Configuration to save.
            output_path: Path to save the configuration.
        """
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            OmegaConf.save(cfg, f)
        
        logger.info(f"Configuration saved to {output_path}")
    
    @staticmethod
    def merge_configs(base_cfg: DictConfig, overrides: Dict[str, Any]) -> DictConfig:
        """
        Merge override values into a base configuration.
        
        Args:
            base_cfg: Base configuration.
            overrides: Dictionary of overrides.
            
        Returns:
            Merged configuration.
        """
        merged_cfg = OmegaConf.merge(base_cfg, OmegaConf.create(overrides))
        return merged_cfg


class DirectoryManager:
    """Manages experiment directories."""
    
    @staticmethod
    def setup_directories(cfg: DictConfig, timestamp: Optional[str] = None) -> Dict[str, Path]:
        """
        Create and setup necessary directories for an experiment.
        
        Args:
            cfg: Configuration containing directory paths.
            timestamp: Optional timestamp string for directory naming.
            
        Returns:
            Dictionary with created directory paths.
        """
        from datetime import datetime
        
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        exp_name = cfg.experiment.name
        
        log_dir = Path(cfg.experiment.log_dir) / f"{exp_name}_{timestamp}"
        model_dir = Path(cfg.experiment.model_dir) / f"{exp_name}_{timestamp}"
        tensorboard_dir = Path(cfg.experiment.log_dir) / "tensorboard" / f"{exp_name}_{timestamp}"
        
        log_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Log directory: {log_dir}")
        logger.info(f"Model directory: {model_dir}")
        logger.info(f"TensorBoard directory: {tensorboard_dir}")
        
        return {
            "log_dir": log_dir,
            "model_dir": model_dir,
            "tensorboard_dir": tensorboard_dir,
            "timestamp": timestamp,
        }


class TelemetryLogger:
    """Logs training telemetry."""
    
    def __init__(self, log_dir: Path):
        """
        Initialize telemetry logger.
        
        Args:
            log_dir: Directory to save logs.
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "training_log.csv"
        self.history = []
    
    def log_step(self, step: int, metrics: Dict[str, float]) -> None:
        """
        Log a training step.
        
        Args:
            step: Training step number.
            metrics: Dictionary of metrics to log.
        """
        entry = {"step": step, **metrics}
        self.history.append(entry)
        
        # Save to CSV (append mode)
        import csv
        mode = "a" if self.log_file.exists() else "w"
        with open(self.log_file, mode, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=entry.keys())
            if mode == "w":
                writer.writeheader()
            writer.writerow(entry)
    
    def get_history(self) -> list:
        """Get all logged entries."""
        return self.history


class MetricsTracker:
    """Tracks and computes metrics during training."""
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []
    
    def record_episode(self, reward: float, length: int, success: bool = False) -> None:
        """
        Record an episode.
        
        Args:
            reward: Total reward for the episode.
            length: Length of the episode.
            success: Whether the episode was successful.
        """
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.episode_successes.append(success)
    
    def get_stats(self) -> Dict[str, float]:
        """
        Get statistics from recorded episodes.
        
        Returns:
            Dictionary of statistics.
        """
        if len(self.episode_rewards) == 0:
            return {}
        
        rewards = np.array(self.episode_rewards)
        lengths = np.array(self.episode_lengths)
        successes = np.array(self.episode_successes)
        
        stats = {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "max_reward": float(np.max(rewards)),
            "min_reward": float(np.min(rewards)),
            "mean_length": float(np.mean(lengths)),
            "success_rate": float(np.mean(successes)) if len(successes) > 0 else 0.0,
        }
        
        return stats
    
    def reset(self) -> None:
        """Reset the tracker."""
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []


def set_seed(seed: int, device: str = "cpu") -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed.
        device: Device type ("cpu" or "cuda").
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if device == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info(f"Set random seed to {seed}")
