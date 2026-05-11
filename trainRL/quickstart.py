#!/usr/bin/env python3
"""
Quick start script for the bimanual VX300s environment.

This script provides an interactive way to set up and run training,
evaluation, or visualization of the environment.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from utils import ConfigManager, DirectoryManager, set_seed

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_banner():
    """Print a welcome banner."""
    banner = """
    ╔════════════════════════════════════════════════════════════╗
    ║  Bimanual VX300s Arm - IsaacLab RL Training System        ║
    ║                                                            ║
    ║  Training bimanual robotic arms for manipulation tasks    ║
    ╚════════════════════════════════════════════════════════════╝
    """
    print(banner)


def show_menu() -> str:
    """Show interactive menu and return user choice."""
    menu = """
    What would you like to do?
    
    1. Train a new policy (PPO)
    2. Evaluate a trained policy
    3. View training results
    4. Generate a configuration file
    5. Run a quick demo
    0. Exit
    
    Enter your choice (0-5): """
    
    return input(menu).strip()


def train_mode(args=None):
    """Enter training mode."""
    logger.info("Entering Training Mode...")
    
    import train_ppo
    train_ppo.main()


def evaluate_mode(args=None):
    """Enter evaluation mode."""
    logger.info("Entering Evaluation Mode...")
    
    model_path = input("Enter path to trained model (ZIP file): ").strip()
    config_path = input("Enter path to configuration file: ").strip()
    num_episodes = int(input("Number of episodes to evaluate (default 10): ") or "10")
    render = input("Render episodes? (y/n, default y): ").lower() != "n"
    
    import evaluate
    evaluate.evaluate_policy(
        model_path=model_path,
        config_path=config_path,
        num_episodes=num_episodes,
        render=render,
    )


def generate_config_mode():
    """Generate a custom configuration file."""
    logger.info("Generating Custom Configuration...")
    
    cfg = ConfigManager.create_default_config()
    
    print("\nEnter configuration parameters (press Enter to keep defaults):")
    
    # Get user inputs
    num_envs = input(f"Number of parallel environments (default {cfg.env.num_envs}): ")
    if num_envs.strip():
        cfg.env.num_envs = int(num_envs)
    
    max_episode_length = input(f"Max episode length (default {cfg.env.max_episode_length}): ")
    if max_episode_length.strip():
        cfg.env.max_episode_length = int(max_episode_length)
    
    total_timesteps = input(f"Total training timesteps (default {cfg.training.total_timesteps}): ")
    if total_timesteps.strip():
        cfg.training.total_timesteps = int(total_timesteps)
    
    learning_rate = input(f"Learning rate (default {cfg.training.learning_rate}): ")
    if learning_rate.strip():
        cfg.training.learning_rate = float(learning_rate)
    
    experiment_name = input(f"Experiment name (default {cfg.experiment.name}): ")
    if experiment_name.strip():
        cfg.experiment.name = experiment_name
    
    # Save configuration
    output_filename = input("Output filename (default custom_config.yaml): ").strip()
    if not output_filename:
        output_filename = "custom_config.yaml"
    
    ConfigManager.save_config(cfg, output_filename)
    logger.info(f"\n✓ Configuration saved to {output_filename}")
    logger.info("\nYou can now train with this configuration:")
    logger.info(f"  python train_ppo.py --config {output_filename}")


def show_results():
    """Show training results and statistics."""
    logger.info("Viewing Training Results...")
    
    log_dir = input("Enter log directory path: ").strip()
    log_path = Path(log_dir)
    
    if not log_path.exists():
        logger.error(f"Log directory not found: {log_dir}")
        return
    
    # Find config and training log
    config_file = log_path / "config.yaml"
    log_csv = log_path / "training_log.csv"
    
    if config_file.exists():
        logger.info("\n" + "="*50)
        logger.info("Configuration:")
        logger.info("="*50)
        with open(config_file, 'r') as f:
            print(f.read())
    
    if log_csv.exists():
        logger.info("\n" + "="*50)
        logger.info("Training Statistics:")
        logger.info("="*50)
        import csv
        with open(log_csv, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
            if rows:
                # Show summary stats
                last_row = rows[-1]
                logger.info(f"Total steps: {last_row.get('step', 'N/A')}")
                logger.info(f"Final reward: {last_row.get('reward', 'N/A')}")
                
                # Show last 10 entries
                logger.info("\nLast 10 training steps:")
                for row in rows[-10:]:
                    logger.info(f"  Step {row.get('step', 'N/A')}: Reward={row.get('reward', 'N/A')}")


def demo_mode():
    """Run a quick demo of the environment."""
    logger.info("Running Quick Demo...")
    
    try:
        from bimanual_vx300s_env import BimanualVX300sEnvCfg
        logger.info("✓ Environment imported successfully")
        logger.info("✓ IsaacLab is properly installed")
        
        # Show environment info
        cfg = BimanualVX300sEnvCfg()
        logger.info(f"\nEnvironment Configuration:")
        logger.info(f"  - Number of environments: {cfg.num_envs}")
        logger.info(f"  - Max episode length: {cfg.max_episode_length}")
        logger.info(f"  - Device: {cfg.sim.device if hasattr(cfg.sim, 'device') else 'CPU'}")
        
    except ImportError as e:
        logger.error(f"Cannot import environment: {e}")
        logger.error("Please ensure IsaacLab and all dependencies are installed.")


def main():
    """Main interactive menu."""
    parser = argparse.ArgumentParser(description="Bimanual VX300s RL Training Quick Start")
    parser.add_argument("--train", action="store_true", help="Start training directly")
    parser.add_argument("--eval", action="store_true", help="Start evaluation mode")
    parser.add_argument("--config", type=str, default=None, help="Configuration file path")
    
    args = parser.parse_args()
    
    print_banner()
    
    # Direct mode if flags provided
    if args.train:
        train_mode(args)
        return
    
    if args.eval:
        evaluate_mode(args)
        return
    
    # Interactive menu
    while True:
        choice = show_menu()
        
        if choice == "1":
            train_mode()
        elif choice == "2":
            evaluate_mode()
        elif choice == "3":
            show_results()
        elif choice == "4":
            generate_config_mode()
        elif choice == "5":
            demo_mode()
        elif choice == "0":
            logger.info("Goodbye!")
            sys.exit(0)
        else:
            logger.warning("Invalid choice. Please try again.")
        
        print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        sys.exit(1)
