#!/usr/bin/env python3
"""
Evaluation script for the trained bimanual VX300s arm policy.

This script evaluates a trained policy on the environment and collects
performance metrics.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def evaluate_policy(
    model_path: str,
    config_path: str,
    num_episodes: int = 10,
    render: bool = True,
):
    """
    Evaluate a trained policy.

    Args:
        model_path: Path to the trained model.
        config_path: Path to the configuration file.
        num_episodes: Number of episodes to evaluate.
        render: Whether to render the environment.
    """
    try:
        from stable_baselines3 import PPO
    except ImportError:
        logger.error("Stable-Baselines3 not installed. Please install it first.")
        return
    
    # Load configuration
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    logger.info(f"Loading model from {model_path}")
    model = PPO.load(model_path)
    
    logger.info(f"Evaluating for {num_episodes} episodes...")
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        obs = model.get_env().reset()
        done = False
        episode_reward = 0.0
        episode_length = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = model.get_env().step(action)
            episode_reward += reward
            episode_length += 1
            
            if render:
                model.get_env().render()
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        logger.info(
            f"Episode {episode + 1}: Reward={episode_reward:.2f}, "
            f"Length={episode_length}"
        )
    
    # Summary statistics
    logger.info("\n" + "="*50)
    logger.info("Evaluation Summary:")
    logger.info(f"Mean Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    logger.info(f"Mean Episode Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    logger.info(f"Max Reward: {np.max(episode_rewards):.2f}")
    logger.info(f"Min Reward: {np.min(episode_rewards):.2f}")
    logger.info("="*50 + "\n")
    
    return {
        "rewards": episode_rewards,
        "lengths": episode_lengths,
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
    }


def main():
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(description="Evaluate trained bimanual VX300s arm policy")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the trained model file",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        required=True,
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=10,
        help="Number of episodes to evaluate",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable rendering during evaluation",
    )
    parser.add_argument(
        "--save-results",
        type=str,
        default=None,
        help="Path to save evaluation results as YAML",
    )
    
    args = parser.parse_args()
    
    # Evaluate policy
    results = evaluate_policy(
        model_path=args.model_path,
        config_path=args.config_path,
        num_episodes=args.num_episodes,
        render=not args.no_render,
    )
    
    # Save results if requested
    if args.save_results:
        output_path = Path(args.save_results)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(results, f)
        
        logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
