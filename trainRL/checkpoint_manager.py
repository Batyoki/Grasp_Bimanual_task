"""
Checkpoint management for training persistence.

Enables:
- Saving training state and models
- Loading previous checkpoints
- Resuming from checkpoint
- Automatic checkpointing during training
"""

import os
import pickle
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import torch
import yaml
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages training checkpoints for save/load/resume."""
    
    def __init__(self, checkpoint_dir: str, keep_best_n: int = 3):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints.
            keep_best_n: Number of best checkpoints to keep.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_best_n = keep_best_n
        self.checkpoint_history = []
    
    def save_checkpoint(
        self,
        step: int,
        model,
        optimizer: Optional[torch.optim.Optimizer],
        metrics: Dict[str, float],
        config: Dict[str, Any],
        is_best: bool = False,
    ) -> Path:
        """
        Save a training checkpoint.
        
        Args:
            step: Training step number.
            model: Trained model.
            optimizer: Optimizer state.
            metrics: Training metrics.
            config: Configuration dictionary.
            is_best: Whether this is the best model so far.
            
        Returns:
            Path to saved checkpoint.
        """
        checkpoint_data = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "config": config,
            "model_state": model.state_dict() if hasattr(model, 'state_dict') else None,
            "optimizer_state": optimizer.state_dict() if optimizer else None,
        }
        
        # Filename with step number
        checkpoint_name = f"checkpoint_step_{step}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Save checkpoint
        torch.save(checkpoint_data, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Save as best if applicable
        if is_best:
            best_path = self.checkpoint_dir / "checkpoint_best.pt"
            shutil.copy(checkpoint_path, best_path)
            logger.info(f"New best checkpoint: {best_path}")
        
        # Track in history
        self.checkpoint_history.append({
            "step": step,
            "path": str(checkpoint_path),
            "metrics": metrics,
            "is_best": is_best,
        })
        
        # Clean up old checkpoints (keep only best N)
        self._cleanup_old_checkpoints()
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file.
            
        Returns:
            Checkpoint data dictionary.
        """
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint_data = torch.load(checkpoint_path, map_location="cpu")
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
        
        return checkpoint_data
    
    def load_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Load the latest checkpoint.
        
        Returns:
            Checkpoint data or None if no checkpoints exist.
        """
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_step_*.pt"))
        if not checkpoints:
            return None
        
        latest = checkpoints[-1]
        return self.load_checkpoint(str(latest))
    
    def load_best_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Load the best checkpoint.
        
        Returns:
            Checkpoint data or None if no best checkpoint exists.
        """
        best_path = self.checkpoint_dir / "checkpoint_best.pt"
        if not best_path.exists():
            return None
        
        return self.load_checkpoint(str(best_path))
    
    def resume_from_checkpoint(
        self,
        checkpoint_path: Optional[str] = None,
        model=None,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Tuple[int, Dict[str, float], Dict[str, Any]]:
        """
        Resume training from a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint. If None, loads latest.
            model: Model to load state into.
            optimizer: Optimizer to load state into.
            
        Returns:
            Tuple of (step, metrics, config).
        """
        if checkpoint_path is None:
            checkpoint_data = self.load_latest_checkpoint()
        else:
            checkpoint_data = self.load_checkpoint(checkpoint_path)
        
        if checkpoint_data is None:
            logger.warning("No checkpoint to resume from")
            return 0, {}, {}
        
        step = checkpoint_data["step"]
        metrics = checkpoint_data["metrics"]
        config = checkpoint_data["config"]
        
        # Load model state
        if model and checkpoint_data["model_state"]:
            model.load_state_dict(checkpoint_data["model_state"])
            logger.info(f"Model state loaded (step {step})")
        
        # Load optimizer state
        if optimizer and checkpoint_data["optimizer_state"]:
            optimizer.load_state_dict(checkpoint_data["optimizer_state"])
            logger.info(f"Optimizer state loaded (step {step})")
        
        logger.info(f"Resumed from step {step}")
        
        return step, metrics, config
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the best N."""
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_step_*.pt"),
            key=lambda p: int(p.stem.split("_")[-1]),
        )
        
        # Keep the best N (by step number, which roughly corresponds to training progress)
        if len(checkpoints) > self.keep_best_n:
            for checkpoint in checkpoints[:-self.keep_best_n]:
                checkpoint.unlink()
                logger.debug(f"Removed old checkpoint: {checkpoint}")
    
    def get_checkpoint_info(self) -> str:
        """Get human-readable checkpoint information."""
        best_path = self.checkpoint_dir / "checkpoint_best.pt"
        latest_path = sorted(self.checkpoint_dir.glob("checkpoint_step_*.pt"))[-1] if self.checkpoint_dir.glob("checkpoint_step_*.pt") else None
        
        info = f"Checkpoint Directory: {self.checkpoint_dir}\n"
        
        if best_path.exists():
            best_data = torch.load(best_path, map_location="cpu")
            info += f"Best Checkpoint (Step {best_data['step']}): {best_path}\n"
            info += f"  Metrics: {best_data['metrics']}\n"
        
        if latest_path:
            latest_data = torch.load(latest_path, map_location="cpu")
            info += f"Latest Checkpoint (Step {latest_data['step']}): {latest_path}\n"
            info += f"  Metrics: {latest_data['metrics']}\n"
        
        return info


class TrainingState:
    """Manages overall training state for persistence."""
    
    def __init__(self, state_file: str):
        """
        Initialize training state manager.
        
        Args:
            state_file: Path to save training state.
        """
        self.state_file = Path(state_file)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
    
    def save_state(self, state: Dict[str, Any]):
        """Save training state to file."""
        with open(self.state_file, 'w') as f:
            yaml.dump(state, f)
        logger.info(f"Training state saved: {self.state_file}")
    
    def load_state(self) -> Optional[Dict[str, Any]]:
        """Load training state from file."""
        if not self.state_file.exists():
            return None
        
        with open(self.state_file, 'r') as f:
            state = yaml.safe_load(f)
        
        logger.info(f"Training state loaded: {self.state_file}")
        return state
    
    def update_state(self, updates: Dict[str, Any]):
        """Update training state."""
        state = self.load_state() or {}
        state.update(updates)
        self.save_state(state)


def create_checkpoint_from_sb3_model(
    model,
    checkpoint_dir: str,
    step: int,
    metrics: Dict[str, float],
) -> Path:
    """
    Create a checkpoint from a Stable-Baselines3 model.
    
    Args:
        model: Stable-Baselines3 model.
        checkpoint_dir: Directory to save checkpoint.
        step: Training step.
        metrics: Training metrics.
        
    Returns:
        Path to saved checkpoint.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = checkpoint_dir / f"model_step_{step}.zip"
    model.save(str(checkpoint_path))
    
    # Also save metrics
    metrics_file = checkpoint_dir / f"metrics_step_{step}.yaml"
    with open(metrics_file, 'w') as f:
        yaml.dump(metrics, f)
    
    logger.info(f"SB3 Model checkpoint saved: {checkpoint_path}")
    
    return checkpoint_path


def load_sb3_model_from_checkpoint(
    checkpoint_path: str,
    model_class,
    env=None,
):
    """
    Load a Stable-Baselines3 model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint.
        model_class: Model class (e.g., PPO).
        env: Environment (optional, for loading policy).
        
    Returns:
        Loaded model.
    """
    model = model_class.load(checkpoint_path, env=env)
    logger.info(f"SB3 Model loaded: {checkpoint_path}")
    
    return model
