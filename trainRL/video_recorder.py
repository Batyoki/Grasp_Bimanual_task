"""
Video and image recording for training visualization.

Records:
- Training episode videos
- Reward plots
- Performance metrics images
"""

import logging
from pathlib import Path
from typing import Optional, List

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib

# Try to import cv2, but make it optional
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# Use imageio as fallback for video encoding
try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

matplotlib.use('Agg')  # Non-interactive backend

logger = logging.getLogger(__name__)


class VideoRecorder:
    """Records videos of training episodes."""
    
    def __init__(
        self,
        video_dir: str,
        fps: int = 30,
        frame_size: tuple = (640, 480),
        codec: str = "mp4v",
    ):
        """
        Initialize video recorder.
        
        Args:
            video_dir: Directory to save videos.
            fps: Frames per second.
            frame_size: Frame size (width, height).
            codec: Video codec.
        """
        self.video_dir = Path(video_dir)
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.frame_size = frame_size
        self.codec = codec
        
        self.writer = None
        self.current_video = None
        self.frame_count = 0
        self.use_cv2 = HAS_CV2
        self.frames_buffer = []
    
    def start_recording(self, video_name: str):
        """
        Start recording a new video.
        
        Args:
            video_name: Name for the video file.
        """
        video_path = self.video_dir / f"{video_name}.mp4"
        
        self.current_video = video_path
        self.frame_count = 0
        self.frames_buffer = []
        
        if HAS_CV2:
            # Use OpenCV if available
            try:
                fourcc = cv2.VideoWriter_fourcc(*self.codec)
                self.writer = cv2.VideoWriter(
                    str(video_path),
                    fourcc,
                    self.fps,
                    self.frame_size,
                )
                self.use_cv2 = True
                logger.info(f"Started recording (cv2): {video_path}")
                return
            except Exception as e:
                logger.warning(f"cv2 recording failed: {e}, falling back to imageio")
                self.use_cv2 = False
        
        # Fallback to imageio
        if HAS_IMAGEIO:
            self.use_cv2 = False
            logger.info(f"Started recording (imageio): {video_path}")
        else:
            logger.error("Neither cv2 nor imageio available for video recording!")
            self.writer = None
    
    def record_frame(self, frame: np.ndarray):
        """
        Record a frame to the current video.
        
        Args:
            frame: Frame as numpy array (BGR or RGB).
        """
        if self.current_video is None:
            logger.warning("No video recording started")
            return
        
        try:
            # Convert to uint8 if needed
            if frame.dtype != np.uint8:
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                else:
                    frame = frame.astype(np.uint8)
            
            # Resize to match frame size
            if frame.shape[:2] != (self.frame_size[1], self.frame_size[0]):
                if HAS_CV2 and self.use_cv2:
                    frame = cv2.resize(frame, self.frame_size)
                else:
                    # Use PIL for resizing as fallback
                    pil_frame = Image.fromarray(frame)
                    pil_frame = pil_frame.resize((self.frame_size[0], self.frame_size[1]))
                    frame = np.array(pil_frame)
            
            if HAS_CV2 and self.use_cv2 and self.writer is not None:
                self.writer.write(frame)
            else:
                # Buffer frames for imageio
                self.frames_buffer.append(frame)
            
            self.frame_count += 1
        except Exception as e:
            logger.warning(f"Frame recording error: {e}")
    
    def stop_recording(self):
        """Stop recording and save the video."""
        if self.current_video is None:
            return
        
        try:
            if HAS_CV2 and self.use_cv2 and self.writer is not None:
                self.writer.release()
                logger.info(
                    f"Saved video (cv2): {self.current_video} ({self.frame_count} frames)"
                )
            elif HAS_IMAGEIO and self.frames_buffer:
                # Save using imageio
                imageio.mimsave(
                    str(self.current_video),
                    self.frames_buffer,
                    fps=self.fps,
                )
                logger.info(
                    f"Saved video (imageio): {self.current_video} ({self.frame_count} frames)"
                )
        except Exception as e:
            logger.error(f"Error saving video: {e}")
        finally:
            self.writer = None
            self.current_video = None
            self.frame_count = 0
            self.frames_buffer = []
    
    def record_episodes(
        self,
        env,
        policy,
        num_episodes: int = 5,
        episode_prefix: str = "episode",
    ):
        """
        Record multiple episodes with a policy.
        
        Args:
            env: Environment.
            policy: Policy function.
            num_episodes: Number of episodes to record.
            episode_prefix: Prefix for video names.
        """
        for episode_idx in range(num_episodes):
            obs, _ = env.reset()
            done = False
            video_name = f"{episode_prefix}_{episode_idx:03d}"
            
            self.start_recording(video_name)
            
            while not done:
                # Get action from policy
                if hasattr(policy, 'predict'):
                    action, _ = policy.predict(obs, deterministic=True)
                else:
                    action = policy(obs)
                
                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Record frame (if available from info)
                if 'rgb' in info:
                    frame = info['rgb']
                    self.record_frame(frame)
            
            self.stop_recording()


class ImageRecorder:
    """Records images of training progress."""
    
    def __init__(self, image_dir: str):
        """
        Initialize image recorder.
        
        Args:
            image_dir: Directory to save images.
        """
        self.image_dir = Path(image_dir)
        self.image_dir.mkdir(parents=True, exist_ok=True)
    
    def save_episode_frames(
        self,
        frames: List[np.ndarray],
        episode_name: str,
        max_frames: int = 10,
    ):
        """
        Save frames from an episode.
        
        Args:
            frames: List of frames.
            episode_name: Name for this episode.
            max_frames: Maximum frames to save (evenly spaced).
        """
        if not frames:
            return
        
        # Evenly sample frames
        indices = np.linspace(0, len(frames) - 1, min(max_frames, len(frames)), dtype=int)
        
        for idx, frame_idx in enumerate(indices):
            frame = frames[frame_idx]
            
            # Convert to uint8 if needed
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            
            # Save as PNG
            image_path = self.image_dir / f"{episode_name}_frame_{idx:02d}.png"
            Image.fromarray(frame).save(image_path)
        
        logger.info(f"Saved {len(indices)} frames for {episode_name}")
    
    def save_training_plot(
        self,
        metrics: dict,
        plot_name: str = "training_progress",
    ):
        """
        Save a training progress plot.
        
        Args:
            metrics: Dictionary of metric names and values.
            plot_name: Name for the plot.
        """
        fig, axes = plt.subplots(len(metrics) // 2 + len(metrics) % 2, 2, figsize=(12, 8))
        
        if len(metrics) == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, (metric_name, values) in enumerate(metrics.items()):
            if idx >= len(axes):
                break
            
            ax = axes[idx]
            ax.plot(values, linewidth=2)
            ax.set_title(metric_name)
            ax.set_xlabel("Step")
            ax.set_ylabel("Value")
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(metrics), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plot_path = self.image_dir / f"{plot_name}.png"
        plt.savefig(plot_path, dpi=100)
        plt.close()
        
        logger.info(f"Saved training plot: {plot_path}")
    
    def save_reward_heatmap(
        self,
        rewards: np.ndarray,
        heatmap_name: str = "reward_heatmap",
    ):
        """
        Save a heatmap of rewards over time.
        
        Args:
            rewards: 2D array of rewards (time x env).
            heatmap_name: Name for the heatmap.
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        im = ax.imshow(rewards.T, aspect='auto', cmap='RdYlGn')
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Environment ID")
        ax.set_title("Rewards Over Training")
        
        plt.colorbar(im, ax=ax, label="Reward")
        plt.tight_layout()
        
        plot_path = self.image_dir / f"{heatmap_name}.png"
        plt.savefig(plot_path, dpi=100)
        plt.close()
        
        logger.info(f"Saved reward heatmap: {plot_path}")
    
    def save_observation_samples(
        self,
        observations: torch.Tensor,
        sample_name: str = "observations",
    ):
        """
        Save samples of observations (e.g., camera images).
        
        Args:
            observations: Observation tensor.
            sample_name: Name for the samples.
        """
        if len(observations.shape) < 3:
            logger.warning("Cannot visualize 1D observations")
            return
        
        # Take first 4 samples
        num_samples = min(4, observations.shape[0])
        
        for idx in range(num_samples):
            obs = observations[idx].cpu().numpy()
            
            # Handle different formats
            if obs.dtype != np.uint8:
                obs = (obs * 255).astype(np.uint8)
            
            # Save
            image_path = self.image_dir / f"{sample_name}_{idx:02d}.png"
            Image.fromarray(obs).save(image_path)
        
        logger.info(f"Saved {num_samples} observation samples")


class PerformancePlotter:
    """Plots and saves performance metrics."""
    
    def __init__(self, plot_dir: str):
        """
        Initialize performance plotter.
        
        Args:
            plot_dir: Directory to save plots.
        """
        self.plot_dir = Path(plot_dir)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.history = {}
    
    def add_metric(self, step: int, metric_name: str, value: float):
        """Add a metric data point."""
        if metric_name not in self.history:
            self.history[metric_name] = {"steps": [], "values": []}
        
        self.history[metric_name]["steps"].append(step)
        self.history[metric_name]["values"].append(value)
    
    def plot_metrics(self, plot_name: str = "metrics"):
        """Plot and save all metrics."""
        if not self.history:
            logger.warning("No metrics to plot")
            return
        
        num_metrics = len(self.history)
        num_cols = 2
        num_rows = (num_metrics + num_cols - 1) // num_cols
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(14, 5 * num_rows))
        
        if num_metrics == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, (metric_name, data) in enumerate(self.history.items()):
            ax = axes[idx]
            
            ax.plot(data["steps"], data["values"], linewidth=2, marker='o', markersize=3)
            ax.set_title(metric_name, fontsize=12, fontweight='bold')
            ax.set_xlabel("Training Step")
            ax.set_ylabel("Value")
            ax.grid(True, alpha=0.3)
        
        # Hide unused plots
        for idx in range(num_metrics, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plot_path = self.plot_dir / f"{plot_name}.png"
        plt.savefig(plot_path, dpi=100)
        plt.close()
        
        logger.info(f"Saved metrics plot: {plot_path}")


# Alias for backward compatibility
MetricsPlotter = ImageRecorder
