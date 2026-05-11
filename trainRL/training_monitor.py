#!/usr/bin/env python3
"""
Training monitoring and analysis tools.

Provides:
- Real-time training monitoring
- Checkpoint analysis
- Video/metrics viewing
- Training summary generation
"""

import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class TrainingMonitor:
    """Monitor active training runs."""
    
    def __init__(self, log_dir=None):
        """Initialize monitor."""
        self.log_dir = Path(log_dir or "logs")
        self.checkpoint_dir = self.log_dir / "checkpoints"
        self.video_dir = self.log_dir / "videos"
        self.tensorboard_dir = self.log_dir / "tensorboard"
    
    def monitor_job(self, job_id):
        """Monitor SLURM job status and output."""
        logger.info(f"Monitoring job {job_id}...")
        logger.info("-" * 60)
        
        # Get job status
        result = subprocess.run(
            ['scontrol', 'show', 'job', job_id],
            capture_output=True,
            text=True,
        )
        
        if result.returncode == 0:
            # Parse key info
            for line in result.stdout.split('\n'):
                if any(key in line for key in ['JobID', 'JobName', 'State', 'RunTime', 'TimeLeft']):
                    logger.info(line.strip())
        else:
            logger.warning(f"Could not find job {job_id}")
        
        # Show log file if exists
        log_files = list(self.log_dir.glob(f"slurm_{job_id}.log"))
        if log_files:
            log_file = log_files[0]
            logger.info(f"\nLatest log entries from {log_file.name}:")
            logger.info("-" * 60)
            
            result = subprocess.run(
                ['tail', '-20', str(log_file)],
                capture_output=True,
                text=True,
            )
            
            print(result.stdout)
    
    def analyze_checkpoints(self):
        """Analyze saved checkpoints."""
        if not self.checkpoint_dir.exists():
            logger.warning(f"No checkpoint directory: {self.checkpoint_dir}")
            return
        
        logger.info("Checkpoint Analysis")
        logger.info("-" * 60)
        
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        if not checkpoints:
            logger.warning("No checkpoints found")
            return
        
        logger.info(f"Total checkpoints: {len(checkpoints)}")
        logger.info("\nMost recent checkpoints:")
        
        for i, ckpt in enumerate(checkpoints[:5]):
            size_mb = ckpt.stat().st_size / 1024 / 1024
            mtime = datetime.fromtimestamp(ckpt.stat().st_mtime)
            
            # Try to load metadata
            metadata_file = ckpt.with_suffix('.json')
            metadata = None
            if metadata_file.exists():
                with open(metadata_file) as f:
                    metadata = json.load(f)
            
            logger.info(f"  {i+1}. {ckpt.name}")
            logger.info(f"     Size: {size_mb:.1f} MB")
            logger.info(f"     Time: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
            
            if metadata:
                logger.info(f"     Episode: {metadata.get('episode', 'N/A')}")
                logger.info(f"     Best Reward: {metadata.get('metrics', {}).get('best_reward', 'N/A')}")
    
    def list_videos(self):
        """List recorded videos."""
        if not self.video_dir.exists():
            logger.warning(f"No video directory: {self.video_dir}")
            return
        
        logger.info("Video Recording Status")
        logger.info("-" * 60)
        
        videos = sorted(
            self.video_dir.glob("episode_*.mp4"),
            key=lambda p: int(p.stem.split('_')[1]),
        )
        
        if not videos:
            logger.warning("No videos found")
            return
        
        logger.info(f"Total videos: {len(videos)}")
        logger.info("\nMost recent videos:")
        
        for video in videos[-5:]:
            episode_num = video.stem.split('_')[1]
            size_mb = video.stat().st_size / 1024 / 1024
            logger.info(f"  Episode {episode_num}: {size_mb:.1f} MB")
    
    def list_metrics(self):
        """List metric plots."""
        if not self.log_dir.exists():
            logger.warning(f"No log directory: {self.log_dir}")
            return
        
        logger.info("Metric Plots")
        logger.info("-" * 60)
        
        plots = sorted(
            self.log_dir.glob("**/episode_*_metrics.png"),
            key=lambda p: int(p.stem.split('_')[1]),
        )
        
        if not plots:
            logger.warning("No metric plots found")
            return
        
        logger.info(f"Total metric plots: {len(plots)}")
        logger.info("\nMost recent metric plots:")
        
        for plot in plots[-5:]:
            episode_num = plot.stem.split('_')[1]
            logger.info(f"  Episode {episode_num}: {plot.relative_to(self.log_dir)}")
    
    def get_training_summary(self):
        """Generate training summary."""
        logger.info("Training Summary")
        logger.info("-" * 60)
        
        # Count files
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.pt")) if self.checkpoint_dir.exists() else []
        videos = list(self.video_dir.glob("episode_*.mp4")) if self.video_dir.exists() else []
        plots = list(self.log_dir.glob("**/episode_*_metrics.png")) if self.log_dir.exists() else []
        
        logger.info(f"Checkpoints saved: {len(checkpoints)}")
        logger.info(f"Videos recorded: {len(videos)}")
        logger.info(f"Metric plots generated: {len(plots)}")
        
        # Try to get training stats from latest checkpoint
        if checkpoints:
            latest_ckpt = max(checkpoints, key=lambda p: p.stat().st_mtime)
            metadata_file = latest_ckpt.with_suffix('.json')
            
            if metadata_file.exists():
                with open(metadata_file) as f:
                    metadata = json.load(f)
                
                logger.info(f"\nLatest checkpoint stats:")
                logger.info(f"  Episode: {metadata.get('episode', 'N/A')}")
                logger.info(f"  Step: {metadata.get('step', 'N/A')}")
                metrics = metadata.get('metrics', {})
                for key, val in metrics.items():
                    logger.info(f"  {key}: {val}")
    
    def launch_tensorboard(self, port=6006):
        """Launch TensorBoard for monitoring."""
        if not self.tensorboard_dir.exists():
            logger.warning(f"No TensorBoard directory: {self.tensorboard_dir}")
            return
        
        logger.info(f"Starting TensorBoard on port {port}...")
        logger.info(f"View at: http://localhost:{port}")
        
        subprocess.run(
            ['tensorboard', '--logdir', str(self.tensorboard_dir), '--port', str(port)],
        )


def main():
    """CLI for training monitoring."""
    parser = argparse.ArgumentParser(description="Training Monitor and Analyzer")
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Monitor command
    monitor_parser = subparsers.add_parser('monitor', help='Monitor active job')
    monitor_parser.add_argument('job_id', type=str)
    monitor_parser.add_argument('--log-dir', type=str, default='logs')
    
    # Checkpoints command
    ckpt_parser = subparsers.add_parser('checkpoints', help='Analyze checkpoints')
    ckpt_parser.add_argument('--log-dir', type=str, default='logs')
    
    # Videos command
    video_parser = subparsers.add_parser('videos', help='List videos')
    video_parser.add_argument('--log-dir', type=str, default='logs')
    
    # Metrics command
    metrics_parser = subparsers.add_parser('metrics', help='List metrics plots')
    metrics_parser.add_argument('--log-dir', type=str, default='logs')
    
    # Summary command
    summary_parser = subparsers.add_parser('summary', help='Training summary')
    summary_parser.add_argument('--log-dir', type=str, default='logs')
    
    # Tensorboard command
    tb_parser = subparsers.add_parser('tensorboard', help='Launch TensorBoard')
    tb_parser.add_argument('--log-dir', type=str, default='logs')
    tb_parser.add_argument('--port', type=int, default=6006)
    
    args = parser.parse_args()
    
    monitor = TrainingMonitor(log_dir=args.log_dir)
    
    if args.command == 'monitor':
        monitor.monitor_job(args.job_id)
    elif args.command == 'checkpoints':
        monitor.analyze_checkpoints()
    elif args.command == 'videos':
        monitor.list_videos()
    elif args.command == 'metrics':
        monitor.list_metrics()
    elif args.command == 'summary':
        monitor.get_training_summary()
    elif args.command == 'tensorboard':
        monitor.launch_tensorboard(args.port)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
