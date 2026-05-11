#!/usr/bin/env python3
"""
Utility scripts for managing training on GPU cluster.

Provides:
- Easy job submission
- Training monitoring
- Checkpoint recovery
- Batch processing
"""

import argparse
import subprocess
import json
import os
from pathlib import Path
from datetime import datetime
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClusterManager:
    """Manages cluster training jobs."""
    
    def __init__(self, base_dir=None):
        """Initialize cluster manager."""
        self.base_dir = Path(base_dir or os.path.expanduser("~/yash"))
        self.training_dir = self.base_dir / "trainRL"
    
    def submit_job(
        self,
        config_file='config.yaml',
        num_envs=4,
        total_timesteps=1_000_000,
        gpu_type='a100',
        time_limit='24:00:00',
        job_name='bimanual_rl',
        array_size=None,
    ):
        """
        Submit a training job to cluster.
        
        Args:
            config_file: Configuration file name
            num_envs: Number of parallel environments
            total_timesteps: Total training timesteps
            gpu_type: GPU type (a100 or h100)
            time_limit: Job time limit
            job_name: Job name
            array_size: For job arrays (multiple runs)
        
        Returns:
            Job ID if successful
        """
        config_path = self.training_dir / config_file
        if not config_path.exists():
            logger.error(f"Config file not found: {config_path}")
            return None
        
        # Create sbatch script
        sbatch_script = self._create_sbatch_script(
            config_file=config_file,
            num_envs=num_envs,
            total_timesteps=total_timesteps,
            gpu_type=gpu_type,
            time_limit=time_limit,
            job_name=job_name,
            array_size=array_size,
        )
        
        # Submit job
        logger.info(f"Submitting job: {job_name}")
        
        result = subprocess.run(
            ['sbatch', sbatch_script],
            capture_output=True,
            text=True,
        )
        
        if result.returncode == 0:
            job_id = result.stdout.strip().split()[-1]
            logger.info(f"✓ Job submitted successfully!")
            logger.info(f"  Job ID: {job_id}")
            logger.info(f"  GPU: {gpu_type}")
            logger.info(f"  Time limit: {time_limit}")
            return job_id
        else:
            logger.error(f"Job submission failed: {result.stderr}")
            return None
    
    def _create_sbatch_script(
        self,
        config_file,
        num_envs,
        total_timesteps,
        gpu_type,
        time_limit,
        job_name,
        array_size,
    ):
        """Create sbatch submission script."""
        gpu_partition = f"gpu-{gpu_type}"
        
        script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={gpu_partition}
#SBATCH --gres=gpu:1
#SBATCH --time={time_limit}
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/slurm_%j.log
#SBATCH --error=logs/slurm_%j.err
"""
        
        if array_size:
            script += f"#SBATCH --array=1-{array_size}\n"
        
        script += f"""
export PYTHONUNBUFFERED=1
export ACCEPT_EULA=Y
export ISAACSIM_ACCEPT_EULA=Y
export OMNI_KIT_ACCEPT_EULA=Y
export PRIVACY_CONSENT=Y

cd {self.training_dir}

source ~/yash/miniforge3/bin/activate
conda activate isaac_fresh

python train_advanced.py \\
    --config {config_file} \\
    --num-envs {num_envs} \\
    --total-timesteps {total_timesteps}
"""
        
        # Write script to temp file
        script_path = Path(f"/tmp/sbatch_{job_name}_{datetime.now().timestamp()}.sh")
        with open(script_path, 'w') as f:
            f.write(script)
        
        os.chmod(script_path, 0o755)
        
        return str(script_path)
    
    def check_job_status(self, job_id):
        """Check status of a submitted job."""
        result = subprocess.run(
            ['squeue', '--job', job_id],
            capture_output=True,
            text=True,
        )
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                logger.info(f"Job {job_id} status:")
                logger.info(lines[1])
                return True
        else:
            logger.warning(f"Job {job_id} not found in queue (may have completed)")
            return False
    
    def cancel_job(self, job_id):
        """Cancel a submitted job."""
        result = subprocess.run(
            ['scancel', job_id],
            capture_output=True,
            text=True,
        )
        
        if result.returncode == 0:
            logger.info(f"✓ Job {job_id} cancelled")
            return True
        else:
            logger.error(f"Failed to cancel job: {result.stderr}")
            return False
    
    def list_jobs(self):
        """List all bimanual_rl jobs."""
        result = subprocess.run(
            ['squeue', '--me', '--name=bimanual*'],
            capture_output=True,
            text=True,
        )
        
        if result.returncode == 0:
            logger.info("Active bimanual training jobs:")
            print(result.stdout)
        else:
            logger.warning("No active jobs found")
    
    def get_best_checkpoint(self, log_dir):
        """Find the best checkpoint in a training run."""
        log_path = Path(log_dir)
        checkpoint_dir = log_path / "checkpoints"
        
        if not checkpoint_dir.exists():
            logger.warning(f"Checkpoint directory not found: {checkpoint_dir}")
            return None
        
        best_checkpoint = checkpoint_dir / "checkpoint_best.pt"
        if best_checkpoint.exists():
            logger.info(f"Best checkpoint found: {best_checkpoint}")
            return str(best_checkpoint)
        else:
            logger.warning("No best checkpoint found")
            return None
    
    def resume_training(self, checkpoint_path, config_file=None):
        """Resume training from checkpoint."""
        if not Path(checkpoint_path).exists():
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            return None
        
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        
        config_file = config_file or 'config.yaml'
        
        return self.submit_job(
            config_file=config_file,
            job_name=f"bimanual_resume_{datetime.now().timestamp()}",
        )


def main():
    """CLI for cluster management."""
    parser = argparse.ArgumentParser(description="GPU Cluster Training Manager")
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Submit command
    submit_parser = subparsers.add_parser('submit', help='Submit training job')
    submit_parser.add_argument('--config', type=str, default='config.yaml')
    submit_parser.add_argument('--num-envs', type=int, default=4)
    submit_parser.add_argument('--total-timesteps', type=int, default=1_000_000)
    submit_parser.add_argument('--gpu', type=str, default='a100', choices=['a100', 'h100'])
    submit_parser.add_argument('--time', type=str, default='24:00:00')
    submit_parser.add_argument('--name', type=str, default='bimanual_rl')
    submit_parser.add_argument('--array', type=int, default=None)
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Check job status')
    status_parser.add_argument('job_id', type=str)
    
    # Cancel command
    cancel_parser = subparsers.add_parser('cancel', help='Cancel job')
    cancel_parser.add_argument('job_id', type=str)
    
    # List command
    list_parser = subparsers.add_parser('list', help='List active jobs')
    
    # Resume command
    resume_parser = subparsers.add_parser('resume', help='Resume from checkpoint')
    resume_parser.add_argument('checkpoint', type=str)
    resume_parser.add_argument('--config', type=str, default='config.yaml')
    
    args = parser.parse_args()
    
    manager = ClusterManager()
    
    if args.command == 'submit':
        manager.submit_job(
            config_file=args.config,
            num_envs=args.num_envs,
            total_timesteps=args.total_timesteps,
            gpu_type=args.gpu,
            time_limit=args.time,
            job_name=args.name,
            array_size=args.array,
        )
    
    elif args.command == 'status':
        manager.check_job_status(args.job_id)
    
    elif args.command == 'cancel':
        manager.cancel_job(args.job_id)
    
    elif args.command == 'list':
        manager.list_jobs()
    
    elif args.command == 'resume':
        manager.resume_training(args.checkpoint, args.config)
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
