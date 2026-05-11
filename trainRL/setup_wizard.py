#!/usr/bin/env python3
"""
Quick setup and troubleshooting guide for bimanual RL training.

This script provides:
1. Dependency verification
2. Environment setup checks
3. Quick-start training commands
4. Common troubleshooting
"""

import subprocess
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check if all required packages are installed."""
    logger.info("Checking dependencies...")
    logger.info("-" * 60)
    
    packages = {
        'torch': 'PyTorch',
        'gymnasium': 'Gymnasium',
        'stable_baselines3': 'Stable-Baselines3',
        'imageio': 'imageio',
        'cv2': 'OpenCV',
        'tensorboard': 'TensorBoard',
        'yaml': 'PyYAML',
        'numpy': 'NumPy',
    }
    
    missing = []
    for module, name in packages.items():
        try:
            __import__(module)
            logger.info(f"✓ {name}")
        except ImportError:
            logger.error(f"✗ {name} - MISSING")
            missing.append(module)
    
    if missing:
        logger.error(f"\nMissing packages: {', '.join(missing)}")
        logger.info("Install with: pip install -r requirements.txt")
        return False
    
    logger.info("✓ All dependencies found\n")
    return True


def check_isaaclab():
    """Check if IsaacLab environment is set up."""
    logger.info("Checking IsaacLab setup...")
    logger.info("-" * 60)
    
    try:
        import isaaclab
        logger.info(f"✓ IsaacLab imported successfully")
        
        from isaaclab.envs import ManagerBasedRLEnv
        logger.info("✓ IsaacLab RL environment available")
        
        logger.info("✓ IsaacLab setup verified\n")
        return True
        
    except ImportError as e:
        logger.error(f"✗ IsaacLab not found: {e}")
        logger.info("IsaacLab must be installed via Isaac SDK")
        logger.info("See: https://docs.omniverse.nvidia.com/isaacsim/latest/installation/index.html\n")
        return False


def check_directories():
    """Check if required directories exist."""
    logger.info("Checking required directories...")
    logger.info("-" * 60)
    
    base_dir = Path(__file__).parent
    required_dirs = [
        'bimanual_vx300s_env',
        'bimanual_vx300s_env/mdp',
        'logs',
        'logs/checkpoints',
        'logs/videos',
        'logs/tensorboard',
    ]
    
    for dir_name in required_dirs:
        dir_path = base_dir / dir_name
        if dir_path.exists():
            logger.info(f"✓ {dir_name}/")
        else:
            logger.warning(f"⚠ {dir_name}/ (will be created)")
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"✓ Created {dir_name}/")
            except Exception as e:
                logger.error(f"✗ Failed to create {dir_name}/: {e}")
                return False
    
    logger.info("✓ All directories ready\n")
    return True


def print_quick_start():
    """Print quick start commands."""
    logger.info("=" * 60)
    logger.info("QUICK START GUIDE")
    logger.info("=" * 60)
    
    commands = [
        ("Run integration tests", "python integration_test.py"),
        ("Train locally (4 envs, 100k steps)", "python train_advanced.py --num-envs 4 --total-timesteps 100000"),
        ("Train on GPU cluster", "sbatch submit_training.sh --num-envs 4"),
        ("Resume from checkpoint", "python train_advanced.py --resume-from-checkpoint logs/checkpoints/checkpoint_latest.pt"),
        ("Monitor training", "python training_monitor.py summary --log-dir logs"),
        ("Check job status", "python cluster_manager.py status {JOB_ID}"),
        ("View TensorBoard", "python training_monitor.py tensorboard --log-dir logs"),
    ]
    
    for i, (desc, cmd) in enumerate(commands, 1):
        logger.info(f"\n{i}. {desc}")
        logger.info(f"   $ {cmd}")
    
    logger.info("\n")


def print_troubleshooting():
    """Print troubleshooting guide."""
    logger.info("=" * 60)
    logger.info("TROUBLESHOOTING")
    logger.info("=" * 60)
    
    issues = [
        {
            'problem': "IsaacLab import error",
            'solution': "Activate isaac_fresh conda environment:\n"
                       "  source ~/yash/miniforge3/bin/activate\n"
                       "  conda activate isaac_fresh",
        },
        {
            'problem': "SLURM job submission fails (sbatch: command not found)",
            'solution': "SLURM not available on local machine.\n"
                       "Use sbatch command only on GPU cluster.",
        },
        {
            'problem': "Training crashes with GPU out of memory",
            'solution': "Reduce number of parallel environments:\n"
                       "  python train_advanced.py --num-envs 2",
        },
        {
            'problem': "No checkpoints being saved",
            'solution': "Check write permissions in logs/ directory:\n"
                       "  ls -la logs/\n"
                       "  chmod -R u+w logs/",
        },
        {
            'problem': "Video recording not working",
            'solution': "Install imageio and opencv:\n"
                       "  pip install imageio opencv-python",
        },
        {
            'problem': "Training killed before saving checkpoint",
            'solution': "Signal handlers may not have time to run.\n"
                       "Use graceful shutdown:\n"
                       "  kill -TERM {PROCESS_PID}  (gives 10 second timeout)\n"
                       "  kill -KILL {PROCESS_PID}  (immediate, no save)",
        },
    ]
    
    for issue in issues:
        logger.info(f"\n❌ {issue['problem']}")
        logger.info(f"   Solution: {issue['solution']}")
    
    logger.info("\n")


def print_architecture():
    """Print system architecture."""
    logger.info("=" * 60)
    logger.info("SYSTEM ARCHITECTURE")
    logger.info("=" * 60)
    
    logger.info("""
┌─ Environment ──────────────────────────────────────┐
│  - Scene: 2 VX300s arms + table + cube + cameras   │
│  - Physics: PhysX 5 on GPU (10 Hz, 5 substeps)     │
│  - Num Environments: Parallel simulation on GPU     │
└────────────────────────────────────────────────────┘
                        ↓
┌─ Observations ────────────────────────────────────┐
│  - Proprioceptive: 28 dims (joint/gripper state)   │
│  - Optional: RGB images from 4 cameras             │
│  - Normalized via VecNormalize wrapper             │
└────────────────────────────────────────────────────┘
                        ↓
┌─ Policy ──────────────────────────────────────────┐
│  - Algorithm: PPO (Proximal Policy Optimization)   │
│  - Network: 2x256 hidden layers                    │
│  - Framework: Stable-Baselines3                    │
└────────────────────────────────────────────────────┘
                        ↓
┌─ Rewards (Multi-Component) ───────────────────────┐
│  - 40% Distance: exp(-5*dist) as arms approach     │
│  - 30% Grasp: stability penalty if grip loosens    │
│  - 20% Lifting: bonus for raising object          │
│  - 10% Efficiency: penalty for jerky movements     │
└────────────────────────────────────────────────────┘
                        ↓
┌─ Termination Conditions ──────────────────────────┐
│  - Arm-table collision: safety constraint          │
│  - Out of workspace: joint limits exceeded         │
│  - Time limit: max episode length                  │
│  - Success: object lifted and held                 │
└────────────────────────────────────────────────────┘
                        ↓
┌─ Training Pipeline ───────────────────────────────┐
│  ├─ Checkpoint Manager: save/resume state         │
│  ├─ Video Recorder: capture episodes              │
│  ├─ Metrics Logger: reward/success tracking       │
│  └─ Signal Handlers: graceful shutdown            │
└────────────────────────────────────────────────────┘
                        ↓
┌─ Outputs ─────────────────────────────────────────┐
│  - Trained model: logs/checkpoints/               │
│  - Episode videos: logs/videos/                   │
│  - Metric plots: logs/plots/                      │
│  - TensorBoard: logs/tensorboard/                 │
└────────────────────────────────────────────────────┘
""")


def print_file_structure():
    """Print project file structure."""
    logger.info("=" * 60)
    logger.info("PROJECT FILE STRUCTURE")
    logger.info("=" * 60)
    
    logger.info("""
trainRL/
├── bimanual_vx300s_env/         # Main environment package
│   ├── __init__.py
│   ├── vx300s_env_cfg.py        # Scene configuration
│   └── mdp/
│       ├── actions.py           # Control interface (14 dims)
│       ├── observations.py      # Sensor reading (28 dims)
│       ├── rewards.py           # Multi-component rewards
│       └── terminations.py      # Episode end conditions
│
├── train_advanced.py            # Main training script (all features)
├── checkpoint_manager.py        # Save/load/resume system
├── video_recorder.py            # Video and metrics recording
├── cluster_manager.py           # SLURM job management
├── training_monitor.py          # Training monitoring tools
├── integration_test.py          # Feature validation
├── submit_training.sh           # SLURM sbatch script
│
├── config.yaml                  # Training configuration
├── requirements.txt             # Python dependencies
└── logs/                        # Training outputs
    ├── checkpoints/             # Saved models
    ├── videos/                  # Episode recordings
    ├── plots/                   # Metric visualizations
    ├── tensorboard/             # TensorBoard logs
    └── slurm_*.log             # Job output logs
""")


def main():
    """Run setup wizard."""
    logger.info("\n")
    logger.info("╔" + "=" * 58 + "╗")
    logger.info("║  BIMANUAL RL - SETUP & TROUBLESHOOTING WIZARD             ║")
    logger.info("╚" + "=" * 58 + "╝")
    logger.info("\n")
    
    # Check everything
    all_good = True
    
    all_good &= check_dependencies()
    all_good &= check_isaaclab()
    all_good &= check_directories()
    
    if not all_good:
        logger.warning("Some checks failed. Fix above issues before training.")
        return 1
    
    # Print guides
    print_file_structure()
    print_architecture()
    print_quick_start()
    print_troubleshooting()
    
    logger.info("=" * 60)
    logger.info("✓ Setup complete! Ready to start training.")
    logger.info("=" * 60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
