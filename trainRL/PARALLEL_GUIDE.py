#!/usr/bin/env python3
"""
╔════════════════════════════════════════════════════════════════════════════╗
║                  PARALLEL GPU TRAINING - QUICK START GUIDE                ║
║                                                                            ║
║  This guide explains how to run multiple training instances in parallel   ║
║  on your GPU cluster using the bash scripts provided.                    ║
╚════════════════════════════════════════════════════════════════════════════╝
"""

GUIDE = r"""
═════════════════════════════════════════════════════════════════════════════
                            QUICK START (5 MINUTES)
═════════════════════════════════════════════════════════════════════════════

1. SUBMIT PARALLEL TRAINING JOB:
   ─────────────────────────────
   
   cd ~/yash/trainRL
   
   # Option A: Submit with sbatch (recommended for cluster)
   sbatch launch_parallel_training.sh
   
   # Option B: Run locally for testing
   bash launch_parallel_training.sh
   
   # Option C: Custom parameters
   sbatch launch_parallel_training.sh --num-instances 8 --timesteps 5000000
   
   Expected output:
   ──────────────
   Submitted batch job 12345
   
   Job submitted successfully!
   Check status with: python cluster_manager.py status 12345

2. MONITOR TRAINING IN REAL-TIME:
   ────────────────────────────────
   
   # Option A: Quick status snapshot
   bash monitor_parallel.sh status
   
   # Option B: Real-time watch mode (updates every 10 seconds)
   bash monitor_parallel.sh watch
   
   # Option C: Check specific job
   squeue -l -j 12345
   
   Expected output:
   ───────────────
   📊 PARALLEL INSTANCES STATUS
   ══════════════════════════════════
   
   Instance: instance_1
     Latest output: Step 5000 | Reward: 125.3
     Videos recorded: 12
     Checkpoints saved: 5
   
   Instance: instance_2
     Latest output: Step 4800 | Reward: 118.9
     Videos recorded: 11
     Checkpoints saved: 4
   
   Total instances: 4

3. COLLECT RESULTS AFTER TRAINING:
   ────────────────────────────────
   
   # Aggregate all videos, plots, and checkpoints
   bash monitor_parallel.sh collect
   
   # Generate completion report
   bash monitor_parallel.sh report
   
   # View aggregated results
   ls -lh logs/videos_aggregate/        # All episode videos
   ls -lh logs/plots_aggregate/         # All metric plots
   ls -lh logs/checkpoints_aggregate/   # All best models
   
   Expected output:
   ───────────────
   ✓ Copied 45 videos from instance_1
   ✓ Copied 43 videos from instance_2
   ✓ Copied 44 videos from instance_3
   ✓ Copied 42 videos from instance_4
   
   Summary:
     Total videos: 174
     Total plots: 174
     Total checkpoints: 4
     Aggregate dir: logs/videos_aggregate/


═════════════════════════════════════════════════════════════════════════════
                        PARAMETER CUSTOMIZATION
═════════════════════════════════════════════════════════════════════════════

SBATCH Headers (Edit launch_parallel_training.sh):
─────────────────────────────────────────────────

#SBATCH --gres=gpu:4              # Number of GPUs (1 per instance)
#SBATCH --time=24:00:00           # Maximum runtime (hours:minutes:seconds)
#SBATCH --cpus-per-task=32        # Total CPU cores
#SBATCH --mem=256G                # Total memory
#SBATCH --partition=gpu-a100      # GPU type: gpu-a100 or gpu-h100
#SBATCH --array=1-4%4             # Job array: create 4 jobs, max 4 running

Training Parameters (Command line):
───────────────────────────────────

NUM_INSTANCES=4          # Number of parallel training instances
NUM_ENVS=4               # Parallel environments per instance
TOTAL_TIMESTEPS=1000000  # Training steps per instance
CONFIG_NAME=config.yaml  # Configuration file

Examples:
─────────

# Light training (2 instances, quick test)
sbatch launch_parallel_training.sh 2 2 100000

# Medium training (4 instances, 1M steps each)
sbatch launch_parallel_training.sh 4 4 1000000

# Heavy training (8 instances, 5M steps each)
sbatch launch_parallel_training.sh 8 4 5000000


═════════════════════════════════════════════════════════════════════════════
                           HOW IT WORKS UNDER THE HOOD
═════════════════════════════════════════════════════════════════════════════

1. SBATCH SUBMISSION:
   ─────────────────
   
   When you run: sbatch launch_parallel_training.sh
   
   SLURM creates a job array with 4 tasks (default), each on a separate GPU:
   
   Job 12345 [array of 4]:
   ├─ Task 1 → GPU 0 → Instance 1 training in logs/parallel_runs/instance_1/
   ├─ Task 2 → GPU 1 → Instance 2 training in logs/parallel_runs/instance_2/
   ├─ Task 3 → GPU 2 → Instance 3 training in logs/parallel_runs/instance_3/
   └─ Task 4 → GPU 3 → Instance 4 training in logs/parallel_runs/instance_4/

2. PER-INSTANCE EXECUTION:
   ─────────────────────────
   
   Each instance:
   ├─ Gets unique GPU (determined by task ID)
   ├─ Creates isolated environment (no conflicts)
   ├─ Saves to separate directory (instance_1/, instance_2/, etc.)
   ├─ Runs independent training loop
   ├─ Records videos locally
   └─ Saves checkpoints independently
   
   Instance directories:
   logs/parallel_runs/
   ├─ instance_1/
   │  ├─ training.log      (training output)
   │  ├─ train_instance.py (instance script)
   │  ├─ videos/           (episode recordings)
   │  ├─ plots/            (metric visualizations)
   │  └─ checkpoints/      (saved models)
   ├─ instance_2/
   │  ...
   └─ instance_4/

3. RESULT AGGREGATION:
   ────────────────────
   
   After training completes, all videos/plots/checkpoints are collected:
   
   logs/
   ├─ videos_aggregate/           ← All 174 videos from all instances
   ├─ plots_aggregate/            ← All 174 plots from all instances
   ├─ checkpoints_aggregate/      ← All best models from each instance
   ├─ parallel_runs/
   │  ├─ instance_1/              ← Instance 1 original files
   │  ├─ instance_2/              ← Instance 2 original files
   │  ...


═════════════════════════════════════════════════════════════════════════════
                            GPU MEMORY MANAGEMENT
═════════════════════════════════════════════════════════════════════════════

Each Instance GPU Usage:
────────────────────────

Per-instance configuration determines memory:

NUM_ENVS=2  →  ~8 GB  per GPU     (good for testing)
NUM_ENVS=4  →  ~12 GB per GPU     (default, balanced)
NUM_ENVS=8  →  ~20 GB per GPU     (large batch, A100 only)
NUM_ENVS=16 →  ~35 GB per GPU     (A100 only, experimental)

A100 GPU: 80GB → can run NUM_ENVS=16
H100 GPU: 96GB → can run NUM_ENVS=16-20

Example Setup:
──────────────

4 instances on A100 with 80GB each:

Configuration 1 (Conservative):
  NUM_ENVS=4
  Total GPU usage: 4 instances × 12 GB = 48 GB
  Remaining: 32 GB per GPU for OS/overhead
  
Configuration 2 (Aggressive):
  NUM_ENVS=8
  Total GPU usage: 4 instances × 20 GB = 80 GB
  Remaining: 0 GB (risky, may OOM)

Recommendation:
───────────────
4 instances × 4 environments = ~12 GB each = safe on any modern GPU


═════════════════════════════════════════════════════════════════════════════
                          MONITORING COMMANDS
═════════════════════════════════════════════════════════════════════════════

REAL-TIME WATCHING:
───────────────────

# Real-time watch mode (updates every 10 seconds)
bash monitor_parallel.sh watch

Output:
───────
📊 PARALLEL TRAINING MONITOR [Mon May  5 14:32:15 2026]
═════════════════════════════════════════

Instance: instance_1
  Steps: 45000
  Videos: 15
  Checkpoints: 4

Instance: instance_2
  Steps: 42500
  Videos: 14
  Checkpoints: 4

Instance: instance_3
  Steps: 48000
  Videos: 16
  Checkpoints: 5

Instance: instance_4
  Steps: 41200
  Videos: 13
  Checkpoints: 4


SLURM JOB TRACKING:
───────────────────

# Check specific job
squeue -l -j 12345

# Check all your jobs
squeue -l --me

# Show detailed job info
scontrol show job 12345

# Cancel a job
scancel 12345

# Show completed job stats
sacct -j 12345 --format=JobID,JobName,State,Elapsed,MaxRSS


VIEWING RESULTS:
────────────────

# Count files
ls -1 logs/videos_aggregate/ | wc -l       # Total videos
ls -1 logs/plots_aggregate/ | wc -l        # Total plots
ls -1 logs/checkpoints_aggregate/ | wc -l  # Total checkpoints

# View specific results
ls -lh logs/videos_aggregate/ | head -20   # Latest videos
ls -lh logs/plots_aggregate/ | tail -20    # Most recent plots

# Find best checkpoint
ls -lh logs/checkpoints_aggregate/ | sort -k 5 -r | head -1


═════════════════════════════════════════════════════════════════════════════
                             TROUBLESHOOTING
═════════════════════════════════════════════════════════════════════════════

PROBLEM: "sbatch: command not found"
─────────────────────────────────────
This means SLURM is not available (probably running locally).

Solution: Run with bash instead
  bash launch_parallel_training.sh

Or submit to cluster from login node:
  ssh gpu-cluster.example.com
  cd ~/yash/trainRL
  sbatch launch_parallel_training.sh


PROBLEM: Job fails immediately with CUDA out of memory
────────────────────────────────────────────────────────
Too many environments per GPU.

Solution: Reduce NUM_ENVS
  sbatch launch_parallel_training.sh 4 2 1000000
  
  (4 instances, 2 environments each)


PROBLEM: No videos are being recorded
────────────────────────────────────────
Check if imageio is installed:
  pip list | grep imageio
  
Install if missing:
  pip install imageio opencv-python


PROBLEM: Training very slow on first steps
───────────────────────────────────────────
This is normal! Isaac engine initialization takes time.
After 5-10 minutes, speed should normalize.

Check progress:
  tail -f logs/parallel_runs/instance_1/training.log


PROBLEM: One instance crashed, others still running
──────────────────────────────────────────────────────
SLURM job array handles this automatically.

Check which failed:
  squeue -l -j 12345
  
Failed task will show state=FAILED
Other tasks continue running independently.

If need to restart failed instance:
  sbatch --dependency=afterok:12345 launch_parallel_training.sh


PROBLEM: Results not aggregating
─────────────────────────────────
Run manual aggregation:
  bash monitor_parallel.sh collect


═════════════════════════════════════════════════════════════════════════════
                          ADVANCED USAGE PATTERNS
═════════════════════════════════════════════════════════════════════════════

PATTERN 1: CHAIN MULTIPLE TRAINING RUNS
────────────────────────────────────────

Run first 1M steps, then continue for another 1M:

# First job
JOB1=$(sbatch launch_parallel_training.sh 4 4 1000000 | grep -o '[0-9]*')

# Second job (starts after first completes)
sbatch --dependency=afterok:$JOB1 launch_parallel_training.sh 4 4 1000000

# Results automatically aggregated in logs/videos_aggregate/, etc.


PATTERN 2: RESUME FROM CHECKPOINTS
───────────────────────────────────

After parallel training:

1. Collect best checkpoints
   bash monitor_parallel.sh collect

2. Find best model
   best_model=$(ls -t logs/checkpoints_aggregate/*.pt | head -1)

3. Resume training (single GPU)
   python train_advanced.py --resume-from-checkpoint $best_model


PATTERN 3: GRID SEARCH (Different Hyperparameters)
───────────────────────────────────────────────────

Train with 4 different configurations in parallel:

# Instance 1: Small learning rate
sbatch launch_parallel_training.sh 1 4 500000

# Instance 2: Large learning rate
sbatch launch_parallel_training.sh 1 4 500000

# Instance 3: Different reward weights
sbatch launch_parallel_training.sh 1 4 500000

# Instance 4: Different network size
sbatch launch_parallel_training.sh 1 4 500000

Compare results to find best configuration.


PATTERN 4: DISTRIBUTED EVALUATION
──────────────────────────────────

Train 4 models in parallel, then evaluate all:

# Train
sbatch launch_parallel_training.sh 4 4 1000000

# Wait for completion
watch -n 60 'squeue -l -j 12345'

# Collect checkpoints
bash monitor_parallel.sh collect

# Evaluate each model
for ckpt in logs/checkpoints_aggregate/*.pt; do
  python evaluate.py --model $ckpt
done

# Compare evaluation results


═════════════════════════════════════════════════════════════════════════════
                            FILE STRUCTURE
═════════════════════════════════════════════════════════════════════════════

trainRL/
├── launch_parallel_training.sh          ← Main parallel launcher
├── monitor_parallel.sh                  ← Monitoring & aggregation
├── train_advanced.py                    ← Single-instance trainer
├── ...other training files...
│
└── logs/
    ├── parallel_runs/                   ← All parallel instances
    │   ├── instance_1/
    │   │   ├── training.log
    │   │   ├── train_instance.py
    │   │   ├── videos/
    │   │   ├── plots/
    │   │   └── checkpoints/
    │   ├── instance_2/
    │   ├── instance_3/
    │   └── instance_4/
    │
    ├── videos_aggregate/                ← Collected from all instances
    │   ├── episode_0.mp4 (instance_1)
    │   ├── episode_50.mp4 (instance_1)
    │   ├── episode_0.mp4 (instance_2)
    │   └── ...
    │
    ├── plots_aggregate/                 ← Collected plots
    ├── checkpoints_aggregate/           ← Collected best models
    │
    └── parallel_*.log                   ← SLURM job logs


═════════════════════════════════════════════════════════════════════════════
                            COMPLETE WORKFLOW
═════════════════════════════════════════════════════════════════════════════

Step 1: PREPARE
─────────────
cd ~/yash/trainRL
python setup_wizard.py          # Verify all dependencies
python integration_test.py      # Test system components

Step 2: SUBMIT PARALLEL JOB
──────────────────────────
sbatch launch_parallel_training.sh

Output: Submitted batch job 12345

Step 3: MONITOR IN REAL-TIME
─────────────────────────────
bash monitor_parallel.sh watch

(Watch updates every 10 seconds, Ctrl+C to exit)

Step 4: WAIT FOR COMPLETION
───────────────────────────
squeue -l                       # Check if job still running
tail -f logs/parallel_*.log     # Follow main log

Step 5: AGGREGATE RESULTS
────────────────────────
bash monitor_parallel.sh collect      # Gather all files
bash monitor_parallel.sh report       # Generate report

Step 6: ANALYZE RESULTS
──────────────────────
ls -lh logs/videos_aggregate/   # View all 174+ videos
python training_monitor.py summary --log-dir logs  # Summary stats


═════════════════════════════════════════════════════════════════════════════
                              KEY COMMANDS
═════════════════════════════════════════════════════════════════════════════

SUBMITTING:
  sbatch launch_parallel_training.sh                    # Standard 4 instances
  sbatch launch_parallel_training.sh 8 4 5000000        # 8 instances, 5M steps

MONITORING:
  bash monitor_parallel.sh status                       # Quick status
  bash monitor_parallel.sh watch                        # Real-time watch
  squeue -l                                             # SLURM queue
  tail -f logs/parallel_runs/instance_1/training.log    # Live instance log

RESULTS:
  bash monitor_parallel.sh collect                      # Aggregate results
  bash monitor_parallel.sh report                       # Generate report
  ls -lh logs/videos_aggregate/                         # All videos
  ls -lh logs/checkpoints_aggregate/                    # All models

MANAGEMENT:
  scancel 12345                                         # Cancel job
  bash monitor_parallel.sh cleanup                      # Remove old instances


═════════════════════════════════════════════════════════════════════════════
                        SYSTEM STATUS: ✓ READY FOR DEPLOYMENT
═════════════════════════════════════════════════════════════════════════════

The parallel training system is complete and ready for use on GPU clusters.

Next step: sbatch launch_parallel_training.sh

═════════════════════════════════════════════════════════════════════════════
"""

def main():
    print(GUIDE)

if __name__ == '__main__':
    main()
