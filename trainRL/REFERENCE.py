#!/usr/bin/env python3
"""
REFERENCE GUIDE - Bimanual RL Training System

This guide explains all features and usage patterns for the complete
training system optimized for GPU clusters.

SECTIONS:
1. Feature Overview
2. Usage Patterns
3. Configuration Guide
4. Cluster Deployment
5. Checkpoint Management
6. Monitoring & Debugging
7. Advanced Topics
"""

GUIDE = """
╔══════════════════════════════════════════════════════════════════════════════╗
║              BIMANUAL RL TRAINING SYSTEM - COMPLETE REFERENCE GUIDE          ║
╚══════════════════════════════════════════════════════════════════════════════╝


═══════════════════════════════════════════════════════════════════════════════
1. FEATURE OVERVIEW
═══════════════════════════════════════════════════════════════════════════════

This system provides a production-grade reinforcement learning training setup for
a bimanual robot arm manipulation task. Key features:

✓ REWARD SHAPING (Multi-Component)
  ├─ Distance-based: Exponential decay reward as gripper approaches object
  ├─ Grasp stability: Penalty if object is dropped after initial grasp
  ├─ Lifting bonus: +100 reward if object lifted above table surface
  └─ Efficiency penalty: Discourages jerky/excessive gripper movements

✓ COLLISION PREVENTION
  ├─ Arm-table collision detection with immediate episode termination
  ├─ Workspace boundary enforcement (joint limits)
  └─ Contact force monitoring and thresholding

✓ PERSISTENT TRAINING STATE
  ├─ Automatic checkpoint saving every 1000 steps
  ├─ Full state preservation: model weights, optimizer, metrics, step count
  ├─ Resume capability: continue from any checkpoint
  └─ Graceful shutdown: Ctrl+C triggers checkpoint save before exit

✓ TRAINING VISUALIZATION
  ├─ Episode video recording (MP4 format, every 50 episodes)
  ├─ Metric plots: reward curves, success rates, distance to object
  ├─ TensorBoard integration for real-time monitoring
  └─ Remote monitoring: watch training progress from cluster login node

✓ GPU CLUSTER OPTIMIZATION
  ├─ SLURM sbatch job submission
  ├─ A100/H100 GPU support with optimal environment variables
  ├─ Headless mode (no X11 display required)
  └─ Job monitoring and log aggregation

✓ MULTI-ENVIRONMENT PARALLELIZATION
  ├─ Run multiple environments simultaneously on single GPU
  ├─ Configurable: 2-16 parallel environments (default: 4)
  ├─ Observation normalization across environment batch
  └─ Vectorized reward computation


═══════════════════════════════════════════════════════════════════════════════
2. USAGE PATTERNS
═══════════════════════════════════════════════════════════════════════════════

PATTERN 1: LOCAL TESTING
─────────────────────────

Run integration tests to verify setup:
  $ python setup_wizard.py           # Check dependencies and directories
  $ python integration_test.py       # Test all components work

Quick training run (for debugging):
  $ python train_advanced.py --num-envs 2 --total-timesteps 10000
  
This will:
  - Create 2 parallel environments
  - Train for 10k timesteps (~2-5 minutes)
  - Save checkpoint every 1000 steps → logs/checkpoints/
  - Record videos every 50 episodes → logs/videos/
  - Logs saved to TensorBoard → logs/tensorboard/


PATTERN 2: GPU CLUSTER SUBMISSION
──────────────────────────────────

Method 1 - Using cluster manager (recommended):
  $ python cluster_manager.py submit \\
      --config config.yaml \\
      --num-envs 4 \\
      --total-timesteps 1000000 \\
      --gpu a100 \\
      --time 24:00:00

Method 2 - Using sbatch directly:
  $ sbatch submit_training.sh --num-envs 4 --total-timesteps 1000000
  
This will:
  - Queue job on gpu-a100 partition
  - Allocate 1 GPU, 8 CPU cores, 64GB RAM
  - Run for up to 24 hours
  - Print job ID for monitoring: Job submitted successfully! Job ID: 12345


PATTERN 3: RESUME FROM CHECKPOINT
──────────────────────────────────

After job completes or is interrupted:
  $ python train_advanced.py --resume-from-checkpoint logs/checkpoints/checkpoint_latest.pt

Or specify exact checkpoint:
  $ python train_advanced.py --resume-from-checkpoint logs/checkpoints/checkpoint_1000_12345.pt

This will:
  - Load all training state from checkpoint
  - Continue from exact step where training stopped
  - Preserve optimizer state (momentum, etc.)
  - Append to same log files


PATTERN 4: MONITORING ACTIVE JOB
────────────────────────────────

Real-time job monitoring:
  $ python cluster_manager.py status 12345
  
Get full training summary:
  $ python training_monitor.py summary --log-dir logs
  
List recent videos and metrics:
  $ python training_monitor.py videos --log-dir logs
  $ python training_monitor.py metrics --log-dir logs
  
Monitor with TensorBoard (on login node):
  $ python training_monitor.py tensorboard --log-dir logs
  
Then open browser to: http://localhost:6006


PATTERN 5: BATCH JOB SUBMISSION
────────────────────────────────

Submit multiple training runs with different configs:
  $ for config in config_small.yaml config_medium.yaml config_large.yaml; do
      python cluster_manager.py submit --config $config
    done

Submit job array (SLURM will distribute across queue):
  $ python cluster_manager.py submit \\
      --config config.yaml \\
      --array 3  # Will create 3 simultaneous jobs

Monitor all jobs:
  $ squeue --me --name=bimanual*


═══════════════════════════════════════════════════════════════════════════════
3. CONFIGURATION GUIDE
═══════════════════════════════════════════════════════════════════════════════

REWARD CONFIGURATION
────────────────────

Default multi-component reward (in bimanual_vx300s_env/mdp/rewards.py):

  distance_reward(ee_pos, object_pos):
    # Exponential decay: higher reward when gripper closer to object
    distance = ||ee_pos - object_pos||_2
    reward = exp(-alpha * distance)
    
    # Default: alpha=5.0, gives 50% reward at 0.3m distance
    # Tune by modifying: distance_scale parameter

  grasp_stability_reward(gripper_force, prev_gripper_force):
    # Penalty if object is dropped after successful grasp
    if gripper_force > threshold and prev_gripper_force < threshold:
        reward = 0  # Normal grasping
    elif gripper_force < threshold and prev_gripper_force > threshold:
        reward = -50  # Object dropped!
    
    # Tune by modifying: drop_penalty parameter

  lifting_reward(object_height, table_height):
    # Bonus if object successfully lifted above table
    if object_height > table_height + 0.05:  # 5cm above table
        reward = +100
    
    # Tune by modifying: lift_threshold and lift_bonus

ENVIRONMENT CONFIGURATION
──────────────────────────

In train_advanced.py, modify:

  # Number of parallel environments (GPU memory trade-off)
  args.num_envs = 4  # Range: 1-16 depending on GPU
  
  # Total training timesteps
  args.total_timesteps = 1_000_000  # 1M default
  
  # Checkpoint save interval
  checkpoint_interval = 1000  # Save every 1000 steps
  
  # Video recording interval
  record_interval = 50  # Record every 50 episodes

CLUSTER CONFIGURATION
─────────────────────

In submit_training.sh, modify SBATCH headers:

  #SBATCH --partition=gpu-a100    # or gpu-h100
  #SBATCH --gres=gpu:1            # Number of GPUs (usually 1)
  #SBATCH --time=24:00:00         # Wall clock time
  #SBATCH --cpus-per-task=8       # CPU cores allocated
  #SBATCH --mem=64G               # RAM per task


═══════════════════════════════════════════════════════════════════════════════
4. CLUSTER DEPLOYMENT
═══════════════════════════════════════════════════════════════════════════════

SUBMIT JOB VIA SBATCH
─────────────────────

1. Prepare configuration:
   Edit config.yaml with desired hyperparameters

2. Submit job:
   $ sbatch submit_training.sh

3. Check submission:
   $ squeue -l -j {JOB_ID}

4. Monitor logs:
   $ tail -f logs/slurm_{JOB_ID}.log

SBATCH OUTPUT INTERPRETATION
────────────────────────────

When you run: sbatch submit_training.sh

You'll see:  Submitted batch job 12345
             ^^^^^^^^^^^^^^^^^^^^^^^ Job ID for monitoring

Check job status:
  $ squeue -l -j 12345
  $ scontrol show job 12345

If job is pending (in queue):
  ST=PD  TimeLimit=1-00:00:00  Reason=QueueGrp  ... PENDING

If job is running:
  ST=R   TimeLimit=1-00:00:00  ... RUNNING

If job completed/failed:
  $ squeue -j 12345  (no output means finished)
  $ sacct -j 12345   (shows final status)

CANCEL/STOP JOB
───────────────

Graceful stop (allows checkpoint save):
  $ kill -TERM {PROCESS_PID}  # or Ctrl+C in direct terminal
  
Force stop (immediate, may lose work):
  $ scancel 12345             # Cancel SLURM job
  $ kill -9 {PROCESS_PID}     # Force kill process

After graceful stop, check logs:
  $ tail logs/slurm_12345.log
  
You should see:
  "Saving checkpoint before exit..."
  "Checkpoint saved to: logs/checkpoints/checkpoint_..."


═══════════════════════════════════════════════════════════════════════════════
5. CHECKPOINT MANAGEMENT
═══════════════════════════════════════════════════════════════════════════════

CHECKPOINT SAVING
─────────────────

Checkpoints are saved automatically:
  - Every N steps (default: 1000 steps)
  - When training is gracefully stopped (Ctrl+C or SIGTERM)
  - On error (if crash protection is enabled)

Checkpoint files:
  logs/checkpoints/
  ├── checkpoint_1000_TIMESTAMP.pt      # Model + optimizer state
  ├── checkpoint_1000_TIMESTAMP.json    # Metadata
  ├── checkpoint_2000_TIMESTAMP.pt
  ├── checkpoint_2000_TIMESTAMP.json
  └── checkpoint_best.pt                # Best performing model

Each .json file contains:
  {
    "episode": 50,
    "step": 1000,
    "timestamp": "2024-01-15 14:32:00",
    "metrics": {
      "best_reward": 450.5,
      "avg_episode_length": 200,
      "success_rate": 0.75
    }
  }

CHECKPOINT LOADING
──────────────────

Get latest checkpoint (recommended):
  checkpoint_path = checkpoint_manager.get_latest_checkpoint()

Load specific checkpoint:
  training_state = checkpoint_manager.load_checkpoint("logs/checkpoints/checkpoint_1000_timestamp.pt")
  
  training_state contains:
    - model: PPO policy network
    - optimizer: Adam optimizer with state
    - metrics: dictionary of training metrics
    - episode: episode number when saved
    - step: total steps completed

RESUME TRAINING
───────────────

1. Identify checkpoint to resume from:
   $ python training_monitor.py checkpoints --log-dir logs
   
   Output:
     Most recent checkpoints:
       1. checkpoint_2000_2024-01-15_14-32-45.pt
          Size: 120.5 MB
          Episode: 100
          Best Reward: 450.5

2. Resume:
   $ python train_advanced.py --resume-from-checkpoint logs/checkpoints/checkpoint_2000_2024-01-15_14-32-45.pt

3. Verify resumption by checking logs:
   $ tail logs/tensorboard/training.log
   
   Should show step continuing from 2000, not restarting at 0


CHECKPOINT CLEANUP
──────────────────

Old checkpoints are automatically cleaned up to save disk space:
  - Keep latest N checkpoints (default: 5)
  - Delete older ones
  
To manually delete all checkpoints:
  $ rm -rf logs/checkpoints/checkpoint_*.pt logs/checkpoints/checkpoint_*.json
  
To keep all checkpoints (disable cleanup):
  Edit checkpoint_manager.py:
    cleanup_old_checkpoints(keep_n=999)  # Keep all


═══════════════════════════════════════════════════════════════════════════════
6. MONITORING & DEBUGGING
═══════════════════════════════════════════════════════════════════════════════

REAL-TIME MONITORING
────────────────────

Option 1: Follow SLURM log file
  $ tail -f logs/slurm_12345.log
  
Shows printed output in real-time

Option 2: Use training monitor
  $ python training_monitor.py summary --log-dir logs
  
Shows:
  - Checkpoints saved count
  - Videos recorded count
  - Metric plots generated count
  - Latest checkpoint stats

Option 3: TensorBoard (best for metrics)
  $ python training_monitor.py tensorboard --log-dir logs
  
Opens browser at http://localhost:6006
Shows:
  - Reward curves over time
  - Episode length distribution
  - Success rate trends
  - Distance to object tracking

DEBUGGING TRAINING
──────────────────

If reward not increasing:
  1. Check reward function in mdp/rewards.py
  2. Verify distance_reward is being computed: tail logs/slurm_*.log | grep "reward"
  3. Increase reward weights temporarily for testing
  4. Plot reward curves: python training_monitor.py metrics

If arms hitting table:
  1. Verify collision detection in mdp/terminations.py
  2. Monitor collision count: check TensorBoard for collision events
  3. Lower collision threshold to be more sensitive
  4. Visualize training: use videos in logs/videos/

If checkpoint not saving:
  1. Check write permissions: ls -la logs/checkpoints/
  2. Verify signal handlers: check if Ctrl+C prints "Saving checkpoint..."
  3. Test locally: python train_advanced.py (Ctrl+C), verify logs/checkpoints/ created

If video recording fails:
  1. Check imageio installed: pip list | grep imageio
  2. Verify logs/videos/ directory exists and writable
  3. Check frame buffer size isn't too large
  4. Look for errors in logs

COMMON ERROR MESSAGES
─────────────────────

Error: "IsaacLab module not found"
  Solution: conda activate isaac_fresh

Error: "CUDA out of memory"
  Solution: Reduce --num-envs to 2 or 1

Error: "Checkpoint file not found"
  Solution: Verify path, use get_latest_checkpoint()

Error: "sbatch: command not found"
  Solution: Only works on cluster with SLURM


═══════════════════════════════════════════════════════════════════════════════
7. ADVANCED TOPICS
═══════════════════════════════════════════════════════════════════════════════

DISTRIBUTED TRAINING (Multi-GPU)
────────────────────────────────

To run on multiple GPUs (advanced):

1. Modify train_advanced.py to use:
   from stable_baselines3.ppo.ppo import PPO
   
2. Create distributed version using torch.nn.parallel.DistributedDataParallel

3. Submit with multiple GPUs:
   sbatch --gres=gpu:4 submit_training.sh  # 4 GPUs

Note: Current implementation uses single GPU with parallel environments.
      For multi-GPU, needs more complex distributed PPO implementation.


CUSTOM REWARD FUNCTION
──────────────────────

To implement custom reward:

1. Edit mdp/rewards.py

2. Add new function:
   def custom_reward(env, **kwargs):
       # Your reward computation
       return reward_tensor  # Shape: (num_envs,)

3. In train_advanced.py, modify reward composition:
   total_reward = 0.5 * distance_reward + custom_reward()


CURRICULUM LEARNING
───────────────────

To add curriculum (easier tasks first):

1. In train_advanced.py, add stages:
   
   stages = [
       {"phase": 1, "steps": 100_000, "task": "reach_object"},
       {"phase": 2, "steps": 200_000, "task": "grasp_object"},
       {"phase": 3, "steps": 700_000, "task": "lift_object"},
   ]

2. Modify reward function per stage:
   if current_phase == 1:
       reward = distance_reward  # Only reward approaching
   elif current_phase == 2:
       reward = distance_reward + grasp_reward
   else:
       reward = full_reward  # All components


SIM-TO-REAL TRANSFER
────────────────────

To prepare for real robot deployment:

1. Train with domain randomization enabled:
   - Modify scene XML: add random mass/friction variations
   - Random camera viewpoints, lighting

2. Record demonstrations:
   - Save successful episodes from logs/videos/
   - Use for imitation learning initialization

3. Test robustness:
   - Evaluate with perturbed physics parameters
   - Check success rate in evaluate.py


═══════════════════════════════════════════════════════════════════════════════
QUICK REFERENCE COMMANDS
═══════════════════════════════════════════════════════════════════════════════

Setup & Testing:
  python setup_wizard.py                              # Check all requirements
  python integration_test.py                          # Test components

Training:
  python train_advanced.py --num-envs 4              # Train locally
  python train_advanced.py --resume-from-checkpoint ... # Resume
  
Cluster:
  sbatch submit_training.sh                           # Submit job
  python cluster_manager.py submit --gpu a100        # With manager
  python cluster_manager.py status {JOB_ID}          # Check status
  python cluster_manager.py cancel {JOB_ID}          # Cancel job

Monitoring:
  python training_monitor.py summary                  # Summary
  python training_monitor.py checkpoints              # List checkpoints
  python training_monitor.py videos                   # List videos
  python training_monitor.py tensorboard              # Launch TensorBoard

Utilities:
  python cluster_manager.py list                      # List active jobs
  squeue -l                                           # All queue jobs
  tail -f logs/slurm_*.log                           # Live logs


═══════════════════════════════════════════════════════════════════════════════
END OF REFERENCE GUIDE
═══════════════════════════════════════════════════════════════════════════════
"""


def main():
    print(GUIDE)


if __name__ == '__main__':
    main()
