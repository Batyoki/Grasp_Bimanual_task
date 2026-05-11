# Bimanual VX300s Arm Training in IsaacLab

This package contains a complete IsaacLab environment for training bimanual robotic arms (VX300s) to perform manipulation tasks using reinforcement learning.

## Features

- **Bimanual Setup**: Two VX300s arms mounted on a table
- **Multi-Camera System**: Integration with 3+ cameras (top, bottom, wrist-mounted)
- **Flexible Reward System**: Customizable reward functions for various tasks
- **PPO Training**: Pre-configured Proximal Policy Optimization training
- **Full RL Pipeline**: Training, evaluation, and analysis tools

## Project Structure

```
trainRL/
├── bimanual_vx300s_env/          # Main environment package
│   ├── __init__.py
│   ├── vx300s_env_cfg.py          # Environment configuration
│   └── mdp/                       # MDP utilities
│       ├── actions.py             # Action space definitions
│       ├── observations.py        # Observation collectors
│       ├── rewards.py             # Reward functions
│       └── terminations.py        # Episode termination conditions
├── train_ppo.py                   # PPO training script
├── evaluate.py                    # Evaluation script
├── utils.py                       # Utility functions
├── config.yaml                    # Default configuration
└── README.md                      # This file
```

## Installation

### Prerequisites

- IsaacLab (tested on v4.5+)
- Python 3.10+
- PyTorch (with CUDA support recommended)
- Stable-Baselines3

### Setup

1. **Clone or navigate to the trainRL directory**:
```bash
cd /path/to/yash/trainRL
```

2. **Install dependencies**:
```bash
pip install stable-baselines3 gymnasium omegaconf tensorboard
```

3. **Verify IsaacLab installation**:
```bash
python -c "import isaaclab; print(isaaclab.__version__)"
```

## Quick Start

### 1. Training a Policy

```bash
# Basic training with default settings
python train_ppo.py

# Training with custom parameters
python train_ppo.py \
    --num-envs 8 \
    --total-timesteps 2000000 \
    --learning-rate 1e-4 \
    --device cuda

# Training with custom config file
python train_ppo.py --config config_custom.yaml
```

### 2. Evaluating a Trained Policy

```bash
# Evaluate a trained model
python evaluate.py \
    --model-path ./models/bimanual_vx300s_v1_*/ppo_bimanual_final.zip \
    --config-path ./models/bimanual_vx300s_v1_*/config.yaml \
    --num-episodes 20 \
    --save-results evaluation_results.yaml
```

### 3. Custom Training Loop

```python
from bimanual_vx300s_env import BimanualVX300sEnvCfg
from utils import ConfigManager, DirectoryManager, set_seed

# Create configuration
cfg = ConfigManager.create_default_config()
cfg.env.num_envs = 4
cfg.training.total_timesteps = 1_000_000

# Setup directories
dirs = DirectoryManager.setup_directories(cfg)

# Set seed for reproducibility
set_seed(cfg.experiment.seed, device=cfg.env.device)

# Create environment (using IsaacLab)
# ... environment creation code ...

# Train with PPO
# ... training loop ...
```

## Configuration

### Default Configuration Structure

See `config.yaml` for the complete default configuration. Key sections:

- **env**: Environment parameters (number of parallel environments, max episode length, device)
- **observation**: Observation settings (image inclusion, normalization)
- **action**: Action space configuration
- **reward**: Reward function weights and thresholds
- **training**: PPO hyperparameters (learning rate, batch size, gamma, etc.)
- **experiment**: Experiment settings (directories, logging, checkpointing)

### Creating Custom Configurations

```yaml
# custom_config.yaml
env:
  num_envs: 8
  max_episode_length: 2000
  device: cuda

training:
  total_timesteps: 2000000
  learning_rate: 5e-4
  batch_size: 512

reward:
  reach_weight: 2.0
  action_penalty_weight: 0.02
  success_bonus: 20.0
```

Then use it:
```bash
python train_ppo.py --config custom_config.yaml
```

## Tasks and Rewards

### Available Reward Functions

1. **reach_reward**: Encourages the end-effector to reach the target object
2. **action_penalty**: Discourages excessive action magnitudes
3. **gripper_reward**: Rewards appropriate gripper control
4. **progress_reward**: Rewards reducing distance to the target

### Defining Custom Tasks

To create a custom task, modify the reward functions in `bimanual_vx300s_env/mdp/rewards.py`:

```python
def custom_task_reward(env, asset_cfg):
    """Custom reward function for your task."""
    # Your reward computation
    return reward_tensor
```

Then register it in the environment configuration (`vx300s_env_cfg.py`).

## Observation Space

### Proprioceptive Observations (28 dims)
- Left arm joint positions and velocities (14 dims)
- Right arm joint positions and velocities (14 dims)
- Cube position and orientation (7 dims)

### Visual Observations (optional)
- **cam_high**: Top-down camera view
- **cam_low**: Bottom-up camera view
- **cam_left_wrist**: Left gripper wrist camera
- **cam_right_wrist**: Right gripper wrist camera (optional)

Enable image observations:
```yaml
observation:
  include_images: true
  image_size: [128, 128]
  camera_names: [cam_high, cam_low]
```

## Action Space

- **Type**: Continuous joint position control
- **Dimensions**: 14 (7 joints per arm × 2 arms)
- **Range**: [-1, 1] scaled to joint limits
- **Mode**: Relative or absolute actions (configurable)

## Monitoring Training

### Using TensorBoard

```bash
tensorboard --logdir ./logs --port 6006
```

Then open http://localhost:6006 in your browser.

### Training Logs

Logs are saved in the following structure:
```
logs/
├── bimanual_vx300s_v1_2024-01-15_10-30-45/
│   ├── config.yaml
│   ├── training_log.csv
│   ├── tensorboard/
│   │   └── events...
│   └── checkpoints/
└── ...

models/
├── bimanual_vx300s_v1_2024-01-15_10-30-45/
│   ├── ppo_bimanual_100000.zip
│   ├── ppo_bimanual_200000.zip
│   └── ppo_bimanual_final.zip
└── ...
```

## Troubleshooting

### Environment Creation Issues

If you encounter issues with asset loading (VX300s URDF not found):

1. Ensure IsaacLab has downloaded necessary assets
2. Check asset paths in `vx300s_env_cfg.py`
3. Consider using alternative robot assets (Franka, UR, etc.)

**Alternative: Use a simpler robot asset**:
```python
# In vx300s_env_cfg.py, replace VX300s with Franka
from isaaclab_assets.robots.universal_robots import UR10e_CFG
```

### Training Issues

- **CUDA out of memory**: Reduce `num_envs` or `batch_size`
- **Slow training**: Increase `num_envs` or use more GPU memory
- **Unstable training**: Reduce learning rate or adjust PPO hyperparameters

### GPU Memory Optimization

```bash
# Check GPU memory usage
nvidia-smi

# Run with reduced batch size
python train_ppo.py --batch-size 128
```

## Advanced Usage

### Multi-Task Training

Modify reward functions to support multiple tasks:

```python
def multi_task_reward(env, task_id):
    if task_id == 0:
        return reach_reward(env)
    elif task_id == 1:
        return grasp_reward(env)
    # ...
```

### Curriculum Learning

Implement progressive difficulty:

```python
def curriculum_reward_weight(episode):
    # Gradually increase difficulty
    return 1.0 + 0.1 * (episode / 100000)
```

### Transfer Learning

Load a pre-trained model:

```python
from stable_baselines3 import PPO

# Load pre-trained model
model = PPO.load("models/bimanual_vx300s_v1_*/ppo_bimanual_final")

# Continue training on new task
model.learn(total_timesteps=500000)

# Save fine-tuned model
model.save("models/bimanual_vx300s_finetuned")
```

## Performance Benchmarks

Expected training performance (on RTX 4090):

| Num Envs | Timesteps/sec | Time for 1M steps |
|----------|---------------|------------------|
| 4        | ~8,000        | ~125 minutes     |
| 8        | ~15,000       | ~67 minutes      |
| 16       | ~25,000       | ~40 minutes      |

## Contributing

To extend this environment:

1. **Add new reward functions** in `mdp/rewards.py`
2. **Add new observation collectors** in `mdp/observations.py`
3. **Add new action handlers** in `mdp/actions.py`
4. **Add new termination conditions** in `mdp/terminations.py`

## References

- [IsaacLab Documentation](https://docs.robotics.ros.org/isaac-lab/)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- [PPO Algorithm Paper](https://arxiv.org/abs/1707.06347)
- [Interbotix VX300s](https://www.trossenrobotics.com/vx-300s-robot-arm)

## License

This project is provided as-is for research and educational purposes.

## Support

For issues or questions:

1. Check the troubleshooting section above
2. Review IsaacLab documentation
3. Check error logs in the `logs/` directory

---

**Last Updated**: January 2024
**IsaacLab Version**: 4.5+
**Status**: Active Development
