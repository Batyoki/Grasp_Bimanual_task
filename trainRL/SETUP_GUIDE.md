# IsaacLab Setup Guide for Bimanual VX300s

This guide walks you through setting up and running the bimanual VX300s environment in IsaacLab.

## Prerequisites

- **IsaacLab**: Version 4.5 or later
- **Isaac Sim**: 4.5 or later (usually installed with IsaacLab)
- **Python**: 3.10+
- **NVIDIA GPU**: RTX 3090 or better recommended (RTX 4090 ideal)
- **CUDA**: 11.8+

## Installation Steps

### 1. Install IsaacLab

If not already installed:

```bash
# Clone IsaacLab repository
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab

# Install using the provided script
./isaaclab.sh --install

# Activate the environment
source _isaac_sim/setup_conda_env.sh
```

### 2. Install Dependencies

From the `trainRL` directory:

```bash
cd /path/to/yash/trainRL

# Install Python dependencies
pip install -r requirements.txt

# Additional IsaacLab-specific packages
pip install stable-baselines3 gymnasium
```

### 3. Environment Registration

To register the bimanual VX300s environment with IsaacLab:

#### Option A: Add to IsaacLab Extension (Recommended)

1. Copy the `bimanual_vx300s_env` folder to:
   ```
   IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/bimanual/
   ```

2. Create `__init__.py` in the bimanual directory:
   ```python
   # bimanual/__init__.py
   from .vx300s_env_cfg import BimanualVX300sEnvCfg
   
   __all__ = ["BimanualVX300sEnvCfg"]
   ```

3. Update IsaacLab's task registry by adding to:
   ```
   IsaacLab/source/isaaclab_tasks/isaaclab_tasks/__init__.py
   ```

#### Option B: Use from Local Directory

In your training scripts, import directly:

```python
import sys
sys.path.insert(0, '/path/to/yash/trainRL')

from bimanual_vx300s_env import BimanualVX300sEnvCfg
```

### 4. Verify Installation

Test the setup:

```bash
cd trainRL

# Test import
python -c "from bimanual_vx300s_env import BimanualVX300sEnvCfg; print('✓ Import successful')"

# Test environment creation
python quickstart.py  # Select option 5 (demo)
```

## Asset Configuration

### Robot Assets

The environment expects VX300s robot assets. You have several options:

**Option 1: Use USD Assets from IsaacLab Nucleus**

```python
# In vx300s_env_cfg.py
spawn=UsdFileCfg(
    usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/InterbotixRobots/VX300S/vx_300s.usd"
)
```

**Option 2: Convert from URDF**

If you have URDF files from the MuJoCo setup:

```bash
# Use IsaacSim's URDF importer
# Launch IsaacSim GUI and use File > Import > Select URDF
# Export as USD

# Or use command-line conversion
python -m isaaclab.sim.utils.conversion import_urdf "path/to/vx300s.urdf"
```

**Option 3: Use Alternative Robots**

If VX300s assets are unavailable, use similar robots:

```python
from isaaclab_assets.robots.universal_robots import UR10_CFG
# Or
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG
```

### Table and Environment Assets

Tables and backgrounds are typically available in IsaacLab Nucleus:

```python
UsdFileCfg(
    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Shelves/ShelfStraight_A.usd"
)
```

## Running Training

### Quick Start

```bash
# Interactive menu
python quickstart.py

# Or direct training
python train_ppo.py --num-envs 4 --total-timesteps 1000000
```

### Advanced Training

```bash
# Using debug configuration (small, fast)
python train_ppo.py --config config_debug.yaml

# Using production configuration (large, thorough)
python train_ppo.py --config config_production.yaml

# Using vision configuration (with camera observations)
python train_ppo.py --config config_vision.yaml

# Custom configuration
python train_ppo.py --config my_custom_config.yaml
```

### Training Parameters

```bash
# Start with 4 parallel environments
python train_ppo.py --num-envs 4 --total-timesteps 1000000

# Scale up for faster convergence
python train_ppo.py --num-envs 16 --total-timesteps 5000000

# Adjust learning rate
python train_ppo.py --learning-rate 5e-4

# Use CPU (slower, but works without GPU)
python train_ppo.py --device cpu
```

## Monitoring Training

### TensorBoard

```bash
# In a separate terminal
tensorboard --logdir ./logs --port 6006

# Open browser to http://localhost:6006
```

### Training Logs

Training progress is saved in:

```
logs/
├── bimanual_vx300s_v1_YYYY-MM-DD_HH-MM-SS/
│   ├── config.yaml           # Training configuration
│   ├── training_log.csv      # Timestep-by-step metrics
│   └── tensorboard/          # TensorBoard event files
└── ...
```

View summary:

```bash
python quickstart.py  # Select option 3 (View Results)
```

## Evaluation

### Evaluate Trained Model

```bash
# Find your trained model
model_path="./models/bimanual_vx300s_v1_YYYY-MM-DD_HH-MM-SS/ppo_bimanual_final.zip"
config_path="./logs/bimanual_vx300s_v1_YYYY-MM-DD_HH-MM-SS/config.yaml"

# Evaluate
python evaluate.py \
    --model-path "$model_path" \
    --config-path "$config_path" \
    --num-episodes 20 \
    --save-results results.yaml
```

### Analysis

```python
import yaml

with open('results.yaml') as f:
    results = yaml.safe_load(f)

print(f"Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
print(f"Success Rate: {results.get('success_rate', 'N/A')}")
```

## Customization

### Adding New Reward Functions

1. Create function in `bimanual_vx300s_env/mdp/rewards.py`:

```python
def custom_reward(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Your custom reward logic."""
    return reward_tensor
```

2. Register in environment config:

```python
self.rewards = {
    "custom": RewTerm(
        func=mdp.custom_reward,
        weight=0.5,
        params={"asset_cfg": SceneEntityCfg("cube")},
    ),
}
```

### Adding New Observations

1. Create function in `bimanual_vx300s_env/mdp/observations.py`:

```python
def custom_observation(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Your custom observation."""
    return obs_tensor
```

2. Register in environment config:

```python
self.observations["policy"].obs_terms["custom"] = ObsTerm(
    func=mdp.custom_observation,
    params={"asset_cfg": SceneEntityCfg("robot")},
)
```

### Modifying Task Difficulty

Adjust in `config.yaml`:

```yaml
reward:
  reach_weight: 2.0  # Increase importance of reaching
  action_penalty_weight: 0.005  # Reduce penalty for aggressive actions
  success_threshold: 0.02  # Tighter success threshold

training:
  total_timesteps: 5000000  # More training time
```

## Troubleshooting

### Common Issues and Solutions

#### 1. "VX300S USD asset not found"

```python
# Fallback to alternative robot
from isaaclab_assets.robots.universal_robots import UR10_CFG

# Or check asset availability
python -c "from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR; print(ISAAC_NUCLEUS_DIR)"
```

#### 2. CUDA out of memory

```bash
# Reduce number of environments
python train_ppo.py --num-envs 2

# Reduce batch size (in config)
batch_size: 64  # Was 256
```

#### 3. Slow training performance

```bash
# Increase number of environments (if GPU memory allows)
python train_ppo.py --num-envs 16

# Use lower image resolution (if using vision)
image_size: [48, 48]  # Was [64, 64]
```

#### 4. Environment creation fails

```bash
# Check IsaacLab installation
python -c "import isaaclab; isaaclab.core.sim.build_env"

# Test basic scene
python -c "
from bimanual_vx300s_env.vx300s_env_cfg import BimanualVX300sSceneCfg
cfg = BimanualVX300sSceneCfg()
print('✓ Scene config OK')
"
```

#### 5. Policy doesn't learn

Check:
- Reward scale: Is reward in reasonable range [-10, 10]?
- Observation scale: Are observations normalized?
- Action scale: Can actions actually affect the environment?
- Episode length: Is it long enough to reach goal?

```bash
# Run in debug mode with smaller environment
python train_ppo.py --config config_debug.yaml --num-envs 1
```

## Performance Benchmarks

Expected training speed:

| GPU | Num Envs | Timesteps/sec | 1M steps |
|-----|----------|---------------|----------|
| RTX 3080 | 4 | 6,000 | 167 min |
| RTX 3080 | 8 | 10,000 | 100 min |
| RTX 4090 | 4 | 8,000 | 125 min |
| RTX 4090 | 16 | 25,000 | 40 min |
| A100 | 16 | 40,000 | 25 min |

## Next Steps

1. **Explore different reward functions**: Reach, grasp, lift, stack, etc.
2. **Implement curriculum learning**: Gradually increase task difficulty
3. **Add vision-based control**: Enable camera observations
4. **Deploy to real robot**: Export trained policy for hardware
5. **Benchmark against baselines**: Compare with other approaches

## Resources

- [IsaacLab Documentation](https://docs.robotics.ros.org/isaac-lab/)
- [IsaacLab GitHub](https://github.com/isaac-sim/IsaacLab)
- [Stable-Baselines3 Docs](https://stable-baselines3.readthedocs.io/)
- [PPO Algorithm Paper](https://arxiv.org/abs/1707.06347)

## Support

For issues:

1. Check `logs/*/training_log.csv` for metrics
2. Review error messages in terminal
3. Check `MIGRATION_GUIDE.md` for MuJoCo→IsaacLab mapping
4. Consult IsaacLab documentation

---

**Maintained by**: Your Research Team  
**Last Updated**: January 2024  
**Status**: Active Development
