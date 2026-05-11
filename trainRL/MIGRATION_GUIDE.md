# Migration Guide: From MuJoCo to IsaacLab

This guide explains how to adapt your MuJoCo bimanual VX300s environment to IsaacLab.

## Key Differences

### 1. Physics Engine

**MuJoCo**:
```python
import mujoco
model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)
```

**IsaacLab**:
```python
import isaaclab.sim as sim_utils
# Physics configured in SimulationCfg
cfg = sim_utils.SimulationCfg(
    dt=0.01,
    disable_contact_processing=False,
)
```

### 2. Scene Setup

**MuJoCo**:
- XML-based scene definition
- Manual asset loading
- Direct XML parsing

**IsaacLab**:
```python
@configclass
class BimanualVX300sSceneCfg(InteractiveSceneCfg):
    # Declarative scene definition
    left_arm = ArticulationCfg(...)
    cube = RigidObjectCfg(...)
    table = AssetBaseCfg(...)
```

### 3. Environment Integration

**MuJoCo**:
```python
class BimanualArmTransferEnv(gym.Env):
    # Manual RL environment implementation
    def step(self, action):
        # Manual dynamics stepping
        mujoco.mj_step(self.model, self.data)
```

**IsaacLab**:
```python
class BimanualVX300sEnv(ManagerBasedRLEnv):
    # Automatic RL task management
    # Rewards, observations, actions handled by managers
```

## Mapping Components

### Observations

**MuJoCo** (`gym_env.py`):
```python
def _get_proprioceptive_obs(self):
    joint_pos = self.data.qpos
    cube_pos = self.data.body(self.body_ids["cube"]).xpos
    # Returns 28-dim observation
    return obs
```

**IsaacLab** (`mdp/observations.py`):
```python
def arm_observations(env, asset_cfg):
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos
    joint_vel = asset.data.joint_vel
    # Registered in managers
    return obs
```

### Rewards

**MuJoCo** (`gym_env.py`):
```python
def _compute_reward(self):
    distance = np.linalg.norm(ee_pos - cube_pos)
    reward = np.exp(-distance)
    # Mixed into environment loop
    return reward
```

**IsaacLab** (`mdp/rewards.py`):
```python
def reach_reward(env, asset_cfg):
    distance = torch.norm(ee_pos - cube_pos, dim=-1)
    reward = torch.exp(-distance)
    # Registered in manager
    return reward
```

### Actions

**MuJoCo** (`gym_env.py`):
```python
action_space = Box(low=-1, high=1, shape=(14,))
# Manual action scaling and application
self.data.ctrl[:] = action * self.action_scale
```

**IsaacLab** (`mdp/actions.py`):
```python
class ArmActionCfg(ActionTermCfg):
    class_type = mdp.JointPositionToLimitsActionCfg
    # Automatic action handling and scaling
```

### Cameras

**MuJoCo** (`gym_env.py`):
```python
def _render_camera(self, camera_name):
    cam_id = mujoco.mj_name2id(
        self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name
    )
    # Manual rendering
    rgb = self.renderer.render(...)
```

**IsaacLab** (`vx300s_env_cfg.py`):
```python
# Cameras defined in scene configuration
# Automatic rendering through sensor managers
camera = CameraCfg(
    prim_path="/World/envs/env_*/Camera",
    resolution=(640, 480),
)
```

## Migration Steps

### 1. Convert Scene Description

**MuJoCo XML**:
```xml
<mujoco model="bimanual">
  <body name="vx300s_left_base_link">
    <!-- Left arm -->
  </body>
  <body name="vx300s_right_base_link">
    <!-- Right arm -->
  </body>
</mujoco>
```

**IsaacLab Configuration**:
```python
@configclass
class BimanualVX300sSceneCfg(InteractiveSceneCfg):
    left_arm = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/LeftArm",
        spawn=UsdFileCfg(usd_path="..."),
    )
    right_arm = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/RightArm",
        spawn=UsdFileCfg(usd_path="..."),
    )
```

### 2. Extract Observation Functions

MuJoCo observation collection logic → IsaacLab observation term functions

```python
# Move from gym_env._get_proprioceptive_obs() to observations.py
def arm_observations(env, asset_cfg):
    # Same computation logic, but:
    # - Use PyTorch tensors instead of NumPy
    # - Vectorized operations for parallel environments
```

### 3. Implement Reward Terms

MuJoCo reward computation → Individual IsaacLab reward functions

```python
# Split env._compute_reward() into separate functions
def reach_reward(env, asset_cfg):
    # Reach component
    
def action_penalty(env):
    # Action penalty component
    
def gripper_reward(env):
    # Gripper component
```

### 4. Define Termination Conditions

MuJoCo termination logic → IsaacLab termination terms

```python
# Move from gym_env._is_done() to terminations.py
def time_out(env):
    return env.episode_length_buf >= env.max_episode_length

def cube_out_of_bounds(env, asset_cfg, boundary=1.0):
    # Out of bounds checking
```

### 5. Configure Environment Manager

Combine all MDP components into unified configuration:

```python
class BimanualVX300sEnvCfg(ManagerBasedRLEnvCfg):
    observations = {
        "policy": ObsGroup(
            obs_terms=dict(
                arm_obs=ObsTerm(func=arm_observations, ...),
                cube_obs=ObsTerm(func=cube_observation, ...),
            )
        )
    }
    
    rewards = {
        "reach": RewTerm(func=reach_reward, weight=1.0, ...),
        "action_penalty": RewTerm(func=action_penalty, weight=0.01, ...),
    }
```

## Common Patterns

### Parallel Environments

MuJoCo typically runs single environment; IsaacLab naturally parallelizes:

```python
# MuJoCo
env = BimanualArmTransferEnv()
obs = env.reset()  # Single environment

# IsaacLab
obs = env.reset()  # Already (num_envs, obs_dim)
```

### Device Compatibility

MuJoCo runs on CPU; IsaacLab leverages GPU:

```python
# IsaacLab automatically handles GPU tensors
reward = torch.exp(-distance)  # GPU-accelerated
```

### Vectorized Operations

Always use PyTorch operations for batch processing:

```python
# ✓ Correct - vectorized
distance = torch.norm(ee_pos - cube_pos, dim=-1)  # Shape: (num_envs,)

# ✗ Wrong - loops
for i in range(num_envs):
    distance[i] = np.linalg.norm(...)
```

## Testing Migration

1. **Unit Tests**: Test each MDP component independently
   ```bash
   pytest tests/test_observations.py
   pytest tests/test_rewards.py
   ```

2. **Integration Tests**: Test environment setup
   ```bash
   python -c "from bimanual_vx300s_env import BimanualVX300sEnvCfg; cfg = BimanualVX300sEnvCfg()"
   ```

3. **Sanity Checks**: Verify behavior
   - Observations in correct shape/range
   - Rewards responsive to actions
   - Terminations trigger appropriately

## Troubleshooting

### Asset Loading Issues

IsaacLab uses USD format; MuJoCo uses URDF/STL:

```python
# Conversion needed
# Option 1: Use IsaacLab's built-in robots
from isaaclab_assets.robots.universal_robots import UR10_CFG

# Option 2: Import URDF and convert to USD
# Use IsaacSim's built-in URDF importer
```

### Physics Mismatch

IsaacLab PhysX vs MuJoCo physics engine differences:

- Friction coefficients may need tuning
- Contact thresholds different
- Damping/stiffness values may differ

**Solution**: Use `config_physics_equivalent` configs in IsaacLab

### Performance Issues

- MuJoCo: ~100s timesteps/sec (CPU)
- IsaacLab: ~10,000s timesteps/sec (GPU, 4 envs)

Use parallel environments efficiently!

## References

- [IsaacLab Manager-Based RL](https://docs.robotics.ros.org/isaac-lab/source/api/controller/manager_based_rl_env.html)
- [MuJoCo Documentation](http://www.mujoco.org/)
- [PhysX vs MuJoCo Comparison](https://nvidia-omniverse.github.io/IsaacSim/latest/)

## Next Steps

1. Convert robot URDF to USD format if needed
2. Implement custom MDP terms for your specific task
3. Fine-tune physics simulation parameters
4. Run benchmarks comparing MuJoCo vs IsaacLab
5. Deploy trained policy to real robots

---

For questions or issues during migration, refer to the main README.md and IsaacLab documentation.
