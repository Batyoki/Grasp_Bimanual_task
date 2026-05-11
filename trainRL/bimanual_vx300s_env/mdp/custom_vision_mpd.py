import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation, RigidObject

def image_features(env, sensor_cfg: SceneEntityCfg, data_type: str = "rgb") -> torch.Tensor:
    """Extracts image tensor from camera sensor. Shape: (num_envs, H, W, C)"""
    camera = env.scene.sensors[sensor_cfg.name]
    return camera.data.output[data_type].clone()

def reach_reward(env, asset_cfg: SceneEntityCfg, min_distance: float = 0.05) -> torch.Tensor:
    """Reward for end-effector reaching the cube."""
    cube: RigidObject = env.scene[asset_cfg.name]
    robot: Articulation = env.scene["robot"]
    
    cube_pos = cube.data.root_pos_w
    ee_pos = robot.data.body_pos_w[:, -1, :] # Assuming last body is EE
    
    distance = torch.norm(ee_pos - cube_pos, dim=-1)
    reach_rew = torch.exp(-2.0 * distance)
    bonus = torch.where(distance < min_distance, torch.ones_like(distance), torch.zeros_like(distance)) * 0.5
    
    return reach_rew + bonus

def grasping_reward(env, asset_cfg: SceneEntityCfg, grasp_threshold: float = 0.05) -> torch.Tensor:
    """Reward for closing gripper around the cube."""
    cube: RigidObject = env.scene[asset_cfg.name]
    robot: Articulation = env.scene["robot"]
    
    cube_pos = cube.data.root_pos_w
    ee_pos = robot.data.body_pos_w[:, -1, :]
    distance = torch.norm(ee_pos - cube_pos, dim=-1)
    
    # Check if gripper joints are closed (assuming last two joints are fingers)
    gripper_pos = robot.data.joint_pos[:, -2:]
    gripper_closed = torch.all(gripper_pos < 0.02, dim=-1)
    
    # Stable cube check
    cube_vel = torch.norm(cube.data.root_vel_w, dim=-1)
    stable = cube_vel < 0.5
    
    grasping = (distance < grasp_threshold) & gripper_closed & stable
    return grasping.float() * 2.0

def lifting_reward(env, asset_cfg: SceneEntityCfg, table_height: float = 0.05) -> torch.Tensor:
    """Reward for lifting the object above the table."""
    cube: RigidObject = env.scene[asset_cfg.name]
    cube_z = cube.data.root_pos_w[:, 2]
    
    height_above_table = cube_z - table_height
    lift_rew = torch.clamp(height_above_table / 0.5, min=0.0, max=1.0)
    return lift_rew * 3.0

def action_penalty(env) -> torch.Tensor:
    """Penalize erratic, high-energy movements."""
    if hasattr(env, 'last_actions') and env.last_actions is not None:
        return -0.05 * torch.norm(env.last_actions, dim=-1)
    return torch.zeros(env.num_envs, device=env.device)