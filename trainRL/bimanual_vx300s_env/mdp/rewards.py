"""
Reward functions for the bimanual VX300s environment.

Advanced reward functions for:
- Distance-based approaching
- Grasping and object control
- Lifting and manipulation
- Gripper control penalties
"""

from typing import Callable, Optional

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs.mdp import RewardTermCfg


def reach_reward(
    env,
    asset_cfg: SceneEntityCfg,
    target_pos: torch.Tensor = None,
    min_distance: float = 0.05,
) -> torch.Tensor:
    """
    Reward for reaching the object (cube) with end-effector.
    
    Uses exponential decay based on distance to encourage close approach.

    Args:
        env: The RL environment.
        asset_cfg: The scene entity configuration for the target object.
        target_pos: Optional target position. If None, uses current cube position.
        min_distance: Minimum distance threshold (m).

    Returns:
        Reward tensor of shape (num_envs,).
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # Get cube position
    cube_pos = asset.data.root_pos_w
    
    # Get left end-effector position
    if hasattr(env, '_left_ee_pos'):
        left_ee_pos = env._left_ee_pos
    else:
        try:
            left_arm: Articulation = env.scene["left_arm"]
            left_ee_pos = left_arm.data.body_pos_w[:, -1, :]
        except KeyError:
            return torch.zeros(env.num_envs, device=env.device)
    
    # Compute distance
    distance = torch.norm(left_ee_pos - cube_pos, dim=-1)
    
    # Exponential reward (closer = more reward)
    # R = e^(-alpha * distance) where alpha controls the decay rate
    alpha = 2.0  # Decay rate
    reach_reward = torch.exp(-alpha * distance)
    
    # Bonus for being very close
    very_close_bonus = torch.where(distance < min_distance, torch.ones_like(distance), torch.zeros_like(distance)) * 0.5
    
    return reach_reward + very_close_bonus


def action_penalty(env) -> torch.Tensor:
    """
    Penalty for excessive action magnitudes (encourages smooth, low-energy motions).

    Args:
        env: The RL environment.

    Returns:
        Penalty tensor of shape (num_envs,).
    """
    # Get the last action
    if hasattr(env, 'last_actions'):
        actions = env.last_actions
        penalty = torch.norm(actions, dim=-1)
    else:
        penalty = torch.zeros(env.num_envs, device=env.device)
    
    return penalty


def gripper_reward(env) -> torch.Tensor:
    """
    Reward for appropriate gripper control.
    
    Encourages smooth gripper movements without unnecessary oscillation.

    Args:
        env: The RL environment.

    Returns:
        Reward tensor of shape (num_envs,).
    """
    if hasattr(env, 'last_actions'):
        # Penalize rapid gripper changes (assume last two action dims are grippers)
        gripper_actions = env.last_actions[:, -2:]
        gripper_penalty = -0.1 * torch.mean(torch.abs(gripper_actions), dim=-1)
        return gripper_penalty
    
    return torch.zeros(env.num_envs, device=env.device)


def grasping_reward(
    env,
    asset_cfg: SceneEntityCfg,
    grasp_threshold: float = 0.01,
    gripper_opening: float = 0.05,
) -> torch.Tensor:
    """
    Reward for successful grasping and holding the object.
    
    Requires:
    - End-effector close to object
    - Gripper closed around object
    - Object not falling

    Args:
        env: The RL environment.
        asset_cfg: Scene entity configuration for the object.
        grasp_threshold: Distance threshold for successful grasp (m).
        gripper_opening: Minimum gripper opening for grasp (m).

    Returns:
        Reward tensor of shape (num_envs,).
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # Get cube position and velocity
    cube_pos = asset.data.root_pos_w
    cube_vel = asset.data.root_vel_w
    
    # Get left end-effector position
    if hasattr(env, '_left_ee_pos'):
        left_ee_pos = env._left_ee_pos
    else:
        try:
            left_arm: Articulation = env.scene["left_arm"]
            left_ee_pos = left_arm.data.body_pos_w[:, -1, :]
        except KeyError:
            return torch.zeros(env.num_envs, device=env.device)
    
    distance_to_obj = torch.norm(left_ee_pos - cube_pos, dim=-1)
    
    if hasattr(env, 'last_actions'):
        # Assume last two action dims correspond to grippers
        gripper_pos = env.last_actions[:, -2:]
        gripper_closed = torch.all(gripper_pos < gripper_opening, dim=-1)
    else:
        gripper_closed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    
    # Object stability check: low velocity means it's stable in hand
    obj_velocity_magnitude = torch.norm(cube_vel, dim=-1)
    obj_stable = obj_velocity_magnitude < 0.5
    
    # Grasp success: close to object, gripper closed, object stable
    grasping = (distance_to_obj < grasp_threshold) & gripper_closed & obj_stable
    
    # Reward for successful grasping
    reward = grasping.float() * 1.0
    
    return reward


def lifting_reward(
    env,
    asset_cfg: SceneEntityCfg,
    table_height: float = 0.025,
    lift_height_threshold: float = 0.05,
) -> torch.Tensor:
    """
    Reward for lifting the object above the table.
    
    Encourages the agent to lift the grasped object high enough.

    Args:
        env: The RL environment.
        asset_cfg: Scene entity configuration for the object.
        table_height: Height of the table surface (m).
        lift_height_threshold: Minimum height above table to count as lifted (m).

    Returns:
        Reward tensor of shape (num_envs,).
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # Get cube height
    cube_z = asset.data.root_pos_w[:, 2]
    
    # Height above table
    height_above_table = cube_z - table_height
    
    # Reward: linear increase with height (up to a maximum)
    max_reward_height = 0.5  # m
    lifting_reward = torch.clamp(height_above_table / max_reward_height, min=0.0, max=1.0)
    
    return lifting_reward


def gripper_penalty(env, max_action: float = 1.0) -> torch.Tensor:
    """
    Penalty for excessive gripper control.
    
    Discourages erratic or wasteful gripper movements.

    Args:
        env: The RL environment.
        max_action: Maximum expected action magnitude.

    Returns:
        Penalty tensor of shape (num_envs,).
    """
    if hasattr(env, 'last_actions'):
        gripper_actions = env.last_actions[:, -2:]
        # Penalty proportional to action magnitude
        penalty = -0.05 * torch.mean(torch.abs(gripper_actions) / max_action, dim=-1)
        return penalty
    
    return torch.zeros(env.num_envs, device=env.device)


def progress_reward(
    env,
    prev_distance: torch.Tensor,
    distance_scale: float = 0.1,
) -> torch.Tensor:
    """
    Reward for making progress towards the object.
    
    Encourages continuous improvement in reaching distance.

    Args:
        env: The RL environment.
        prev_distance: Previous distance to object.
        distance_scale: Scaling factor for progress reward.

    Returns:
        Progress reward tensor of shape (num_envs,).
    """
    # Compute current distance
    if hasattr(env, '_left_ee_pos'):
        try:
            asset: RigidObject = env.scene["cube"]
        except KeyError:
            return torch.zeros(env.num_envs, device=env.device)
        
        cube_pos = asset.data.root_pos_w
        left_ee_pos = env._left_ee_pos
        current_distance = torch.norm(left_ee_pos - cube_pos, dim=-1)
        
        # Reward for reducing distance
        progress = prev_distance - current_distance
        reward = distance_scale * torch.clamp(progress, min=-0.1, max=0.5)
    else:
        reward = torch.zeros(env.num_envs, device=env.device)
    
    return reward


def distance_reward(
    env,
    asset_cfg: SceneEntityCfg,
    max_distance: float = 2.0,
    min_reward: float = -0.1,
) -> torch.Tensor:
    """
    Reward based on distance to object.
    
    Continuous reward that encourages moving closer.

    Args:
        env: The RL environment.
        asset_cfg: Scene entity configuration for the target.
        max_distance: Maximum distance to consider (m).
        min_reward: Minimum reward value.

    Returns:
        Distance reward tensor of shape (num_envs,).
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    cube_pos = asset.data.root_pos_w
    
    if hasattr(env, '_left_ee_pos'):
        left_ee_pos = env._left_ee_pos
    else:
        try:
            left_arm: Articulation = env.scene["left_arm"]
            left_ee_pos = left_arm.data.body_pos_w[:, -1, :]
        except KeyError:
            return torch.zeros(env.num_envs, device=env.device)
    
    distance = torch.norm(left_ee_pos - cube_pos, dim=-1)
    
    # Normalized distance reward: closer = higher reward
    distance_normalized = torch.clamp(distance / max_distance, 0.0, 1.0)
    reward = (1.0 - distance_normalized) + min_reward
    
    return reward
