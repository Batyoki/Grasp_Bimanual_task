"""
Termination conditions for the bimanual VX300s environment.

Includes:
- Episode timeout
- Table collision detection
- Out of bounds checking
- Object grasping success
"""

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg


def time_out(env) -> torch.Tensor:
    """
    Terminate episode when maximum episode length is reached.

    Args:
        env: The RL environment.

    Returns:
        Termination signal tensor of shape (num_envs,).
    """
    return env.episode_length_buf >= env.max_episode_length


def table_collision(
    env,
    asset_cfg: SceneEntityCfg,
    table_height: float = 0.025,
    min_clearance: float = 0.01,
) -> torch.Tensor:
    """
    Terminate if arm collides with the table (goes below surface).
    
    Protects the arms from drilling into the table.

    Args:
        env: The RL environment.
        asset_cfg: The scene entity configuration for the arm.
        table_height: Height of table surface (m).
        min_clearance: Minimum clearance above table (m).

    Returns:
        Termination signal tensor of shape (num_envs,).
    """
    try:
        arm: Articulation = env.scene[asset_cfg.name]
    except KeyError:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    
    # Get end-effector Z position (height)
    ee_pos = arm.data.body_pos_w[:, -1, :]
    ee_height = ee_pos[:, 2]
    
    # Check if end-effector is below table (collision)
    collision = ee_height < (table_height + min_clearance)
    
    return collision


def cube_out_of_bounds(
    env,
    asset_cfg: SceneEntityCfg,
    boundary_x: float = 1.0,
    boundary_y: float = 1.0,
    boundary_z_min: float = 0.2,
    boundary_z_max: float = 1.5,
) -> torch.Tensor:
    """
    Terminate if the cube moves too far away from the table or too high.

    Args:
        env: The RL environment.
        asset_cfg: The scene entity configuration for the cube.
        boundary_x: X-axis boundary (m).
        boundary_y: Y-axis boundary (m).
        boundary_z_min: Minimum Z boundary (m).
        boundary_z_max: Maximum Z boundary (m).

    Returns:
        Termination signal tensor of shape (num_envs,).
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    
    pos = asset.data.root_pos_w
    
    # Check boundaries
    out_of_x = torch.abs(pos[:, 0]) > boundary_x
    out_of_y = torch.abs(pos[:, 1]) > boundary_y
    out_of_z_min = pos[:, 2] < boundary_z_min
    out_of_z_max = pos[:, 2] > boundary_z_max
    
    out_of_bounds = out_of_x | out_of_y | out_of_z_min | out_of_z_max
    
    return out_of_bounds


def arm_collision(
    env,
    asset_cfg: SceneEntityCfg,
    table_height: float = 0.025,
) -> torch.Tensor:
    """
    Terminate if the arm collides with the table or itself.

    Args:
        env: The RL environment.
        asset_cfg: The scene entity configuration for the arm.
        table_height: Height of table surface (m).

    Returns:
        Termination signal tensor of shape (num_envs,).
    """
    try:
        arm: Articulation = env.scene[asset_cfg.name]
    except KeyError:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    
    # Get all link positions
    body_pos = arm.data.body_pos_w
    
    # Check if any link is below table (except base)
    z_positions = body_pos[:, 1:, 2]  # Skip base link
    below_table = z_positions < table_height
    
    # Collision if any link goes below table
    collision = torch.any(below_table, dim=-1)
    
    return collision


def object_grasped(
    env,
    ee_pos: torch.Tensor = None,
    cube_cfg: SceneEntityCfg = None,
    grasp_threshold: float = 0.01,
) -> torch.Tensor:
    """
    Optional: Terminate successfully when object is grasped and lifted.

    Args:
        env: The RL environment.
        ee_pos: End-effector position tensor.
        cube_cfg: Scene entity configuration for the cube.
        grasp_threshold: Distance threshold for successful grasp.

    Returns:
        Termination signal tensor of shape (num_envs,).
    """
    if cube_cfg is None or ee_pos is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    
    asset: RigidObject = env.scene[cube_cfg.name]
    cube_pos = asset.data.root_pos_w
    
    distance = torch.norm(ee_pos - cube_pos, dim=-1)
    grasped = distance < grasp_threshold
    
    return grasped


def high_object_lift(
    env,
    asset_cfg: SceneEntityCfg,
    table_height: float = 0.025,
    lift_threshold: float = 0.2,
    success_termination: bool = False,
) -> torch.Tensor:
    """
    Terminate (optionally as success) when object is lifted high enough.

    Args:
        env: The RL environment.
        asset_cfg: Scene entity configuration for the object.
        table_height: Height of table surface (m).
        lift_threshold: Height threshold above table for success (m).
        success_termination: If True, counts as episode success.

    Returns:
        Termination signal tensor of shape (num_envs,).
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    cube_z = asset.data.root_pos_w[:, 2]
    
    height_above_table = cube_z - table_height
    lifted = height_above_table > lift_threshold
    
    return lifted
