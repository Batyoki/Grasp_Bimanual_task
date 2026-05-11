"""
Observation functions for the bimanual VX300s environment.
"""

from typing import Callable

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs.mdp import ObservationTermCfg
from isaaclab.managers import SceneEntityCfg

def arm_observations(
    env,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """
    Collect observations from the arm (joint positions, velocities, and end-effector state).

    Args:
        env: The RL environment.
        asset_cfg: The scene entity configuration for the arm.

    Returns:
        Tensor of shape (num_envs, 28) containing arm observations.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get joint positions and velocities (7 DOF per arm × 2 arms = 14 dims)
    joint_pos = asset.data.joint_pos
    joint_vel = asset.data.joint_vel
    
    # Combine joint states
    obs = torch.cat([joint_pos, joint_vel], dim=-1)
    
    return obs


def cube_observation(
    env,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """
    Collect observations from the cube (position and orientation).

    Args:
        env: The RL environment.
        asset_cfg: The scene entity configuration for the cube.

    Returns:
        Tensor of shape (num_envs, 7) containing cube position and orientation.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # Get position (3 dims) and orientation as quaternion (4 dims)
    position = asset.data.root_pos_w
    orientation = asset.data.root_quat_w
    
    # Combine position and orientation
    obs = torch.cat([position, orientation], dim=-1)
    
    return obs


def image_observation(
    env,
    cameras: list,
) -> torch.Tensor:
    """
    Collect image observations from cameras.

    Args:
        env: The RL environment.
        cameras: List of camera names to collect from.

    Returns:
        Tensor of shape (num_envs, height*width*channels).
    """
    images = []
    for cam_name in cameras:
        camera = env.scene[cam_name]
        # Get RGB image data
        image_data = camera.data.output["rgb"]
        # Flatten the image
        flattened = image_data.reshape(image_data.shape[0], -1)
        images.append(flattened)
    
    # Concatenate all camera images
    obs = torch.cat(images, dim=-1)
    
    return obs

def image_features(env, sensor_cfg: SceneEntityCfg, data_type: str = "rgb") -> torch.Tensor:
    """
    Extracts the image tensor from a camera sensor.
    Returns shape: (num_envs, Height, Width, Channels)
    """
    camera = env.scene.sensors[sensor_cfg.name]
    # Clone to prevent PyTorch buffer modification errors
    image_tensor = camera.data.output[data_type].clone()
    return image_tensor