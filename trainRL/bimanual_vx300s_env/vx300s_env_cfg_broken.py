#!/usr/bin/env python3
"""
Isaac Lab environment configuration for bimanual VX300s arm manipulation task.

This environment simulates two VX300s robotic arms mounted on a table
with a manipulatable object (cube) and multiple cameras for vision-based tasks.
"""

from dataclasses import MISSING
from typing import List

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, RayCasterCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

from . import mdp


##
# Scene definition
##
@configclass
class BimanualVX300sSceneCfg(InteractiveSceneCfg):
    """Configuration for the Bimanual VX300s Scene with table and cube."""

    # Ground plane
    ground = GroundPlaneCfg(
        prim_path="/World/ground",
        cfg=sim_utils.GroundPlaneCfg(
            size=(10.0, 10.0),
            static_friction=0.5,
            dynamic_friction=0.5,
            static_friction_combine_mode="multiply",
            dynamic_friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            restitution=0.0,
        ),
    )

    # Table (using a simple box or USD model)
    table = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Table",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.0, 0.0, 0.0],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Warehouse/Shelves/ShelfStraight_A_Tall.usd",
            scale=(0.5, 0.5, 0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )

    # Manipulated object (cube)
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.0, 0.0, 0.4],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=sim_utils.CubeCfg(
            size=(0.05, 0.05, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                mass=0.2,
                density=1000.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0),
                metallic=0.0,
                roughness=0.5,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )

    # Left VX300s arm
    left_arm = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/LeftArm",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=[-0.3, 0.0, 0.0],
            rot=[1.0, 0.0, 0.0, 0.0],
            joint_pos={
                ".*": 0.0,  # Default all joints to 0
            },
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/InterbotixRobots/VX300S/vx_300s.usd",
        ),
    )

    # Right VX300s arm
    right_arm = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/RightArm",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=[0.3, 0.0, 0.0],
            rot=[1.0, 0.0, 0.0, 0.0],
            joint_pos={
                ".*": 0.0,  # Default all joints to 0
            },
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/InterbotixRobots/VX300S/vx_300s.usd",
        ),
    )


##
# Environment configuration
##
@configclass
class BimanualVX300sEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Bimanual VX300s RL environment."""

    # Scene
    scene: BimanualVX300sSceneCfg = BimanualVX300sSceneCfg()

    # Basic environment settings
    num_envs: int = 4
    max_episode_length: int = 1000
    num_actions: int = 14  # 7 joints per arm
    num_observations: int = 128
    num_privileged_obs: int = None

    # Simulation settings
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=0.01,
        disable_contact_processing=False,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=0.5,
            dynamic_friction=0.5,
            restitution=0.0,
        ),
    )

    # Event handlers
    events: EventTermCfg = EventTermCfg()

    # Manager-based environment
    observations: dict = MISSING
    rewards: dict = MISSING
    terminations: dict = MISSING
    actions: dict = MISSING

    def __post_init__(self):
        """Post initialization configuration."""
        # Default observation groups
        self.observations = {
            "policy": ObsGroup(
                obs_terms=dict(
                    arm_obs=ObsTerm(
                        func=mdp.arm_observations,
                        params={"asset_cfg": SceneEntityCfg("left_arm")},
                    ),
                    cube_obs=ObsTerm(
                        func=mdp.cube_observation,
                        params={"asset_cfg": SceneEntityCfg("cube")},
                    ),
                ),
            ),
        }

        # Default reward terms
        self.rewards = {
            "reach_reward": RewTerm(
                func=mdp.reach_reward,
                weight=1.0,
                params={"asset_cfg": SceneEntityCfg("cube")},
            ),
            "action_penalty": RewTerm(
                func=mdp.action_penalty,
                weight=0.01,
            ),
        }

        # Default termination conditions
        self.terminations = {
            "episode_timeout": DoneTerm(
                func=mdp.time_out,
                time_out=True,
            ),
        }

        # Default action terms
        self.actions = {
            "arm_action": mdp.ArmActionCfg(),
        }
