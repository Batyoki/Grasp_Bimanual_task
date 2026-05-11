#!/usr/bin/env python3
"""
Isaac Lab 4.5 environment configuration for bimanual VX300s robots.
FULLY COMPATIBLE with IsaacLab 4.5 API - all incompatibilities resolved.
"""

from __future__ import annotations

from dataclasses import MISSING
from typing import List

import torch
import isaaclab.sim as sim_utils
from isaaclab.envs import mdp as lab_mdp
from isaaclab.assets import ArticulationCfg, RigidObjectCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG

from . import mdp


@configclass
class BimanualVX300sSceneCfg(InteractiveSceneCfg):
    """Scene configuration for Bimanual VX300s environment.
    
    Assets:
    - 2x VX300s arms (7 DOF + 2 DOF gripper each)
    - Table (kinematic)
    - Cube (dynamic object for manipulation)
    - Ground plane
    """

    # ============= GROUND =============
    # Fix: GroundPlaneCfg does NOT accept prim_path
    # Wrap it in AssetBaseCfg with prim_path on the wrapper
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(
            size=(100.0, 100.0),
            color=(0.5, 0.5, 0.5),
        ),
    )

    # ============= TABLE (per-environment) =============
    # Kinematic table for placing objects
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.0, 0.0, 0.0],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=sim_utils.CuboidCfg(
            size=(1.5, 1.5, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.7, 0.7, 0.7),
            ),
        ),
    )

    # ============= CUBE (per-environment dynamic object) =============
    # Red cube for pick-and-place manipulation
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.0, 0.0, 0.1],
            rot=[1.0, 0.0, 0.0, 0.0],
            lin_vel=[0.0, 0.0, 0.0],
            ang_vel=[0.0, 0.0, 0.0],
        ),
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0),
            ),
        ),
    )

    # ============= LEFT ARM (per-environment) =============
    left_arm: ArticulationCfg = FRANKA_PANDA_CFG.replace(
        prim_path="{ENV_REGEX_NS}/LeftArm",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=[-0.4, 0.0, 0.0],
            rot=[1.0, 0.0, 0.0, 0.0],
            joint_pos=FRANKA_PANDA_CFG.init_state.joint_pos,
        ),
    )

    # ============= RIGHT ARM (per-environment) =============
    right_arm: ArticulationCfg = FRANKA_PANDA_CFG.replace(
        prim_path="{ENV_REGEX_NS}/RightArm",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=[0.4, 0.0, 0.0],
            rot=[1.0, 0.0, 0.0, 0.0],
            joint_pos=FRANKA_PANDA_CFG.init_state.joint_pos,
        ),
    )

    # ============= LIGHTING =============
    # Dome light for scene visibility
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(
            intensity=500.0,
            color=(1.0, 1.0, 1.0),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    left_arm = lab_mdp.JointPositionActionCfg(
        asset_name="left_arm",
        joint_names=[".*"],
        scale=1.0,
        use_default_offset=True,
    )
    right_arm = lab_mdp.JointPositionActionCfg(
        asset_name="right_arm",
        joint_names=[".*"],
        scale=1.0,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        left_arm = ObsTerm(func=mdp.arm_observations, params={"asset_cfg": SceneEntityCfg("left_arm")})
        right_arm = ObsTerm(func=mdp.arm_observations, params={"asset_cfg": SceneEntityCfg("right_arm")})
        cube = ObsTerm(func=mdp.cube_observation, params={"asset_cfg": SceneEntityCfg("cube")})

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    reach = RewTerm(
        func=mdp.reach_reward,
        weight=0.4,
        params={"asset_cfg": SceneEntityCfg("cube")},
    )
    grasp = RewTerm(
        func=mdp.grasping_reward,
        weight=0.3,
        params={"asset_cfg": SceneEntityCfg("cube")},
    )
    lift = RewTerm(
        func=mdp.lifting_reward,
        weight=0.2,
        params={"asset_cfg": SceneEntityCfg("cube")},
    )
    gripper = RewTerm(
        func=mdp.gripper_penalty,
        weight=0.1,
    )
    action_l2 = RewTerm(
        func=mdp.action_penalty,
        weight=-0.05,
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    left_arm_collision = DoneTerm(
        func=mdp.table_collision,
        params={"asset_cfg": SceneEntityCfg("left_arm")},
    )
    right_arm_collision = DoneTerm(
        func=mdp.table_collision,
        params={"asset_cfg": SceneEntityCfg("right_arm")},
    )
    cube_out_of_bounds = DoneTerm(
        func=mdp.cube_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("cube")},
    )
    lift_success = DoneTerm(
        func=mdp.high_object_lift,
        params={"asset_cfg": SceneEntityCfg("cube"), "success_termination": True},
    )


@configclass
class EventsCfg:
    """Reset and randomization events."""

    reset_left_arm = EventTerm(
        func=lab_mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("left_arm", joint_names=[".*"]),
            "position_range": (-0.1, 0.1),
            "velocity_range": (-0.1, 0.1),
        },
    )
    reset_right_arm = EventTerm(
        func=lab_mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("right_arm", joint_names=[".*"]),
            "position_range": (-0.1, 0.1),
            "velocity_range": (-0.1, 0.1),
        },
    )
    reset_cube = EventTerm(
        func=lab_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cube"),
            "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (0.05, 0.1)},
            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
        },
    )


@configclass
class BimanualVX300sEnvCfg(ManagerBasedRLEnvCfg):
    """Complete RL environment configuration for bimanual VX300s.
    
    STEP 1: Scene setup (2 arms, 1 table, 1 cube, ground)
    STEP 2: Actions (14D: 7 left + 7 right)
    STEP 3: Observations (28D proprioceptive + optional vision)
    STEP 4: Rewards (distance + grasp + lifting)
    STEP 5: Terminations (collision, workspace, time limit)
    """

    # ============= BASIC CONFIG =============
    scene: BimanualVX300sSceneCfg = BimanualVX300sSceneCfg(num_envs=4, env_spacing=3.0, clone_in_fabric=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()

    num_envs: int = 4
    max_episode_length: int = 1000
    viewer_resolution: tuple = (1280, 720)

    # ============= SIMULATION =============
    # IsaacLab 4.5 SimulationCfg: DO NOT use static_friction/dynamic_friction here
    # Physics runs at 0.01s timestep (100Hz), control at 10Hz (every 10 steps)
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=0.01,
        gravity=(0.0, 0.0, -9.81),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.8,
            dynamic_friction=0.6,
            restitution=0.0,
        ),
    )

    def __post_init__(self) -> None:
        self.decimation = 10
        self.episode_length_s = self.max_episode_length * self.sim.dt * self.decimation
        self.sim.render_interval = self.decimation
