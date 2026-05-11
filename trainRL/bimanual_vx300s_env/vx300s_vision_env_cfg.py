#!/usr/bin/env python3
from __future__ import annotations

import torch
import isaaclab.sim as sim_utils
from isaaclab.envs import mdp as lab_mdp
from isaaclab.assets import ArticulationCfg, RigidObjectCfg, AssetBaseCfg, Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.camera import CameraCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG
from isaaclab.envs import ViewerCfg

# =========================================================================
# CUSTOM MDP FUNCTIONS (Embedded directly here to prevent path errors!)
# =========================================================================

def image_features(env, sensor_cfg: SceneEntityCfg, data_type: str = "rgb") -> torch.Tensor:
    """Extracts image tensor from camera sensor."""
    camera = env.scene.sensors[sensor_cfg.name]
    return camera.data.output[data_type].clone()

def reach_reward(env, asset_cfg: SceneEntityCfg, min_distance: float = 0.05) -> torch.Tensor:
    cube: RigidObject = env.scene[asset_cfg.name]
    robot: Articulation = env.scene["robot"]
    cube_pos = cube.data.root_pos_w
    ee_pos = robot.data.body_pos_w[:, -1, :] 
    distance = torch.norm(ee_pos - cube_pos, dim=-1)
    reach_rew = torch.exp(-2.0 * distance)
    bonus = torch.where(distance < min_distance, torch.ones_like(distance), torch.zeros_like(distance)) * 0.5
    return reach_rew + bonus

def grasping_reward(env, asset_cfg: SceneEntityCfg, grasp_threshold: float = 0.05) -> torch.Tensor:
    cube: RigidObject = env.scene[asset_cfg.name]
    robot: Articulation = env.scene["robot"]
    cube_pos = cube.data.root_pos_w
    ee_pos = robot.data.body_pos_w[:, -1, :]
    distance = torch.norm(ee_pos - cube_pos, dim=-1)
    gripper_pos = robot.data.joint_pos[:, -2:]
    gripper_closed = torch.all(gripper_pos < 0.02, dim=-1)
    cube_vel = torch.norm(cube.data.root_vel_w, dim=-1)
    stable = cube_vel < 0.5
    grasping = (distance < grasp_threshold) & gripper_closed & stable
    return grasping.float() * 2.0

def lifting_reward(env, asset_cfg: SceneEntityCfg, table_height: float = 0.05) -> torch.Tensor:
    cube: RigidObject = env.scene[asset_cfg.name]
    cube_z = cube.data.root_pos_w[:, 2]
    height_above_table = cube_z - table_height
    lift_rew = torch.clamp(height_above_table / 0.5, min=0.0, max=1.0)
    return lift_rew * 3.0

def action_penalty(env) -> torch.Tensor:
    if hasattr(env, 'last_actions') and env.last_actions is not None:
        return -0.05 * torch.norm(env.last_actions, dim=-1)
    return torch.zeros(env.num_envs, device=env.device)


# =========================================================================
# ENVIRONMENT CONFIGURATION
# =========================================================================

@configclass
class VisionVX300sSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0), color=(0.5, 0.5, 0.5)),
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, 0.0], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=sim_utils.CuboidCfg(
            size=(1.5, 1.5, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.7, 0.7)),
        ),
    )

    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.0, 0.0, 0.1], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
    )

    robot = FRANKA_PANDA_CFG.replace(
        prim_path="{ENV_REGEX_NS}/RobotArm",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=[-0.4, 0.0, 0.0],
            rot=[1.0, 0.0, 0.0, 0.0],
            joint_pos=FRANKA_PANDA_CFG.init_state.joint_pos,
        ),
    )

    top_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/TopCamera",
        update_period=0.1,
        height=84, width=84, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, focus_distance=400.0),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 1.5), rot=(0.707, 0.0, 0.707, 0.0)),
    )

    front_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/FrontCamera",
        update_period=0.1,
        height=84, width=84, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, focus_distance=400.0),
        offset=CameraCfg.OffsetCfg(pos=(1.0, 0.0, 0.5), rot=(0.5, -0.5, 0.5, -0.5)),
    )

    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=1000.0, color=(1.0, 1.0, 1.0)),
    )


@configclass
class ActionsCfg:
    robot = lab_mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=1.0, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        robot_state = ObsTerm(func=lab_mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot")})
        # Pointing to the local functions defined at the top of this file!
        top_image = ObsTerm(func=image_features, params={"sensor_cfg": SceneEntityCfg("top_camera")})
        front_image = ObsTerm(func=image_features, params={"sensor_cfg": SceneEntityCfg("front_camera")})

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False 

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    reach = RewTerm(func=reach_reward, weight=1.0, params={"asset_cfg": SceneEntityCfg("cube")})
    grasp = RewTerm(func=grasping_reward, weight=2.0, params={"asset_cfg": SceneEntityCfg("cube")})
    lift = RewTerm(func=lifting_reward, weight=3.0, params={"asset_cfg": SceneEntityCfg("cube")})
    penalty = RewTerm(func=action_penalty, weight=1.0)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=lab_mdp.time_out, time_out=True)


@configclass
class EventsCfg:
    reset_robot = EventTerm(
        func=lab_mdp.reset_joints_by_offset,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]), "position_range": (-0.1, 0.1), "velocity_range": (-0.05, 0.05)},
    )
    reset_cube = EventTerm(
        func=lab_mdp.reset_root_state_uniform,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("cube"), "pose_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "z": (0.05, 0.1)}, "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)}},
    )



@configclass
class VisionVX300sEnvCfg(ManagerBasedRLEnvCfg):
    # CRITICAL FIX 1: Prevent the scene from cloning into the buggy Fabric backend
    scene = VisionVX300sSceneCfg(num_envs=4, env_spacing=3.0, clone_in_fabric=False)
    
    observations = ObservationsCfg()
    actions = ActionsCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()
    events = EventsCfg()
    viewer = ViewerCfg()

    num_envs = 4
    max_episode_length = 500
    
    sim = sim_utils.SimulationCfg(
        dt=0.01, render_interval=10, gravity=(0.0, 0.0, -9.81),
        # CRITICAL FIX 2: Globally disable Fabric to prevent the usdrt.hierarchy crash
        use_fabric=False,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.6)
    )

    def __post_init__(self) -> None:
        self.decimation = 10
        self.episode_length_s = self.max_episode_length * self.sim.dt * self.decimation