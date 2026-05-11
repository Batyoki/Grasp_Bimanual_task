"""Action space configuration for the bimanual VX300s environment."""

from isaaclab.envs import mdp
from isaaclab.utils import configclass


@configclass
class ArmActionCfg(mdp.JointPositionActionCfg):
    """Joint position action config for a single arm."""

    asset_name: str = "left_arm"
    joint_names: list[str] = [".*"]
    scale: float = 1.0
    use_default_offset: bool = True
