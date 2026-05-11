"""
Bimanual VX300s environment for IsaacLab.

This package contains the configuration and utilities for training bimanual
robotic arms (VX300s) to perform manipulation tasks in IsaacLab.
"""

from .vx300s_env_cfg import BimanualVX300sEnvCfg
from .mdp import *

__all__ = ["BimanualVX300sEnvCfg"]
