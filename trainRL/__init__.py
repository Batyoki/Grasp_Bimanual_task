#!/usr/bin/env python3
"""
trainRL package - Bimanual VX300s RL training in IsaacLab.

This package provides a complete reinforcement learning training framework
for controlling bimanual robotic arms in IsaacLab.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from .bimanual_vx300s_env import BimanualVX300sEnvCfg
from .utils import ConfigManager, DirectoryManager, MetricsTracker, TelemetryLogger

__all__ = [
    "BimanualVX300sEnvCfg",
    "ConfigManager",
    "DirectoryManager",
    "MetricsTracker",
    "TelemetryLogger",
]
