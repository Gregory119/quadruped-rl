from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import SPHERE_MARKER_CFG
from isaaclab.utils import configclass

from .pose_command import UniformEnvPosCommand

@configclass
class UniformEnvPosCommandCfg(CommandTermCfg):
    """Configuration for uniform world position command generator."""

    class_type: type = UniformEnvPosCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    body_name: str = MISSING
    """Name of the body in the asset for which the commands are generated."""

    @configclass
    class Ranges:
        """Uniform distribution ranges for the pose commands."""

        pos_x: tuple[float, float] = MISSING
        """Range for the x position (in m)."""

        pos_y: tuple[float, float] = MISSING
        """Range for the y position (in m)."""

        pos_z: tuple[float, float] = MISSING
        """Range for the z position (in m)."""

    ranges: Ranges = MISSING
    """Ranges for the commands."""

    goal_pos_visualizer_cfg: VisualizationMarkersCfg = SPHERE_MARKER_CFG.replace(prim_path="/Visuals/Command/goal_pos")
    """The configuration for the goal pos visualization marker. Defaults to SPHERE_MARKER_CFG."""

    current_pos_visualizer_cfg: VisualizationMarkersCfg = SPHERE_MARKER_CFG.replace(
        prim_path="/Visuals/Command/body_pos"
    )
    """The configuration for the current pos visualization marker. Defaults to SPHERE_MARKER_CFG."""
