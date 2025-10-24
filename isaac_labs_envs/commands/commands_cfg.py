from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import SPHERE_MARKER_CFG, FRAME_MARKER_CFG
from isaaclab.utils import configclass

from .pose_command import UniformEnvPosCommand
from .height_command import UniformHeightCommand

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
        """Uniform distribution ranges in spherical coordinates for the position
        commands."""

        pos_r: tuple[float, float] = MISSING
        """Range for the radial distance (in m)."""

        pos_theta: tuple[float, float] = MISSING
        """Range for the azimuth angle (in rad)."""

        pos_z: tuple[float, float] = MISSING
        """Range for the z position (in m)."""

    ranges: Ranges = MISSING
    """Ranges for the commands."""

    offset_sample_xyz: tuple[float, float, float] = MISSING
    """Optional (x,y,z) offset of sampling origin with respect to the
    environment frame."""

    goal_pos_visualizer_cfg: VisualizationMarkersCfg = SPHERE_MARKER_CFG.replace(prim_path="/Visuals/Command/goal_pos")
    """The configuration for the goal pos visualization marker. Defaults to
    SPHERE_MARKER_CFG."""

    current_pos_visualizer_cfg: VisualizationMarkersCfg = SPHERE_MARKER_CFG.replace(
        prim_path="/Visuals/Command/body_pos"
    )
    """The configuration for the current pos visualization marker. Defaults to
    SPHERE_MARKER_CFG."""


@configclass
class UniformHeightCommandCfg(CommandTermCfg):
    """Configuration for uniform height command generator."""

    class_type: type = UniformHeightCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    body_name: str = MISSING
    """Name of the body in the asset for which the commands are generated."""

    range_height: tuple[float, float] = MISSING
    """Range for the z (height) position (in m)."""

    goal_height_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/goal_height")
    """The configuration for the goal pos visualization marker. Defaults to FRAME_MARKER_CFG."""

    current_height_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/body_height"
    )
    """The configuration for the current pos visualization marker. Defaults to FRAME_MARKER_CFG."""

    # Set the scale of the visualization markers to (0.1, 0.1, 0.1)
    goal_height_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    current_height_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
