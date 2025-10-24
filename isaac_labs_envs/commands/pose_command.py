"""Sub-module containing command generators for position tracking."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg import UniformEnvPosCommandCfg


class UniformEnvPosCommand(CommandTerm):
    """Command generator for generating position commands uniformly in the
    environment frame, which is assumed to have the same orientation as the
    simulation world frame.

    The command generator generates positions by sampling positions uniformly
    within specified regions in cartesian space.

    The position command is generated in the environment frame of the robot, and
    then represented in the base frame of the robot.

    .. caution::

        Sampling orientations uniformly is not strictly the same as sampling euler angles uniformly.
        This is because rotations are defined by 3D non-Euclidean space, and the mapping
        from euler angles to rotations is not one-to-one.

    """

    cfg: UniformEnvPosCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: UniformEnvPosCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # extract the robot and body index for which the command is generated
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.body_idx = self.robot.find_bodies(cfg.body_name)[0][0]

        # create buffers
        # -- commands: (x, y, z) in root frame
        # positions command in the base frame
        self.pos_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        # pos command in the environment frame
        self.pos_command_e = torch.zeros_like(self.pos_command_b)
        # pos command in the world frame
        self.pos_command_w = torch.zeros_like(self.pos_command_b)
        # Tensor of (4,4) homogeneous transformations representing the
        # environment frame w.r.t the world frame
        self.pose_we = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_we[:, 3] = 1.0 # same orientation as world
        self.pose_we[:, :3] = self._env.scene.env_origins[:] # shape=(num_envs, 3)
        # -- metrics
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "UniformEnvPosCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired position command. Shape is (num_envs, 3).
        """
        return self.pos_command_b

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # compute the error
        pos_error = self.pos_command_w - self.robot.data.body_pos_w[:, self.body_idx]
        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)

    def _resample_command(self, env_ids: Sequence[int]):
        # sample new position targets in the environment frame
        r = torch.empty((len(env_ids),1), device=self.device)
        pos_r = r.uniform_(*self.cfg.ranges.pos_r).clone()
        pos_theta = r.uniform_(*self.cfg.ranges.pos_theta).clone()
        pos_z = r.uniform_(*self.cfg.ranges.pos_z)
        # convert cylindrical coordinates to cartesian coordinates
        # x = r*cos(theta)
        # y = r*sin(theta)
        # z = z
        pos_xyz = torch.cat((pos_r * torch.cos(pos_theta),
                             pos_r * torch.sin(pos_theta),
                             pos_z), dim=1)
        # offset sample in xyz coordinates
        pos_xyz += torch.tensor(self.cfg.offset_sample_xyz, device=self.device)
        
        self.pos_command_e[env_ids, :] = pos_xyz

        # also represent command in world frame for visualization
        # p_w = Rwe*p_e + p_we
        self.pos_command_w[env_ids], _ = combine_frame_transforms(
            self.pose_we[env_ids, :3],
            self.pose_we[env_ids, 3:],
            self.pos_command_e[env_ids],
            None,
        )

    def _update_command(self):
        # The base frame has likely moved with respect to the environment frame
        # so recalculate the command in the base frame.

        # transform command from environment origin frame into base frame
        # Tbc = Tbe * Tec
        # where
        # Tbc: command in base frame
        # Tbe: environment frame w.r.t base frame
        # Tec: command in environment frame
        
        # find Tbe = Twb^{-1} * Twe
        pose_wb = self.robot.data.root_pose_w # [[pos, quat]], shape=(num_envs, 7)
        pos_be, quat_be = subtract_frame_transforms(
            pose_wb[:,:3],
            pose_wb[:,3:],
            self.pose_we[:,:3],
            self.pose_we[:,3:]
        )
        # find position command in robot base frame
        self.pos_command_b[:], _ = combine_frame_transforms(
            pos_be,
            quat_be,
            self.pos_command_e,
            None
        )

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first time
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                # -- goal pos
                self.goal_pos_visualizer = VisualizationMarkers(self.cfg.goal_pos_visualizer_cfg)
                # -- current body pos
                self.current_pos_visualizer = VisualizationMarkers(self.cfg.current_pos_visualizer_cfg)
            # set their visibility to true
            self.goal_pos_visualizer.set_visibility(True)
            self.current_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)
                self.current_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # update the markers
        # -- goal pos
        self.goal_pos_visualizer.visualize(self.pos_command_w)
        # -- current body pose
        body_link_pose_w = self.robot.data.body_link_pose_w[:, self.body_idx]
        self.current_pos_visualizer.visualize(body_link_pose_w[:, :3])
