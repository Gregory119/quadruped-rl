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

    from .commands_cfg import UniformHeightCommandCfg


class UniformHeightCommand(CommandTerm):
    # """Command generator for generating position commands uniformly in the
    # environment frame, which is assumed to have the same orientation as the
    # simulation world frame.

    # The command generator generates positions by sampling positions uniformly
    # within specified regions in cartesian space.

    # The position command is generated in the environment frame of the robot, and
    # then represented in the base frame of the robot.

    # .. caution::

    #     Sampling orientations uniformly is not strictly the same as sampling euler angles uniformly.
    #     This is because rotations are defined by 3D non-Euclidean space, and the mapping
    #     from euler angles to rotations is not one-to-one.

    # """

    cfg: UniformHeightCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: UniformHeightCommandCfg, env: ManagerBasedEnv):
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
        self.height_command = torch.zeros(self.num_envs, 3, device=self.device)
        # -- metrics
        self.metrics["height_error"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "UniformHeightCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired 1D height command. Shape is (num_envs, 1). The height
        command is scalar so that it adds the least amount of dimension to an
        observation configuration.

        """
        # this is reshaped so that accessing a row gives a tensor that can be
        # concatenated in an observation configuration
        return self.height_command[:,2].reshape((-1,1))

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # compute the error
        height_error = self.height_command_w - self.robot.data.body_pos_w[:, self.body_idx]
        self.metrics["height_error"] = torch.norm(height_error, dim=-1)

    def _resample_command(self, env_ids: Sequence[int]):
        # sample new position targets in the environment frame
        r = torch.empty(len(env_ids), device=self.device)
        self.height_command[env_ids, 2] = r.uniform_(*self.cfg.range_height)
        # height command in the world frame
        self.height_command_w = self.height_command + self._env.scene.env_origins[:]

    def _update_command(self):
        # command remains the same in the world or environment frame as the
        # robot moves
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first time
        if debug_vis:
            if not hasattr(self, "goal_height_visualizer"):
                # -- goal height
                self.goal_height_visualizer = VisualizationMarkers(self.cfg.goal_height_visualizer_cfg)
                # -- current body pos
                self.current_height_visualizer = VisualizationMarkers(self.cfg.current_height_visualizer_cfg)
            # set their visibility to true
            self.goal_height_visualizer.set_visibility(True)
            self.current_height_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_height_visualizer"):
                self.goal_height_visualizer.set_visibility(False)
                self.current_height_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # update the markers
        # -- goal pos
        self.goal_height_visualizer.visualize(self.height_command_w)
        # -- current body pose
        body_link_pose_w = self.robot.data.body_link_pose_w[:, self.body_idx]
        self.current_height_visualizer.visualize(body_link_pose_w[:, :3])
