"""Sub-module containing command generators for pose tracking."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error, quat_from_euler_xyz, quat_unique, pose_inv, make_pose, unmake_pose, quat_from_matrix, matrix_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg import UniformEnvPoseCommandCfg


class UniformEnvPoseCommand(CommandTerm):
    """Command generator for generating pose commands uniformly in the
    environment frame, which is assumed to have the same orientation as the
    simulation world frame.

    The command generator generates poses by sampling positions uniformly within specified
    regions in cartesian space. For orientation, it samples uniformly the euler angles
    (roll-pitch-yaw) and converts them into quaternion representation (w, x, y, z).

    The position and orientation commands are generated in the environment frame
    of the robot, and then represented in the base frame of the robot.

    .. caution::

        Sampling orientations uniformly is not strictly the same as sampling euler angles uniformly.
        This is because rotations are defined by 3D non-Euclidean space, and the mapping
        from euler angles to rotations is not one-to-one.

    """

    cfg: UniformEnvPoseCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: UniformEnvPoseCommandCfg, env: ManagerBasedEnv):
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
        # -- commands: (x, y, z, qw, qx, qy, qz) in root frame
        self.pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_b[:, 3] = 1.0 # set qw=1
        self.pose_command_e = torch.zeros_like(self.pose_command_b)
        self.pose_command_w = torch.zeros_like(self.pose_command_b)
        # -- metrics
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "UniformEnvPoseCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired pose command. Shape is (num_envs, 7).

        The first three elements correspond to the position, followed by the quaternion orientation in (w, x, y, z).
        """
        return self.pose_command_b

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # compute the error
        pos_error, rot_error = compute_pose_error(
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:],
            self.robot.data.body_pos_w[:, self.body_idx],
            self.robot.data.body_quat_w[:, self.body_idx],
        )
        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)
        self.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)

    def _resample_command(self, env_ids: Sequence[int]):
        # sample new pose targets in the world frame
        # -- position
        r = torch.empty(len(env_ids), device=self.device)
        self.pose_command_e[env_ids, 0] = r.uniform_(*self.cfg.ranges.pos_x)
        self.pose_command_e[env_ids, 1] = r.uniform_(*self.cfg.ranges.pos_y)
        self.pose_command_e[env_ids, 2] = r.uniform_(*self.cfg.ranges.pos_z)
        # -- orientation
        euler_angles = torch.zeros_like(self.pose_command_e[env_ids, :3])
        euler_angles[:, 0].uniform_(*self.cfg.ranges.roll)
        euler_angles[:, 1].uniform_(*self.cfg.ranges.pitch)
        euler_angles[:, 2].uniform_(*self.cfg.ranges.yaw)
        quat = quat_from_euler_xyz(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
        # make sure the quaternion has real part as positive
        self.pose_command_e[env_ids, 3:] = quat_unique(quat) if self.cfg.make_quat_unique else quat

        # find Twe (environment frame w.r.t world frame with the same orientation)
        pos_we = self._env.scene.env_origins[env_ids]
        quat_we = torch.zeros(len(env_ids), 4, device=self.device)
        quat_we[:, 0] = 1.0 # qw=1.0 (same orientation as world)
        
        # also represent command in world frame for visualization
        self.pose_command_w[env_ids, :3], self.pose_command_w[env_ids, 3:] = combine_frame_transforms(
            pos_we,
            quat_we,
            self.pose_command_e[env_ids, :3],
            self.pose_command_e[env_ids, 3:]
        )
        
        # transform command from environment origin frame into base frame
        # Tbc = Tbe * Tec
        # where
        # Tbc: command in base frame
        # Tbe: environment frame w.r.t base frame
        # Tec: command in environment frame

        # find Twe (environment frame w.r.t world frame with the same orientation)
        Twe = torch.zeros(len(env_ids), 4, 4, device=self.device)
        Twe[:, :4, :4] = torch.eye(4) # same orientation as world
        Twe[:, :3, 3] = self._env.scene.env_origins[env_ids] # shape=(num_envs, 3)
        
        # find Twb
        pose_wb = self.robot.data.root_pose_w[env_ids] # [[pos, quat]], shape=(num_envs, 7)
        R_wb = matrix_from_quat(pose_wb[:,3:])
        Twb = make_pose(pos=pose_wb[:,:3], rot=R_wb) # shape=(num_envs, (4,4))
        
        # find Tbe = (Twb)^{-1} Twe
        Tbe = torch.matmul(pose_inv(Twb), Twe) # shape=(num_envs, (4,4))
        pos_be, R_be = unmake_pose(Tbe)
        quat_be = quat_from_matrix(R_be)
        # find Tbc
        self.pose_command_b[env_ids, :3], self.pose_command_b[env_ids, 3:] = combine_frame_transforms(
            pos_be,
            quat_be,
            self.pose_command_e[env_ids, :3],
            self.pose_command_e[env_ids, 3:],
        )
        

    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first time
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                # -- goal pose
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
                # -- current body pose
                self.current_pose_visualizer = VisualizationMarkers(self.cfg.current_pose_visualizer_cfg)
            # set their visibility to true
            self.goal_pose_visualizer.set_visibility(True)
            self.current_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
                self.current_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # update the markers
        # -- goal pose
        self.goal_pose_visualizer.visualize(self.pose_command_w[:, :3], self.pose_command_w[:, 3:])
        # -- current body pose
        body_link_pose_w = self.robot.data.body_link_pose_w[:, self.body_idx]
        self.current_pose_visualizer.visualize(body_link_pose_w[:, :3], body_link_pose_w[:, 3:7])
