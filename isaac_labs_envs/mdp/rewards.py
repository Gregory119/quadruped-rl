from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.utils.math import subtract_frame_transforms, matrix_from_quat

import torch

def track_height_exp(env: ManagerBasedRLEnv,
                     var: float,
                     body_name="trunk",
                     command_name="height") -> torch.Tensor:
    assert(var >= 0.0)
    # Get height goal in environment frames. Keep the xy coordinates at the
    # origin.
    height_cmd = env.command_manager.get_command(command_name)

    # get body id/index
    robot = env.scene["robot"]
    body_ids, _ = robot.find_bodies(body_name)
    assert(len(body_ids)==1)
    body_idx = body_ids[0]

    # current body position in world frame
    pos_body_w = robot.data.body_pos_w[:, body_idx]

    # error
    error = height_cmd - pos_body_w[:,2].reshape((-1,1))

    # calculate reward
    return torch.exp(-torch.norm(error, dim=1) / var)


# helper function for calculating the reward for foot tracking
def track_foot_exp(env: ManagerBasedRLEnv,
                   var: float,
                   foot_body_name="FR_foot",
                   command_name="right_foot_pos"):
    assert(var >= 0.0)
    # get foot target in base frame (Tbg)
    pos_goal_b = env.command_manager.get_command(command_name)

    # get foot body id/index
    robot = env.scene["robot"]
    body_ids, _ = robot.find_bodies(foot_body_name)
    assert(len(body_ids)==1)
    body_idx = body_ids[0]

    # current foot pos in world frame (Twf)
    pos_foot_w = robot.data.body_pos_w[:, body_idx]

    # transform current foot pos into robot base frame
    pose_base_w = robot.data.root_pose_w # Twb
    # p_bf = Rwb^{-1} p_wf + p_bw
    pos_foot_b, _ = subtract_frame_transforms(
        pose_base_w[:,:3],
        pose_base_w[:,3:],
        pos_foot_w,
        None,
    )

    # position error
    pos_error = pos_foot_b - pos_goal_b

    # calculate reward
    return torch.exp(-torch.norm(pos_error, dim=1) / var)


def stay_at_zero_xy_exp(env: ManagerBasedRLEnv,
                        var: float,
                        body_name="trunk") -> torch.Tensor:
    assert(var >= 0.0)
    # get body id/index
    robot = env.scene["robot"]
    body_ids, _ = robot.find_bodies(body_name)
    assert(len(body_ids)==1)
    body_idx = body_ids[0]

    # current body position in world frame
    pos_body_w = robot.data.body_pos_w[:, body_idx]
    # transform current body pos into environment frame
    pos_we = env.scene.env_origins
    quat_we = torch.zeros((len(pos_we), 4), device=env.device)
    quat_we[:,0] = 1.0

    # p_body_e = Rwe^{-1} pos_body_w + p_we
    pos_body_e, _ = subtract_frame_transforms(
        pos_we,
        quat_we,
        pos_body_w,
        None,
    )

    # xy error
    error = pos_body_e[:,:2]

    # calculate reward
    return torch.exp(-torch.norm(error, dim=1) / var)


def orient_forward(env: ManagerBasedRLEnv,
                   body_name="trunk") -> torch.Tensor:
    # goal pose
    goal_quat_e = torch.tensor([1.0, 0., 0., 0.], device=env.device)
    
    # get body id/index
    robot = env.scene["robot"]
    body_ids, _ = robot.find_bodies(body_name)
    assert(len(body_ids)==1)
    body_idx = body_ids[0]

    # body pose in environment frame
    pose_body_w = robot.data.body_pose_w[:, body_idx]
    pos_we = env.scene.env_origins
    quat_we = torch.zeros((len(pos_we), 4), device=env.device)
    quat_we[:,0] = 1.0
    # p_body_e = Rwe^{-1}*pos_body_w + p_we
    _, quat_body_e = subtract_frame_transforms(
        pos_we,
        quat_we,
        pose_body_w[:,:3],
        pose_body_w[:,3:],
    )

    # get normalized projection of body x axis in environment frame
    Reb = matrix_from_quat(quat_body_e)
    x_eb = Reb[:,:3,0]
    x_proj_e = x_eb[:,:2]
    x_proj_e /= torch.norm(x_proj_e, dim=1).reshape((-1,1))

    # error between normalized projected x and environment x
    error = x_proj_e - torch.tensor([1.0, 0.], device=env.device)

    return torch.exp(-torch.norm(error, dim=1)/0.5)
