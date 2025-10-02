"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Test RL environment for Go1.")
parser.add_argument("--num_envs", type=int, default=3, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.utils import configclass
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg

import isaaclab.envs.mdp as mdp
import isaac_labs_envs as envs
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.utils.math import subtract_frame_transforms


# pre-defined configs
from isaaclab_assets import UNITREE_GO1_CFG


@configclass
class Go1SceneCfg(InteractiveSceneCfg):
    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    # lights
    dome_light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)))
    # articulation
    robot: ArticulationCfg = UNITREE_GO1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class ActionsCfg:
    joint_positions = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], use_default_offset=True)


@configclass
class ObservationsCfg:
    """Observation specifications for the environment."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)

        # joint positions and velocities relative to the default values
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

        # gravity vector in the base frame
        base_gravity = ObsTerm(func=mdp.projected_gravity)

        # robot base height relative to world frame, expressed in the world frame
        base_pos_z = ObsTerm(func=mdp.base_pos_z)

        # linear velocity of the base expressed in the base frame
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)

        # angular velocity of the base expressed in the base frame
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)

        # foot pose command
        foot_pose_command = ObsTerm(func=mdp.generated_commands,
                                    params={"command_name": "right_foot_pose"})

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


# helper function for calculating the reward for foot tracking
def track_foot_exp(env: ManagerBasedRLEnv,
                   var: float,
                   foot_body_name="FR_foot",
                   command_name="right_foot_pose"):
    assert(var >= 0.0)
    # get foot target in base frame (Tbg)
    pose_goal_b = env.command_manager.get_command(command_name)

    # get foot body id/index
    robot = env.scene["robot"]
    body_ids, _ = robot.find_bodies(foot_body_name)
    assert(len(body_ids)==1)
    body_idx = body_ids[0]

    # current foot pose in world frame (Twf)
    pos_foot_w = robot.data.body_pos_w[:, body_idx]
    quat_foot_w = robot.data.body_quat_w[:, body_idx]

    # transform current foot pose into robot base frame
    pose_base_w = robot.data.root_pose_w # Twb
    # Tbf = Twb^{-1} Twf
    pos_foot_b, quat_foot_b = subtract_frame_transforms(
        pose_base_w[:,:3],
        pose_base_w[:,3:],
        pos_foot_w,
        quat_foot_w,
    )

    # position error
    pos_error = pos_foot_b - pose_goal_b[:,:3]

    # calculate reward
    return torch.exp(-torch.norm(pos_error, dim=1) / var)


@configclass
class RewardsCfg:
    foot_tracking = RewTerm(func=track_foot_exp, weight=1.0, params={"var": 0.6})


@configclass
class TerminationCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class CommandsCfg:
    # Pose commands are generated in the environment frame and represented in
    # the base frame of the robot
    right_foot_pose = envs.UniformEnvPoseCommandCfg(
        asset_name = "robot",
        body_name = "FR_foot",
        resampling_time_range = (5.0, 5.0),
        debug_vis = True,
        ranges = mdp.UniformPoseCommandCfg.Ranges(
            pos_x = (0.4, 0.4),
            pos_y = (-0.15, -0.15),
            pos_z = (0.2, 0.2),
            roll = (0.0, 0.0),
            pitch = (0.0, 0.0),
            yaw = (0.0, 0.0),
        )
    )
    
    
@configclass
class Go1EnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: Go1SceneCfg = Go1SceneCfg(num_envs=3, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # leave events to reset to default state (don't set 'events')
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationCfg = TerminationCfg()
    commands: CommandsCfg = CommandsCfg()

    def __post_init__(self):
        """Post initialization."""
        # viewer settings
        self.viewer.eye = [4.5, 0.0, 6.0]
        self.viewer.lookat = [0.0, 0.0, 2.0]
        # step settings
        self.decimation = 4  # env step every 4 sim steps: 200Hz / 4 = 50Hz
        self.episode_length_s = 5
        # simulation settings
        self.sim.dt = 0.005  # sim step every 5ms: 200Hz


def main():
    # create environment configuration
    env_cfg = Go1EnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    # setup RL environment
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # simulate physics
    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # sample random actions
            joint_positions = torch.randn_like(env.action_manager.action)
            # step the environment
            obs, rew, terminated, truncated, info = env.step(joint_positions)

    # close the environment
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
