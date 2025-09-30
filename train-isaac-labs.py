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
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs import ManagerBasedRLEnv

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
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    alive = RewTerm(func=mdp.is_alive, weight=1.0)


@configclass
class TerminationCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


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
