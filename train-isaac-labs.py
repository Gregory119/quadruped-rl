"""Run as:
python train-isaac-labs.py --task <env. name>

eg.
python train-isaac-labs.py --task Isaac-Reach-Go1-v0
"""

# Launch Isaac Sim Simulator first

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Test RL environment for Go1.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=3, help="Number of environments to spawn.")
parser.add_argument("--task", type=str, default=None, help="Name of task/environment")
parser.add_argument(
    "--agent", type=str, default="sb3_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--env_steps", type=int, default=100000, help="Number of steps per environment instance.")
parser.add_argument(
    "--keep_all_info",
    action="store_true",
    default=False,
    help="Use a slower SB3 wrapper but keep all the extra training info.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse known arguments and keep unknown arguments for hydra parsing
args_cli, unknown_args = parser.parse_known_args()

# Set command line arguments to only contain unknown arguments (to the
# ArgumentParser) to be parsed by hydra. This also avoids hydra trying to parse
# arguments intended for the ArgumentParser.
sys.argv = [sys.argv[0]] + unknown_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch
import gymnasium as gym

import isaac_labs_envs
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_rl.sb3 import Sb3VecEnvWrapper

from stable_baselines3 import PPO

import contextlib


@hydra_task_config(task_name=args_cli.task, agent_cfg_entry_point=args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: dict):
    # override hydra configuration with ArgumentParser arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_steps = args_cli.env_steps
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg["device"] = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # create environment using environment configuration
    env = gym.make(args_cli.task, cfg=env_cfg)

    # wrap environment so that it can be used with stable baselines3
    env = Sb3VecEnvWrapper(env, fast_variant=not args_cli.keep_all_info)

    # create agent/policy using agent configuration
    policy_arch = agent_cfg.pop("policy")
    agent = PPO(policy_arch, env, verbose=1, **agent_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    # reset environment
    env.reset()

    # train agent
    with contextlib.suppress(KeyboardInterrupt):
        agent.learn(
            total_timesteps=env_steps*env_cfg.scene.num_envs,
            progress_bar=True,
        )

    # close the environment
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
