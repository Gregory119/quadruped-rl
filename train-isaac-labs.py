"""Run as:
python train-isaac-labs.py --task <env. name>

eg.
python train-isaac-labs.py --task Isaac-Reach-Go1-v0
"""

# Launch Isaac Sim Simulator first

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Test RL environment for Go1.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=3, help="Number of environments to spawn.")
parser.add_argument("--task", type=str, default=None, help="Name of task/environment")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch
import gymnasium as gym

import isaac_labs_envs
from isaaclab_tasks.utils import parse_env_cfg


def main():
    # create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    # reset environment
    env.reset()

    # simulate environment
    while simulation_app.is_running():
        with torch.inference_mode():
            # sample random actions
            joint_positions = torch.randn(env.action_space.shape, device=env.unwrapped.device)
            # step the environment
            obs, rew, terminated, truncated, info = env.step(joint_positions)

    # close the environment
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
