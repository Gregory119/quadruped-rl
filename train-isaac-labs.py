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
parser.add_argument("--log_interval", type=int, default=100, help="Number of timesteps per environment to log data.")
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
parser.add_argument("--checkpoint", type=str, default=None, help="Continue the training from checkpoint.")

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
from isaaclab_rl.sb3 import Sb3VecEnvWrapper, process_sb3_cfg
from isaaclab.utils.io import dump_pickle, dump_yaml

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, LogEveryNTimesteps

import contextlib
from pathlib import Path
from datetime import datetime
import os


def create_log_dir():
    # directory for logging into
    run_info = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_root_path = os.path.abspath(os.path.join("logs", "sb3", args_cli.task))
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = os.path.join(log_root_path, run_info)
    return log_dir


def save_run_cfg(log_dir: str, env_cfg: ManagerBasedRLEnvCfg, agent_cfg: dict):
    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # save command used to run the script
    command = " ".join(sys.orig_argv)
    (Path(log_dir) / "command.txt").write_text(command)


def post_process_cfg(log_dir, env_cfg, agent_cfg):
    # override hydra configuration with ArgumentParser arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg["device"] = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # post-process agent configuration (convert value strings to types)
    agent_cfg = process_sb3_cfg(agent_cfg, env_cfg.scene.num_envs)
    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir


# todo:
# - video recording
# - normalization
# - create agent from command line checkpoint
# - sb3 callbacks: save checkpoints


@hydra_task_config(task_name=args_cli.task, agent_cfg_entry_point=args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: dict):
    log_dir = create_log_dir()
    
    post_process_cfg(log_dir=log_dir, env_cfg=env_cfg, agent_cfg=agent_cfg)

    save_run_cfg(log_dir=log_dir, env_cfg=env_cfg, agent_cfg=agent_cfg)
    
    # create environment using environment configuration
    env = gym.make(args_cli.task, cfg=env_cfg)

    # wrap environment so that it can be used with stable baselines3
    env = Sb3VecEnvWrapper(env, fast_variant=not args_cli.keep_all_info)

    # create agent/policy using agent configuration
    policy_arch = agent_cfg.pop("policy")
    agent = PPO(policy_arch, env, tensorboard_log=log_dir, verbose=1, **agent_cfg)
    if args_cli.checkpoint is not None:
        print("Loading policy from checkpoint to continue training.")
        agent = agent.load(args_cli.checkpoint, env, print_system_info=True)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    # reset environment
    env.reset()

    # Set save frequency based on the number of policy updates (learning
    # iterations).  Every environment is stepped this many times before saving.
    scale = max(1000 // agent_cfg["n_steps"], 1)
    save_freq_p_env = agent_cfg["n_steps"] * scale
    checkpoint_cb = CheckpointCallback(save_freq=save_freq_p_env,
                                       save_path=log_dir,
                                       name_prefix="model",
                                       save_vecnormalize=True,
                                       save_replay_buffer=False,
                                       verbose=2)
    callbacks = [checkpoint_cb, 
                 LogEveryNTimesteps(n_steps=args_cli.log_interval*env_cfg.scene.num_envs)]
    
    # train agent
    print("Training...")
    with contextlib.suppress(KeyboardInterrupt):
        agent.learn(
            total_timesteps=args_cli.env_steps*env_cfg.scene.num_envs,
            callback=callbacks,
            progress_bar=True,
            log_interval=None,
        )

    # save the final model
    print("Trainging complete. Saving model.")
    agent.save(os.path.join(log_dir, "model"))
        
    # close the environment
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
