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
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments to spawn.")
parser.add_argument("--task", type=str, default=None, help="Name of task/environment")
parser.add_argument("--log_interval", type=int, default=100, help="Number of timesteps per environment to log data.")
parser.add_argument(
    "--agent", type=str, default="sb3_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--env_steps", type=int, default=30000, help="Number of steps per environment instance.")
parser.add_argument(
    "--keep_all_info",
    action="store_true",
    default=False,
    help="Use a slower SB3 wrapper but keep all the extra training info.",
)
parser.add_argument("--checkpoint", type=str, default=None, help="Continue the training from checkpoint.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_interval", type=int, default=15, help="Interval between video recordings (in episodes).")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse known arguments and keep unknown arguments for hydra parsing
args_cli, unknown_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    # this is a setting provided by the AppLauncher
    args_cli.enable_cameras = True

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
from isaaclab.utils.dict import print_dict

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, LogEveryNTimesteps
from rl_utils.sb3.log_episode_reward import LogMeanEpisodeRewardCallback

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
    # dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    # dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

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
    env_cfg.seed = agent_cfg["seed"]
    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir


@hydra_task_config(task_name=args_cli.task, agent_cfg_entry_point=args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: dict):
    log_dir = create_log_dir()
    
    post_process_cfg(log_dir=log_dir, env_cfg=env_cfg, agent_cfg=agent_cfg)

    save_run_cfg(log_dir=log_dir, env_cfg=env_cfg, agent_cfg=agent_cfg)
    
    # create environment using environment configuration
    env = gym.make(args_cli.task, 
                   cfg=env_cfg, 
                   render_mode="rgb_array" if args_cli.video else None)

    # setup for video recording    
    dur_p_ep = env_cfg.sim.dt * env_cfg.decimation
    steps_p_ep = env_cfg.episode_length_s / dur_p_ep
    ep_cnt = -1
    def step_trigger_cb(steps):
        # Convert steps to episode count. 'steps' is increment each time step()
        # is called on the isaac env so 'steps' is the accumulate steps per
        # environment.
        nonlocal ep_cnt
        episodes = steps // steps_p_ep
        if episodes <= ep_cnt:
            return False
        ep_cnt = episodes
        return episodes % args_cli.video_interval == 0

    # record duration
    vid_dur_ep = 2
    vid_dur_steps = vid_dur_ep * steps_p_ep

    # Note that episode triggering cannot be used so step triggering is used
    # instead. SB3 vectorized environments are expected to reset individual
    # internal environments automatically, so a call to reset() on the SB3
    # vectorized environment interface does not occur. Episodic video triggering
    # depends on external calls to reset() the environment, and because this
    # does not happen the episodic video trigger does not occur.
    print("Video fps set to: {}".format(env.metadata["render_fps"]))
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": step_trigger_cb,
            "video_length": vid_dur_steps,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap environment so that it can be used with stable baselines3
    env = Sb3VecEnvWrapper(env, fast_variant=not args_cli.keep_all_info)

    # create agent/policy using agent configuration
    minibatch_size_p_env = agent_cfg.pop("minibatch_size_p_env")
    # sb3 PPO uses batch_size as the mini-batch size over all environments (not
    # per environment)
    agent_cfg["batch_size"] = minibatch_size_p_env * env_cfg.scene.num_envs
    policy_arch = agent_cfg.pop("policy")
    agent = PPO(policy_arch, env, tensorboard_log=log_dir, verbose=1, **agent_cfg)
    if args_cli.checkpoint is not None:
        print("Loading policy from checkpoint to continue training.")
        agent = agent.load(args_cli.checkpoint, env, print_system_info=True)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

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
    
    # reset environment
    env.reset()

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
