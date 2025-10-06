import gymnasium as gym

gym.register(
    id="Isaac-Reach-Go1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.reach_go1_env_cfg:ReachGo1EnvCfg",
    }
)
