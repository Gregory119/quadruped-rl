from stable_baselines3.common.callbacks import BaseCallback

class LogMeanEpisodeRewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)


    def _on_step(self) -> bool:
        # if the tensorboard directory is set for the policy then a logger will be available here
        if self.logger is None:
            return True

        # todo: speed up by using incremental average with only done episodes
        
        # Mean of all the environment episode rewards and lengths. Note that
        # this does not only consider recently complete episodes for a smoother
        # average. For content of 'info' see
        # isaaclab_rl.sb3.Sb3VecEnvWrapper.process_extras().
        ep_rw = 0.0
        ep_len = 0.0
        num_envs = len(self.locals["infos"])
        for ep_dict in self.locals["infos"]:
            if "episode" in ep_dict:
                ep_rw += ep_dict["episode"]["r"]
                ep_len += ep_dict["episode"]["l"]
            else:
                num_envs -= 1

        if num_envs == 0:
            return True

        ep_rw /= num_envs
        ep_len /= num_envs
        self.logger.record("episode/reward", ep_rw, self.num_timesteps)
        self.logger.record("episode/len", ep_len, self.num_timesteps)
        return True
