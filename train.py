import envs

import gymnasium as gym

from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import PPO

from robot_descriptions import go1_mj_description

from os import path


def main():
    render_mode = 'human'

    env = gym.make("Go1-v0",
                   render_mode=render_mode,
                   xml_file=path.join(go1_mj_description.PACKAGE_PATH, "scene.xml"))
    env.reset()
    
    # venv = SubprocVecEnv([env])
    device = "cpu" # for PPO
    
    model = PPO("MlpPolicy", env, verbose=0, device=device)
    print("Training...")
    model.learn(total_timesteps=1e6)
    print("Training done")
    venv.close()


if __name__ == "__main__":
    main()
