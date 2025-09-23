import envs

import gymnasium as gym

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import PPO

from robot_descriptions import go1_mj_description

from os import path


def main():
    render_mode = 'human'

    venv = make_vec_env("Go1-v0",
                        n_envs=2,
                        env_kwargs={'render_mode': render_mode,
                                    'xml_file': path.join(go1_mj_description.PACKAGE_PATH,
                                                          "scene.xml")},
                        vec_env_cls=SubprocVecEnv,
                        )
    venv.reset()
    
    device = "cpu" # for PPO
    
    model = PPO("MlpPolicy", venv, verbose=0, device=device)
    print("Training...")
    model.learn(total_timesteps=1e6)
    print("Training done")
    venv.close()


if __name__ == "__main__":
    main()
