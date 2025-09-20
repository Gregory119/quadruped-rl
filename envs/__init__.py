from gymnasium.envs.registration import register

from .go1 import Go1Env


register(id="Go1-v0",
         entry_point="envs.go1:Go1Env",
         max_episode_steps=1000)
