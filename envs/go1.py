from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

import numpy as np
from numpy.typing import NDArray
import time
from os import path


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": 1.5,
}

class Go1Env(MujocoEnv):

    metadata = {
        "render_modes": [
        "human",
        ],}

    def __init__(self,
                 xml_file: str,
                 rate_hz: int = 50,
                 **kwargs):
        
        self.rate_hz = rate_hz

        if not path.exists(xml_file):
            raise FileNotFoundError(f"Mujoco model not found: {xml_file}")

        observation_space = Box(low=-np.inf, high=np.inf, shape=(19,), dtype=np.float64)

        # todo: set mujoco simulation timestep (assumed for now)
        mj_timestep = 0.001
        env_timestep = 1 / rate_hz
        frame_skip = int(env_timestep // mj_timestep)
        MujocoEnv.__init__(self,
                           model_path=xml_file,
                           frame_skip=frame_skip,
                           observation_space=observation_space,
                           default_camera_config=DEFAULT_CAMERA_CONFIG,
                           **kwargs)

        self.prev_step_ts_ns = None


    def step(self, action):
        # If rendering, ensure that step is not called faster than the desired
        # rate. This is not needed when not rendering because the simulation is
        # stepped by the appropriate number of skip frames, so step() should be
        # called as fast as possible.
        if self.render_mode == "human":
            self._step_sleep(display_rate=True)
        self.do_simulation(action, self.frame_skip)

        # visualization
        if self.render_mode == "human":
            self.render()

        reward = 0
        truncated = False
        terminated = False
        obs = self.data.qpos
        info = {}

        return obs, reward, terminated, truncated, info

        
    def _step_sleep(self, display_rate=False):
        """Call this within step() to sleep the required amount to meet the
        desired step rate. This of course cannot take time away to speed up the
        actual step rate."""
        step_ts_ns = time.perf_counter_ns()
        if self.prev_step_ts_ns is not None:
            dur = (step_ts_ns - self.prev_step_ts_ns)*1e-9
            desired_dur = 1 / self.rate_hz
            dur_diff = desired_dur - dur
            if dur_diff > 0:
                time.sleep(dur_diff)

        # display the measured rate
        if display_rate and self.prev_step_ts_ns is not None:
            step_ts_ns = time.perf_counter_ns()
            dur = (step_ts_ns - self.prev_step_ts_ns)*1e-9
            actual_rate = 1 / dur
            # note that the actual rate cannot go faster the 60 Hz because
            # that's the limit of the mujoco renderer and probably the
            # physical monitor limit
            print("measured rate [Hz]: {}".format(actual_rate))

        # step() is only being performed at this point and a sleep might have
        # occurred in the above logic, so update the previous step timestep
        # accordingly
        self.prev_step_ts_ns = time.perf_counter_ns()


    def get_obs():
        # actuated joint positions and velocities
        qpos = np.zeros((12,))
        qvel = np.zeros((12,))
        for i in range(self.model.nu):
            id = self.model.actuator(i).id
            qpos[i] = self.data.qpos[id]
            qvel[i] = self.data.qvel[id]

        return np.concatenate((qpos,qvel))

        
    # override
    def reset_model(self) -> NDArray[np.float64]:
        """
        Reset the robot state.
        """
        self.set_state(qpos=self.init_qpos, qvel=self.init_qvel)
        return self.get_obs()
