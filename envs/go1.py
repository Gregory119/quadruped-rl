from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

import mujoco
import numpy as np
from numpy.typing import NDArray
import time
from os import path


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": 3,
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

        observation_space = Box(low=-np.inf, high=np.inf, shape=(24,), dtype=np.float64)

        self.goal_pos_world = np.array([0.4, -0.15, 0.2])

        # todo: set mujoco simulation timestep (assumed for now)
        self.mj_timestep = 0.001
        env_timestep = 1 / rate_hz
        frame_skip = int(env_timestep // self.mj_timestep)
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
            
            # Visualize the site frames. This is a bit of a hack but it works
            # and is simple. This is specifically done after self.render() to
            # ensure that the renderer exists.
            self.mujoco_renderer.viewer.vopt.frame = mujoco.mjtFrame.mjFRAME_SITE

            # enable visualization of the site frames at the feet
            self.mujoco_renderer.viewer.vopt.sitegroup[self.model.site("FR").group] = 1
            

        # reward
        reward = self.get_reward()

        
        # timeout truncation handled externally so set false here
        truncated = False
        terminated = False
        obs = self.get_obs()
        info = {}

        return obs, reward, terminated, truncated, info


    def get_reward(self):
        # foot distance from goal position

        # The goal and foot position must always be in the same frame in order
        # to calculate the norm of the distance between them. Only the desired
        # foot position is an input to the policy, not the current foot
        # position. If the world frame is used as the reference, then it
        # requires the input goal position to also be in this world frame. To
        # make the policy agnostic to the world frame, rather use the base frame
        # of the robot.

        pos_foot_world = self.data.site("FR").xpos
        
        # get pose of trunk frame relative to world frame and invert
        Rwt = self.data.body("trunk").xmat.copy().reshape((3,3))
        pwt = self.data.body("trunk").xpos
        inv_Twt = np.zeros((4,4))
        inv_Twt[:3,:3] = np.transpose(Rwt)
        inv_Twt[:3,3] = -np.transpose(Rwt) @ pwt

        # represent current and goal foot position in robot base frame
        pos_foot_trunk_homo = inv_Twt @ np.concatenate((pos_foot_world, [1]))
        pos_foot_trunk = pos_foot_trunk_homo[:3]
        goal_pos_trunk_homo = inv_Twt @ np.concatenate((self.goal_pos_world, [1]))
        goal_pos_trunk = goal_pos_trunk_homo[:3]
        
        return np.exp(-np.linalg.norm(pos_foot_trunk - goal_pos_trunk) / 0.6)
    
        
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


    def get_obs(self):
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


    # override
    def _initialize_simulation(self) -> tuple["mujoco.MjModel", "mujoco.MjData"]:
        """
        Initialize MuJoCo simulation data structures `mjModel` and `mjData`.
        """
        # load the model specification
        spec = mujoco.MjSpec.from_file(self.fullpath)

        # add sites whose frames will be displayed by default

        # add a site for the goal
        spec.worldbody.add_site(
            pos=self.goal_pos_world,
            quat=[0, 1, 0, 0],
        )

        # compile model and create data
        model = spec.compile()
        model.vis.global_.offwidth = self.width
        model.vis.global_.offheight = self.height
        data = mujoco.MjData(model)

        # set simulation timestep
        model.opt.timestep = self.mj_timestep

        return model, data
