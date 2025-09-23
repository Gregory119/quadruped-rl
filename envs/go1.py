from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

import mujoco
import numpy as np
from numpy.typing import NDArray
import time
from os import path


def inv_htrans(T: NDArray[np.float64]):
    """Invert a homogeneous transformation."""
    assert(T.shape == (4,4))
    R = T[:3, :3]
    p = T[:3, 3]
    inv_T = np.zeros((4,4))
    inv_T[:3,:3] = np.transpose(R)
    inv_T[:3,3] = -np.transpose(R) @ p
    return inv_T


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

        observation_space = Box(low=-np.inf, high=np.inf, shape=(37,), dtype=np.float64)

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

        # goal position in trunk frame
        Twt = self.get_trunk_frame()
        goal_pos_trunk_homo = inv_htrans(Twt) @ np.concatenate((self.goal_pos_world, [1]))
        self.goal_pos_trunk = goal_pos_trunk_homo[:3]


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
        terminated = self.fallen()
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
        Twt = self.get_trunk_frame()

        # represent current and goal foot position in robot base frame
        pos_foot_trunk_homo = inv_htrans(Twt) @ np.concatenate((pos_foot_world, [1]))
        pos_foot_trunk = pos_foot_trunk_homo[:3]
        
        return np.exp(-np.linalg.norm(pos_foot_trunk - self.goal_pos_trunk) / 0.6)
    
        
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

        # height of robot trunk frame
        h = self.data.body("trunk").xpos[2]

        # gravity vector in trunk body frame
        Twt = self.get_trunk_frame()
        gravity_world = self.model.opt.gravity
        gravity_trunk_h = inv_htrans(Twt) @ np.concatenate((gravity_world, [1]))
        gravity_trunk = gravity_trunk_h[:3]

        # Get the spatial velocity (twist) of the trunk body in the inertial
        # frame of the body (located at the c.o.m) and then transform the
        # spatial velocity to the body frame. The resulting linear velocity
        # component is of the body attached point at the frame origin. Note that
        # the body frame is considered to be instantaneously stationary.
        twist_trunk_i = np.zeros((6,))
        mujoco.mj_objectVelocity(self.model,
                                 self.data,
                                 mujoco.mjtObj.mjOBJ_BODY,
                                 self.model.body("trunk").id,
                                 twist_trunk_i,
                                 True) # use local frame orientation

        twist_trunk_b = np.zeros((6,))
        Twi = self.get_trunk_iframe()
        Tit = inv_htrans(Twi) @ Twt
        mujoco.mju_transformSpatial(twist_trunk_b, # result
                                    twist_trunk_i, # old twist
                                    False, # motion vectors (not force)
                                    Tit[:3,3], # newpos expressed in old frame
                                    np.zeros((3,)), # oldpos expressed in old frame
                                    Tit[:3,:3].ravel()); # rotnew2old new relative to old
        
        omega = twist_trunk_b[:3]
        vel = twist_trunk_b[3:]
        
        return np.concatenate((qpos, qvel, vel, omega, [h], self.goal_pos_trunk, gravity_trunk))


    def goal_pos_in_trunk(self):
        """Get the foot goal position in the trunk body frame."""
        Twt = self.get_trunk_frame()
        goal_pos_trunk_homo = inv_htrans(Twt) @ np.concatenate((self.goal_pos_world, [1]))
        return goal_pos_trunk_homo[:3]
    

    def get_trunk_frame(self):
        """Get pose of trunk body frame w.r.t world frame."""
        Twt = np.identity(4)
        Twt[:3,:3] = self.data.body("trunk").xmat.copy().reshape((3,3))
        Twt[:3,3] = self.data.body("trunk").xpos.copy()
        return Twt
    

    def get_trunk_iframe(self):
        """Get the pose of the trunk inertial frame w.r.t world frame."""
        T = np.identity(4)
        T[:3,:3] = self.data.body("trunk").ximat.copy().reshape((3,3))
        T[:3,3] = self.data.body("trunk").xipos.copy()
        return T

    
    def fallen(self):
        # if any component of the z axis of the trunk frame is directed down,
        # then the robot has fallen
        Twt = self.get_trunk_frame()
        # trunk z-axis represented in the world frame
        z = Twt[:3,2]
        return z[2] < 0.0

        
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
