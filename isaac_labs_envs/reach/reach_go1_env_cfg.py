"""
Environment configuration for the Go1 robot to reach a target point with one of its front feet.
"""

from __future__ import annotations

import torch
import math
import isaaclab.sim as sim_utils
import isaac_labs_envs as envs
import isaaclab.envs.mdp as mdp

from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.utils import configclass
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.math import subtract_frame_transforms

# pre-defined configs
from isaaclab_assets import UNITREE_GO1_CFG

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@configclass
class Go1SceneCfg(InteractiveSceneCfg):
    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    # lights
    dome_light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)))
    # articulation
    robot: ArticulationCfg = UNITREE_GO1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    ground_pad = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ground_pad",
        spawn=sim_utils.CuboidCfg(
            size=(5, 5, 0.001),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            activate_contact_sensors=True,
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0., 0., 0.)),
    )    
    
    # sensors
    contact_sensors = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*",
                                       debug_vis=True,
                                       history_length=4) # same as env decimation

    # As described here
    # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.sensors.html#isaaclab.sensors.ContactSensor,
    # a contact sensor can use filter_prim_paths_expr to filter against names of
    # bodies of interest that the sensor makes contact with. This body name
    # filtered data can only be accessed through
    # contact_sensor.data.force_matrix*. It only supports a contact sensor
    # containing one body which can come into contact with many environment
    # bodies. Instead of creating one sensor per robot part, it is simpler to
    # create a single ground sensor to then filter against robot body
    # parts. Another example is at
    # https://isaac-sim.github.io/IsaacLab/main/source/overview/core-concepts/sensors/contact_sensor.html.
    ground_contact_sensors = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/ground_pad",
        debug_vis=True,
        history_length=4, # same as env
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/FL_hip",
                                "{ENV_REGEX_NS}/Robot/FR_hip",
                                "{ENV_REGEX_NS}/Robot/RL_hip",
                                "{ENV_REGEX_NS}/Robot/RR_hip",
                                "{ENV_REGEX_NS}/Robot/FL_thigh",
                                "{ENV_REGEX_NS}/Robot/FR_thigh",
                                "{ENV_REGEX_NS}/Robot/RL_thigh",
                                "{ENV_REGEX_NS}/Robot/RR_thigh",
                                "{ENV_REGEX_NS}/Robot/FL_calf",
                                "{ENV_REGEX_NS}/Robot/FR_calf",
                                "{ENV_REGEX_NS}/Robot/RL_calf",
                                "{ENV_REGEX_NS}/Robot/RR_calf",
                                "{ENV_REGEX_NS}/Robot/trunk"]
    )
    

@configclass
class ActionsCfg:
    joint_positions = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], use_default_offset=True)


@configclass
class ObservationsCfg:
    """Observation specifications for the environment."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)

        # joint positions and velocities relative to the default values
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

        # gravity vector in the base frame
        base_gravity = ObsTerm(func=mdp.projected_gravity)

        # robot base height relative to world frame, expressed in the world frame
        #base_pos_z = ObsTerm(func=mdp.base_pos_z)

        # robot base pose in the environment frame
        base_pose = ObsTerm(func=mdp.body_pose_w)

        # linear velocity of the base expressed in the base frame
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)

        # angular velocity of the base expressed in the base frame
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)

        # foot pos command
        foot_pos_command = ObsTerm(func=mdp.generated_commands,
                                   params={"command_name": "right_foot_pos"})

        # base/trunk height command
        base_height_command = ObsTerm(func=mdp.generated_commands,
                                      params={"command_name": "height"})

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


# helper function for calculating the reward for foot tracking
def track_foot_exp(env: ManagerBasedRLEnv,
                   var: float,
                   foot_body_name="FR_foot",
                   command_name="right_foot_pos"):
    assert(var >= 0.0)
    # get foot target in base frame (Tbg)
    pos_goal_b = env.command_manager.get_command(command_name)

    # get foot body id/index
    robot = env.scene["robot"]
    body_ids, _ = robot.find_bodies(foot_body_name)
    assert(len(body_ids)==1)
    body_idx = body_ids[0]

    # current foot pos in world frame (Twf)
    pos_foot_w = robot.data.body_pos_w[:, body_idx]

    # transform current foot pos into robot base frame
    pose_base_w = robot.data.root_pose_w # Twb
    # p_bf = Rwb^{-1} p_wf + p_bw
    pos_foot_b, _ = subtract_frame_transforms(
        pose_base_w[:,:3],
        pose_base_w[:,3:],
        pos_foot_w,
        None,
    )

    # position error
    pos_error = pos_foot_b - pos_goal_b

    # calculate reward
    return torch.exp(-torch.norm(pos_error, dim=1) / var)


def track_height_exp(env: ManagerBasedRLEnv,
                     var: float,
                     body_name="trunk",
                     command_name="height") -> torch.Tensor:
    assert(var >= 0.0)
    # get height goal in environment frames
    height_cmd = env.command_manager.get_command(command_name)
    height_goal = torch.zeros((len(height_cmd), 3), device=env.device)
    height_goal[:,2] = height_cmd

    # get body id/index
    robot = env.scene["robot"]
    body_ids, _ = robot.find_bodies(body_name)
    assert(len(body_ids)==1)
    body_idx = body_ids[0]

    # current height in world/environment frames
    height = robot.data.body_pos_w[:, body_idx]

    # error
    error = height_goal - height

    # calculate reward
    return torch.exp(-torch.norm(error, dim=1) / var)


@configclass
class RewardsCfg:
    foot_tracking = RewTerm(func=track_foot_exp, weight=0.1, params={"var": 1.0/3.0})
    collisions = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.1,
        params={"threshold": 0.1,
                "sensor_cfg": SceneEntityCfg("contact_sensors",
                                             body_names=[".*_hip", ".*_thigh", ".*_calf", "trunk"])})
    height_tracking = RewTerm(func=track_foot_exp, weight=0.9, params={"var": 1.0/3.0})
    

def illegal_contact_filtered(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Terminate when the contact force between the sensor and filtered body
    names exceeds the force threshold.

    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # Extract the force matrix only available for filtered body names. Shape of
    # force matrix w history: (num_envs, history_length, num_bodies,
    # num_filters, 3)
    forces = contact_sensor.data.force_matrix_w_history[:, :, sensor_cfg.body_ids]
    shape = forces.shape
    assert len(shape) == 5
    forces = forces.reshape((shape[0], shape[1], shape[2]*shape[3], 3)) # combine num_bodies and num_filters
    # check if any contact force exceeds the threshold
    return torch.any(
        torch.max(torch.norm(forces, dim=-1), dim=1)[0] > threshold, dim=1
    )
    

@configclass
class TerminationCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    fall = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": math.pi/2})
    # terminate if the trunk body collides with anything (eg. legs hitting trunk)
    collision_base = DoneTerm(
        func=mdp.illegal_contact,
        params={'threshold': 0.1,
                'sensor_cfg': SceneEntityCfg("contact_sensors", body_names="trunk")})
    # Terminate if anything other than the feet collide with the ground. This
    # avoids the robot trying to rest a knee on the ground.
    collision_ground = DoneTerm(
        func=illegal_contact_filtered,
        params={'threshold': 0.1,
                'sensor_cfg': SceneEntityCfg("ground_contact_sensors",
                                             body_names="ground_pad"),
                },
    )


@configclass
class CommandsCfg:
    # Position commands are generated in the environment frame and represented
    # in the base frame of the robot
    right_foot_pos = envs.UniformEnvPosCommandCfg(
        asset_name = "robot",
        body_name = "FR_foot",
        resampling_time_range = (5.0, 5.0),
        debug_vis = True,
        ranges = envs.UniformEnvPosCommandCfg.Ranges(
            pos_x = (0.4, 0.4),
            pos_y = (-0.15, -0.15),
            pos_z = (0.2, 0.2),
        )
    )

    height = envs.UniformHeightCommandCfg(
        asset_name = "robot",
        body_name = "trunk",
        resampling_time_range = (5.0, 5.0),
        debug_vis = True,
        range_height = (0.3, 0.3),
    )
    
    
@configclass
class ReachGo1EnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: Go1SceneCfg = Go1SceneCfg(num_envs=3, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # leave events to reset to default state (don't set 'events')
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationCfg = TerminationCfg()
    commands: CommandsCfg = CommandsCfg()

    def __post_init__(self):
        """Post initialization."""
        # viewer settings
        self.viewer.eye = [4.5, 0.0, 6.0]
        self.viewer.lookat = [0.0, 0.0, 2.0]
        # step settings
        self.decimation = 4  # env step every 4 sim steps: 200Hz / 4 = 50Hz
        self.episode_length_s = 5
        # simulation settings
        self.sim.dt = 0.005  # sim step every 5ms: 200Hz

        # By default the render interval is 1, which means rendering will occur
        # at every simulation step. However, video recording expects an FPS the
        # same as the action step rate. This means that rendering is running
        # faster than necessary, so set it to match the expected FPS.
        self.sim.render_interval = self.decimation
