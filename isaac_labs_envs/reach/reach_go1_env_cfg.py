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
from isaaclab.terrains import (
    HfRandomUniformTerrainCfg,
    TerrainImporterCfg,
    TerrainGeneratorCfg
)
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from isaaclab.envs import ManagerBasedRLEnvCfg

# pre-defined configs
from isaaclab_assets import UNITREE_GO1_CFG

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# robot dimensions for reference (folded)
g_length = 0.588
g_width = 0.220
g_height = 0.290
g_height_standing = 0.400
g_height_stand_trunk = g_height_standing - g_height/2
g_max_abs_r = g_height_standing


ROUGH_GROUND_CFG = TerrainGeneratorCfg(
    seed=42,
    size=(1.5, 1.5), # keep this at (1.5, 1.5) for good tuning performance
    border_width=0.,
    # size should fit 4096/4=1024 environments (32x32)
    num_rows=32,
    num_cols=32,
    horizontal_scale=0.05,
    vertical_scale=0.01,
    use_cache=True,
    sub_terrains={
        "height-field": HfRandomUniformTerrainCfg(
            proportion=1.0,
            noise_range=(0.0,0.03),
            noise_step=0.03,
            border_width=0.,
            horizontal_scale=0.05,
            vertical_scale=0.01,
        )
    },
)


@configclass
class Go1SceneCfg(InteractiveSceneCfg):
    # lights
    dome_light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)))
    # articulation
    robot: ArticulationCfg = UNITREE_GO1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_GROUND_CFG,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
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
    # contact_sensor.data.force_matrix*. Apparently it only supports a contact
    # sensor containing one body which can come into contact with many
    # environment bodies, but it seems to work here for the case of many bodies
    # of a contact sensor contacting the single ground terrain mesh
    # body. Another example is at
    # https://isaac-sim.github.io/IsaacLab/main/source/overview/core-concepts/sensors/contact_sensor.html.
    robot_to_ground_contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
        filter_prim_paths_expr=["/World/ground/terrain/mesh"],
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
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_limit_normalized)
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
        # base_height_command = ObsTerm(func=mdp.generated_commands,
        #                               params={"command_name": "height"})

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    foot_tracking = RewTerm(func=envs.mdp.track_foot_exp, weight=1.0, params={"var": g_max_abs_r})
    collisions = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.5,
        params={"threshold": 0.1,
                "sensor_cfg": SceneEntityCfg("contact_sensors",
                                             body_names=[".*_hip", ".*_thigh", ".*_calf", "trunk"])})
    # This helps the learn to lift the foot off of the floor, but it doesn't
    # help it learn to move its foot to the goal position. For that it's
    # probably worth investigating (by looking at training stats) whether an
    # adaptive learning rate algorithm will help. Check the code of the paper.
    # foot_off_floor = RewTerm(
    #     func=mdp.undesired_contacts,
    #     weight=-0.1,
    #     params={"threshold": 0.01,
    #             "sensor_cfg": SceneEntityCfg("contact_sensors",
    #                                          body_names=["FR_foot"])})

    # track_height = RewTerm(func=envs.mdp.track_height_exp, weight=0.25, params={"var": g_max_abs_r})
    # base_pos = RewTerm(func=envs.mdp.stay_at_zero_xy_exp, weight=0.25, params={"var": g_max_abs_r})
    # orient_forward = RewTerm(func=envs.mdp.orient_forward, weight=0.25)
    

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
        params={'threshold': 0.01,
                'sensor_cfg': SceneEntityCfg(
                    "contact_sensors",
                    body_names=["trunk"])})
    # Terminate if anything other than the feet collide with the ground. This
    # avoids the robot trying to rest a knee on the ground.
    collision_ground = DoneTerm(
        func=illegal_contact_filtered,
        params={'threshold': 0.01,
                'sensor_cfg': SceneEntityCfg("robot_to_ground_contact_sensor",
                                             body_names=["FL_hip",
                                                         "FR_hip",
                                                         "RL_hip",
                                                         "RR_hip",
                                                         "FL_thigh",
                                                         "FR_thigh",
                                                         "RL_thigh",
                                                         "RR_thigh",
                                                         "FL_calf",
                                                         "FR_calf",
                                                         "RL_calf",
                                                         "RR_calf",
                                                         "trunk"]),
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
            pos_r = (0., g_height_stand_trunk),
            pos_theta = (-math.pi/2, 0.0),
            pos_z = (0.0, g_height_stand_trunk*2),
        ),
        offset_sample_xyz = (
            g_length/2,
            -g_width/2,
            0.0
        )
    )

    # height = envs.UniformHeightCommandCfg(
    #     asset_name = "robot",
    #     body_name = "trunk",
    #     resampling_time_range = (5.0, 5.0),
    #     debug_vis = True,
    #     range_height = (0.3, 0.3),
    # )
    
    
@configclass
class ReachGo1EnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: Go1SceneCfg = Go1SceneCfg(num_envs=3, env_spacing=1.5)
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
