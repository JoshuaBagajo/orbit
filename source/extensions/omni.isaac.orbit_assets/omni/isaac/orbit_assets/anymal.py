# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the ANYbotics robots.

The following configuration parameters are available:

* :obj:`ANYMAL_B_CFG`: The ANYmal-B robot with ANYdrives 3.0
* :obj:`ANYMAL_C_CFG`: The ANYmal-C robot with ANYdrives 3.0
* :obj:`ANYMAL_D_CFG`: The ANYmal-D robot with ANYdrives 3.0

Reference:

* https://github.com/ANYbotics/anymal_b_simple_description
* https://github.com/ANYbotics/anymal_c_simple_description
* https://github.com/ANYbotics/anymal_d_simple_description

"""

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.actuators import ActuatorNetLSTMCfg, DCMotorCfg, ImplicitActuatorCfg
from omni.isaac.orbit.assets.articulation import ArticulationCfg
from omni.isaac.orbit.utils.assets import ISAAC_ORBIT_NUCLEUS_DIR

##
# Configuration - Actuators.
##

ANYDRIVE_3_SIMPLE_ACTUATOR_CFG = DCMotorCfg(
    joint_names_expr=[".*HAA", ".*HFE", ".*KFE"],
    saturation_effort=120.0,
    effort_limit=80.0,
    velocity_limit=7.5,  # considered in saturation model
    stiffness={".*": 85.0},  # default: 40.0
    damping={".*": 3.5},  # default: 5.0
)
"""Configuration for ANYdrive 3.x with DC actuator model."""

ANYDRIVE_3_SYSID_ACTUATOR_CFG = DCMotorCfg(
    joint_names_expr=[".*HAA", ".*HFE", ".*KFE"],
    saturation_effort=120.0,
    effort_limit=80.0,
    velocity_limit=7.5,  # considered in saturation model
    stiffness={".*": 85.0},
    damping={
        ".*LF_HAA": 2.4304,
        ".*LH_HAA": 3.3335,
        ".*RF_HAA": 3.1598,
        ".*RH_HAA": 3.0478,
        ".*LF_HFE": 3.6843,
        ".*LH_HFE": 4.8727,
        ".*RF_HFE": 3.5460,
        ".*RH_HFE": 5.0738,
        ".*LF_KFE": 4.1307,
        ".*LH_KFE": 3.9806,
        ".*RF_KFE": 3.7354,
        ".*RH_KFE": 4.4794,
    },
    # friction={
    #     ".*LF_HAA": 0.0441,
    #     ".*LH_HAA": 0.0248,
    #     ".*RF_HAA": 0.0273,
    #     ".*RH_HAA": 0.0339,
    #     ".*LF_HFE": 0.0393,
    #     ".*LH_HFE": 0.0174,
    #     ".*RF_HFE": 0.0455,
    #     ".*RH_HFE": 0.0087,
    #     ".*LF_KFE": 0.1566,
    #     ".*LH_KFE": 0.2266,
    #     ".*RF_KFE": 0.2357,
    #     ".*RH_KFE": 0.1863,
    # },  # used but not sure where
    armature={".*": 0.09},  # used but not sure where
)
"""Configuration for ANYdrive 3.x with DC actuator model and specific parameters."""

ANYDRIVE_3_IMPLICIT_ACTUATOR_CFG = ImplicitActuatorCfg(
    joint_names_expr=[".*HAA", ".*HFE", ".*KFE"],
    # saturation_effort=120.0,
    effort_limit=80.0,
    velocity_limit=7.5,  # not respected
    stiffness={".*": 85.0},
    damping={
        ".*LF_HAA": 2.4304,
        ".*LH_HAA": 3.3335,
        ".*RF_HAA": 3.1598,
        ".*RH_HAA": 3.0478,
        ".*LF_HFE": 3.6843,
        ".*LH_HFE": 4.8727,
        ".*RF_HFE": 3.5460,
        ".*RH_HFE": 5.0738,
        ".*LF_KFE": 4.1307,
        ".*LH_KFE": 3.9806,
        ".*RF_KFE": 3.7354,
        ".*RH_KFE": 4.4794,
    },
    friction={
        ".*LF_HAA": 0.0441,
        ".*LH_HAA": 0.0248,
        ".*RF_HAA": 0.0273,
        ".*RH_HAA": 0.0339,
        ".*LF_HFE": 0.0393,
        ".*LH_HFE": 0.0174,
        ".*RF_HFE": 0.0455,
        ".*RH_HFE": 0.0087,
        ".*LF_KFE": 0.1566,
        ".*LH_KFE": 0.2266,
        ".*RF_KFE": 0.2357,
        ".*RH_KFE": 0.1863,
    },
    armature={".*": 0.09},
)
"""Configuration for ANYdrive 3.x with implicit model and specific parameters."""


ANYDRIVE_3_LSTM_ACTUATOR_CFG = ActuatorNetLSTMCfg(
    joint_names_expr=[".*HAA", ".*HFE", ".*KFE"],
    network_file=f"{ISAAC_ORBIT_NUCLEUS_DIR}/ActuatorNets/ANYbotics/anydrive_3_lstm_jit.pt",
    saturation_effort=120.0,
    effort_limit=80.0,
    velocity_limit=7.5,
)
"""Configuration for ANYdrive 3.0 (used on ANYmal-C) with LSTM actuator model."""


##
# Configuration - Articulation.
##

ANYMAL_B_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ORBIT_NUCLEUS_DIR}/Robots/ANYbotics/ANYmal-B/anymal_b.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.02, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.6),
        joint_pos={
            ".*HAA": 0.0,  # all HAA
            ".*F_HFE": 0.4,  # both front HFE
            ".*H_HFE": -0.4,  # both hind HFE
            ".*F_KFE": -0.8,  # both front KFE
            ".*H_KFE": 0.8,  # both hind KFE
        },
    ),
    actuators={"legs": ANYDRIVE_3_LSTM_ACTUATOR_CFG},
    soft_joint_pos_limit_factor=0.95,
)
"""Configuration of ANYmal-B robot using actuator-net."""


ANYMAL_C_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ORBIT_NUCLEUS_DIR}/Robots/ANYbotics/ANYmal-C/anymal_c.usd",
        # usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/ANYbotics/anymal_instanceable.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.02, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.6),
        joint_pos={
            ".*HAA": 0.0,  # all HAA
            ".*F_HFE": 0.4,  # both front HFE
            ".*H_HFE": -0.4,  # both hind HFE
            ".*F_KFE": -0.8,  # both front KFE
            ".*H_KFE": 0.8,  # both hind KFE
        },
    ),
    actuators={"legs": ANYDRIVE_3_SYSID_ACTUATOR_CFG},  # ANYDRIVE_3_SIMPLE_ACTUATOR_CFG
    soft_joint_pos_limit_factor=0.95,
)
# ISAAC_ORBIT_NUCLEUS_DIR = https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/2023.1.1/Isaac/Samples/Orbit
"""Configuration of ANYmal-C robot using actuator-net."""


ANYMAL_D_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ORBIT_NUCLEUS_DIR}/Robots/ANYbotics/ANYmal-D/anymal_d.usd",
        # usd_path=f"{ISAAC_ORBIT_NUCLEUS_DIR}/Robots/ANYbotics/ANYmal-D/anymal_d_minimal.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.02, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.6),
        joint_pos={
            ".*HAA": 0.0,  # all HAA
            ".*F_HFE": 0.4,  # both front HFE
            ".*H_HFE": -0.4,  # both hind HFE
            ".*F_KFE": -0.8,  # both front KFE
            ".*H_KFE": 0.8,  # both hind KFE
        },
    ),
    actuators={"legs": ANYDRIVE_3_SYSID_ACTUATOR_CFG},
    soft_joint_pos_limit_factor=0.95,
)
"""Configuration of ANYmal-D robot using actuator-net.

Note:
    Since we don't have a publicly available actuator network for ANYmal-D, we use the same network as ANYmal-C.
    This may impact the sim-to-real transfer performance.
"""
