# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""
This is an adaptation from play.py:
    - need to run app to simulate
    - observations and actions are saved in "simulation.pt"
    - TODO: a --save_animation flag can be passed to save the plots
    - TODO: make sure this option only applies when --sim_steps is specified
    - flag --play_wp_policy allows us to play warp policy
"""

import argparse

from omni.isaac.orbit.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--sim_steps", type=int, default=None, help="Simulation steps for recording with triggered shutdown"
)
parser.add_argument("--save_animation", type=bool, default=False, help="Save animation of generated plots")
parser.add_argument(
    "--play_wp_policy", type=bool, default=False, help="Play warp policy with different joint configuration"
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

from rsl_rl.runners import OnPolicyRunner

import omni.isaac.orbit_tasks  # noqa: F401
from omni.isaac.orbit_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.orbit_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_onnx,
)
from omni.isaac.orbit.utils.dict import print_dict
from forl.shac_actor import ActorStochasticMLP
from forl.utils.exporter import FoRLtoOnnxExporter


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, "videos"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # if specified load shac policy
    normalize_actions = False
    load_shac = False
    if load_shac:
        normalize_actions = True
        # load policy
        checkpoint = torch.load(resume_path)
        obs_rms = checkpoint["obs_rms"].to(agent_cfg.device)
        actor = ActorStochasticMLP(
            obs_dim=48, action_dim=12, units=[128, 64, 32], normalizer=obs_rms
        )  # NOTE: normalization happening in here
        actor.to(agent_cfg.device)
        actor.load_state_dict(checkpoint["actor"])
        actor.eval()
        policy = lambda x: actor(x)  # noqa: E731

        # export actor
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        if not os.path.exists(export_model_dir):
            os.makedirs(export_model_dir, exist_ok=True)

        policy_exporter = FoRLtoOnnxExporter(actor)  # NOTE: we're adding a tanh operation in this
        policy_exporter.export(path=export_model_dir, filename="policy.onnx")

    else:
        # load previously trained model
        ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        ppo_runner.load(resume_path)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")

        # obtain the trained policy for inference
        policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

        # export policy to onnx
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        export_policy_as_onnx(ppo_runner.alg.actor_critic, export_model_dir, filename="policy.onnx")

    # reset environment
    obs, _ = env.get_observations()
    # obs, _ = env.reset()
    # print(f"Initial obs: {obs}")

    # initialize plot
    fig, axes, lines = initialize_plot()

    # initialize data lists
    xdata, x_vel, y_vel, z_vel = [], [], [], []

    # initialize buffers
    obs_buff = []
    obs_buff.append(obs)
    acts_buff = []

    # test
    # print(dir(env.unwrapped.scene))  # [..., '_articulations', '_rigid_objects', '_sensors', '_terrain', 'articulations', 'keys', 'rigid_objects', 'sensors', 'stage', 'terrain']
    # print(env.unwrapped.scene.articulations.keys())  # dict_keys(['robot'])
    # print(env.unwrapped.scene.rigid_objects.keys())  # dict_keys([])
    # print(env.unwrapped.scene.sensors.keys())  # dict_keys(['contact_forces'])

    # print(env.unwrapped.scene.sensors["contact_forces"])  # body names        : ['base', 'LF_HIP', 'LF_THIGH', 'LF_SHANK', 'LF_FOOT', 'LH_HIP', 'LH_THIGH', 'LH_SHANK', 'LH_FOOT', 'RF_HIP', 'RF_THIGH', 'RF_SHANK', 'RF_FOOT', 'RH_HIP', 'RH_THIGH', 'RH_SHANK', 'RH_FOOT']
    # print(dir(env.unwrapped.scene.sensors["contact_forces"]))  # ['body_names', 'body_physx_view', 'cfg', 'compute_first_air', 'compute_first_contact', 'contact_physx_view', 'data', 'device', 'find_bodies', 'has_debug_vis_implementation', 'num_bodies']
    # print(env.unwrapped.scene["contact_forces"])  # same as above
    # print(dir(env.unwrapped.scene.sensors["contact_forces"].find_bodies))
    # print(dir(env.unwrapped.scene.sensors["contact_forces"].body_physx_view))  # [..., 'apply_forces', 'apply_forces_and_torques_at_position', 'check', 'count', 'get_coms', 'get_contact_offsets', 'get_disable_gravities', 'get_disable_simulations', 'get_inertias', 'get_inv_inertias', 'get_inv_masses', 'get_masses', 'get_material_properties', 'get_rest_offsets', 'get_transforms', 'get_velocities', 'max_shapes', 'prim_paths', 'set_coms', 'set_contact_offsets', 'set_disable_gravities', 'set_disable_simulations', 'set_dynamic_targets', 'set_inertias', 'set_kinematic_targets', 'set_masses', 'set_material_properties', 'set_rest_offsets', 'set_transforms', 'set_velocities']
    # print(dir(env.unwrapped.scene.sensors["contact_forces"].contact_physx_view))  # ['__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_backend', '_frontend', '_num_components', 'check', 'filter_count', 'get_contact_data', 'get_contact_force_matrix', 'get_net_contact_forces', 'max_contact_data_count', 'sensor_count', 'sensor_names', 'sensor_paths']
    # print(env.unwrapped.scene.sensors["contact_forces"].data)  # shows all data
    # print(dir(env.unwrapped.scene.sensors["contact_forces"].data))  # ['current_air_time', 'current_contact_time', 'force_matrix_w', 'last_air_time', 'last_contact_time', 'net_forces_w', 'net_forces_w_history', 'pos_w', 'quat_w']
    print((env.unwrapped.scene.sensors["contact_forces"].data.net_forces_w))
    # print(env.unwrapped.scene.keys())  # ['terrain', 'robot', 'contact_forces', 'light', 'sky_light']
    # print(dir(env.unwrapped.scene["robot"]))  # ['FORWARD_VEC_B', 'GRAVITY_VEC_W', ..., 'actuators', 'body_names', 'body_physx_view', 'cfg', 'data', 'device', 'find_bodies', 'find_joints', 'joint_names', 'num_bodies', 'num_fixed_tendons', 'num_instances', 'num_joints', 'reset', 'root_physx_view', 'set_debug_vis', 'set_external_force_and_torque', 'set_joint_effort_target', 'set_joint_position_target', 'set_joint_velocity_target', 'update', 'write_data_to_sim', 'write_joint_armature_to_sim', 'write_joint_damping_to_sim', 'write_joint_effort_limit_to_sim', 'write_joint_friction_to_sim', 'write_joint_state_to_sim', 'write_joint_stiffness_to_sim', 'write_root_pose_to_sim', 'write_root_state_to_sim', 'write_root_velocity_to_sim']
    # print(env.unwrapped.scene["robot"].body_names)  # ['base', 'LF_HIP', 'LH_HIP', 'RF_HIP', 'RH_HIP', 'LF_THIGH', 'LH_THIGH', 'RF_THIGH', 'RH_THIGH', 'LF_SHANK', 'LH_SHANK', 'RF_SHANK', 'RH_SHANK', 'LF_FOOT', 'LH_FOOT', 'RF_FOOT', 'RH_FOOT']
    # print((env.unwrapped.scene["robot"].body_physx_view)  # = print(dir(env.unwrapped.scene.sensors["contact_forces"].body_physx_view))
    # print((env.unwrapped.scene["robot"].body_physx_view.get_masses()))
    # print(dir(env.unwrapped.scene["robot"].root_physx_view))  # [..., 'get_coms', 'get_contact_offsets', 'get_coriolis_and_centrifugal_forces', 'get_dof_actuation_forces', 'get_dof_armatures', 'get_dof_dampings', 'get_dof_forces', 'get_dof_friction_coefficients', 'get_dof_limits', 'get_dof_max_forces', 'get_dof_max_velocities', 'get_dof_motions', 'get_dof_position_targets', 'get_dof_positions', 'get_dof_projected_joint_forces', 'get_dof_stiffnesses', 'get_dof_types', 'get_dof_velocities', 'get_dof_velocity_targets', 'get_fixed_tendon_dampings', 'get_fixed_tendon_limit_stiffnesses', 'get_fixed_tendon_limits', 'get_fixed_tendon_offsets', 'get_fixed_tendon_rest_lengths', 'get_fixed_tendon_stiffnesses', 'get_force_sensor_forces', 'get_generalized_gravity_forces', 'get_inertias', 'get_inv_inertias', 'get_inv_masses', 'get_jacobians', 'get_link_accelerations', 'get_link_incoming_joint_force', 'get_link_transforms', 'get_link_velocities', 'get_mass_matrices', 'get_masses', 'get_material_properties', 'get_metatype', 'get_rest_offsets', 'get_root_transforms', 'get_root_velocities', ...]
    print((env.unwrapped.scene["robot"].body_physx_view.get_coms()))  # different order to print((env.unwrapped.scene["robot"].root_physx_view.get_coms()))
    # print(env.unwrapped.scene["robot"].data)  # ArticulationData ...
    # print(vars(env.unwrapped.scene.contact_forces))

    # simulate environment
    writer = FFMpegWriter(fps=10)  # extra_args=['-i', '/home/jbagajo/ffmpeg-7.0-amd64-static/ffmpeg'])
    if args_cli.sim_steps is not None:
        contact_forces_buffer = torch.zeros(
            (args_cli.sim_steps, 4, 3), device=agent_cfg.device
        )  # for 4 legs (3 forces)
        foot_coms_buffer = torch.zeros(
            (args_cli.sim_steps, 4, 7), device=agent_cfg.device
        )  # for 4 legs (3 position coords + 4 orientation entries)
        foot_link_buffer = foot_coms_buffer.clone()
    else:
        contact_forces_buffer = torch.zeros(
            (1000, 4, 3), device=agent_cfg.device
        )  # for 4 legs (3 forces)
        foot_coms_buffer = torch.zeros(
            (1000, 4, 7), device=agent_cfg.device
        )  # for 4 legs (3 position coords + 4 orientation entries)
        foot_link_buffer = foot_coms_buffer.clone()

    with writer.saving(fig, "base_lin_vel_animation.mp4", dpi=100):
        count = 0
        # actions = torch.zeros(12, dtype=torch.float32, device=agent_cfg.device).repeat(1, 1)
        while simulation_app.is_running():
            # start a counter
            if count == 0:
                start_time = time.time()
            # if count == 10:
            #     actions = torch.ones(12, dtype=torch.float32, device=agent_cfg.device).repeat(1, 1) * 0.8
            # if count == 60:
            #     actions = torch.zeros(12, dtype=torch.float32, device=agent_cfg.device).repeat(1, 1)
            # if count == 110:
            #     actions = -torch.ones(12, dtype=torch.float32, device=agent_cfg.device).repeat(1, 1) * 1.5
            # if count == 160:
            #     actions = torch.ones(12, dtype=torch.float32, device=agent_cfg.device).repeat(1, 1) * 1.5
            # if count == 170:
            #     actions = torch.zeros(12, dtype=torch.float32, device=agent_cfg.device).repeat(1, 1)
            # run everything in inference mode
            with torch.inference_mode():
                if args_cli.play_wp_policy:
                    # adjust obs for warp policy
                    obs[:, 12:24] = reorder_joints_inverse(obs[:, 12:24])  # joint pos
                    obs[:, 24:36] = reorder_joints_inverse(obs[:, 24:36])  # joint vel
                    obs[:, 36:48] = reorder_joints_inverse(obs[:, 36:48])  # actions
                    # agent stepping
                    actions = policy(obs)  # torch.Size([1, 12])
                    # adjust actions for orbit simulation
                    actions = reorder_joints(actions)
                    # env stepping
                    if normalize_actions:
                        obs, _, _, _ = env.step(torch.tanh(actions))  # torch.Size([1, 48])
                    else:
                        obs, _, _, _ = env.step(actions)
                else:
                    # agent stepping
                    actions = policy(obs)  # torch.Size([1, 12])
                    # env stepping
                    obs, _, _, _ = env.step(actions)  # torch.Size([1, 48])

                contact_forces = env.unwrapped.scene.sensors["contact_forces"].data.net_forces_w  # (envs, 17 bodies, 3d) unsqueezed
                body_coms = env.unwrapped.scene["robot"].body_physx_view.get_coms()  # (envs, 17 bodies, 7) squeezed
                link_transforms = env.unwrapped.scene["robot"].root_physx_view.get_link_transforms()  # (envs, 17, 7) unsqueezed
                # root_transforms = env.unwrapped.scene["robot"].root_physx_view.get_root_transforms()  # only base
                contact_forces_buffer[count] = contact_forces[0, [4, 8, 12, 16], :]
                foot_coms_buffer[count] = body_coms[[4, 8, 12, 16], :]
                foot_link_buffer[count] = link_transforms[0, [13, 15, 14, 16], :]  # change order from LF, LH, RF, RH to LF, RF, LH, RH
                # print(foot_coms_buffer[count, 0, 2])  # constant
                # print(foot_link_buffer[count, 0, 2])
                contact_offsets = env.unwrapped.scene["robot"].root_physx_view.get_contact_offsets()  # (envs, 17 bodies) unsqueezed
                # print(contact_offsets[0, 13:])  # constant 0.0013

            # get data of interest
            base_lin_vel = obs[0, :3].detach().cpu()

            # update data
            xdata.append(count)
            x_vel.append(base_lin_vel[0].item())
            y_vel.append(base_lin_vel[1].item())
            z_vel.append(base_lin_vel[2].item())

            # keep only the most recent 50 data points
            xdata[:] = xdata[-50:]
            x_vel[:] = x_vel[-50:]
            y_vel[:] = y_vel[-50:]
            z_vel[:] = z_vel[-50:]

            # update the plot
            # update_plot(fig, axes, lines, xdata, [x_vel, y_vel, z_vel])

            # save plot as animation
            writer.grab_frame()

            # update buffers
            obs_buff.append(obs)
            acts_buff.append(actions)

            if args_cli.sim_steps is not None and count >= args_cli.sim_steps - 1:
                print("Stopping sim...")
                end_time = time.time()
                print(f"Elapsed time: {end_time - start_time} seconds")
                print(f"Simulation time: {env_cfg.sim.dt * env_cfg.decimation * count} seconds")
                break

            # update counter
            count += 1

    # save actions and observations
    buff_dict = {"actions": acts_buff, "observations": obs_buff, "extras": (contact_forces_buffer, foot_coms_buffer, foot_link_buffer)}
    torch.save(buff_dict, log_root_path + "/simulation.pt")

    # optionally, keep the plot open after the simulation ends
    print("Close plot to exit properly...")
    plt.ioff()
    plt.show()

    # close the simulator
    print("Closing env...")
    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time} seconds")
    print(f"Simulation time: {env_cfg.sim.dt * env_cfg.decimation * count} seconds")
    env.close()


def initialize_plot():
    """Initializes the plot and returns figure, axes, and line objects."""
    plt.ion()  # Turn on interactive mode
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)  # Create 3 subplots sharing the x-axis

    # Initialize lines for each subplot
    (line1,) = ax1.plot([], [], "b-")  # Blue line for the first dimension
    (line2,) = ax2.plot([], [], "g-")  # Green line for the second dimension
    (line3,) = ax3.plot([], [], "r-")  # Red line for the third dimension

    # Set initial limits (adjust as needed)
    # ax1.set_ylim(-1, 1)
    # ax2.set_ylim(-1, 1)
    # ax3.set_ylim(-1, 1)

    # Labels for the y-axis
    ax1.set_ylabel(r"$v_x$ [m/s]")
    ax2.set_ylabel(r"$v_y$ [m/s]")
    ax3.set_ylabel(r"$v_z$ [m/s]")

    # Label for the x-axis
    ax3.set_xlabel("Time step")

    return fig, (ax1, ax2, ax3), (line1, line2, line3)


def update_plot(fig, axes, lines, xdata, ydata):
    """Updates the plot with new data."""
    for ax, line, ydat in zip(axes, lines, ydata):
        line.set_xdata(xdata)
        line.set_ydata(ydat)

        # Set x-axis limits to show only the most recent 50 data points
        ax.relim()
        ax.autoscale_view()

    # Draw the plot
    fig.canvas.draw()
    fig.canvas.flush_events()


def reorder_joints(joint_var):
    """Reorder joint variables (pos/vel) from warp config to orbit config."""

    # copy actions
    joint_var = joint_var.clone()

    joint_var[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]] = joint_var[:, [6, 3, 9, 1, 7, 4, 10, 2, 8, 5]]

    return joint_var


def reorder_joints_inverse(joint_var):
    """Reorder joint variables (pos/vel) from orbit config to warp config."""

    # copy joint_var
    joint_var = joint_var.clone()

    joint_var[:, [6, 3, 9, 1, 7, 4, 10, 2, 8, 5]] = joint_var[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]

    return joint_var


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    print("Closing app...")
    simulation_app.close()
