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
parser.add_argument("--sim_steps", type=int, default=None, help="Simulation steps for recording with triggered shutdown")
parser.add_argument("--save_animation", type=bool, default=False, help="Save animation of generated plots")
parser.add_argument("--play_wp_policy", type=bool, default=False, help="Play warp policy with different joint configuration")
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
    print(f"Initial obs: {obs}")

    # initialize plot
    fig, axes, lines = initialize_plot()

    # initialize data lists
    xdata, x_vel, y_vel, z_vel = [], [], [], []

    # initialize buffers
    obs_buff = []
    obs_buff.append(obs)
    acts_buff = []

    # simulate environment
    writer = FFMpegWriter(fps=10)  # extra_args=['-i', '/home/jbagajo/ffmpeg-7.0-amd64-static/ffmpeg'])
    with writer.saving(fig, "base_lin_vel_animation.mp4", dpi=100):
        count = 0
        # actions = torch.zeros(12, dtype=torch.float32, device=agent_cfg.device).repeat(1, 1)
        while simulation_app.is_running():
            # start a counter
            count += 1
            if count == 1:
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
                    obs, _, _, _ = env.step(actions)  # torch.Size([1, 48])
                else:
                    # agent stepping
                    actions = policy(obs)  # torch.Size([1, 12])
                    # env stepping
                    obs, _, _, _ = env.step(actions)  # torch.Size([1, 48])

            # get data of interest
            base_lin_vel  = obs[0, :3].detach().cpu()

            # update data
            xdata.append(count-1)
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

            if args_cli.sim_steps is not None and count >= args_cli.sim_steps:
                print("Stopping sim...")
                end_time = time.time()
                print(f"Elapsed time: {end_time - start_time} seconds")
                print(f"Simulation time: {env_cfg.sim.dt * env_cfg.decimation * count} seconds")
                break

    # save actions and observations
    buff_dict = {"actions": acts_buff, "observations": obs_buff}
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
    line1, = ax1.plot([], [], 'b-')  # Blue line for the first dimension
    line2, = ax2.plot([], [], 'g-')  # Green line for the second dimension
    line3, = ax3.plot([], [], 'r-')  # Red line for the third dimension

    # Set initial limits (adjust as needed)
    # ax1.set_ylim(-1, 1)
    # ax2.set_ylim(-1, 1)
    # ax3.set_ylim(-1, 1)

    # Labels for the y-axis
    ax1.set_ylabel(r'$v_x$ [m/s]')
    ax2.set_ylabel(r'$v_y$ [m/s]')
    ax3.set_ylabel(r'$v_z$ [m/s]')

    # Label for the x-axis
    ax3.set_xlabel('Time step')

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
