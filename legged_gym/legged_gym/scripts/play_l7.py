# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import os
import sys

from legged_gym.envs import *
from legged_gym.gym_utils import get_args, task_registry
import torch
import faulthandler
from tqdm import tqdm
from termcolor import cprint
from isaacgym import gymtorch


def _compose_side_by_side_frame(reference_img, policy_img):
    reference_rgb = reference_img[..., :3]
    policy_rgb = policy_img[..., :3]
    return __import__("numpy").concatenate((reference_rgb, policy_rgb), axis=1)


def _sync_reference_env_to_policy_motion(env, reference_env_id=0, policy_env_id=1):
    env_ids = torch.tensor([reference_env_id], device=env.device, dtype=torch.long)
    policy_ids = torch.tensor([policy_env_id], device=env.device, dtype=torch.long)

    env._motion_ids[env_ids] = env._motion_ids[policy_ids]
    env._motion_time_offsets[env_ids] = env._motion_time_offsets[policy_ids]
    env.episode_length_buf[env_ids] = env.episode_length_buf[policy_ids]
    env.episode_init_origin[env_ids, :2] = env.episode_init_origin[policy_ids, :2]
    env._update_ref_motion()

    env.root_states[env_ids, 0:3] = env._ref_root_pos[env_ids]
    env.root_states[env_ids, 3:7] = env._ref_root_rot[env_ids]
    env.root_states[env_ids, 7:10] = env._ref_root_vel[env_ids]
    env.root_states[env_ids, 10:13] = env._ref_root_ang_vel[env_ids]
    env.dof_pos[env_ids] = env._ref_dof_pos[env_ids]
    env.dof_vel[env_ids] = env._ref_dof_vel[env_ids]
    env.last_actions[env_ids] = 0.
    env.actions[env_ids] = 0.
    env.torques[env_ids] = 0.

    env_ids_int32 = env_ids.to(dtype=torch.int32)
    env.gym.set_actor_root_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env.root_states),
        gymtorch.unwrap_tensor(env_ids_int32),
        len(env_ids_int32),
    )
    env.gym.set_dof_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env.dof_state),
        gymtorch.unwrap_tensor(env_ids_int32),
        len(env_ids_int32),
    )
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_dof_state_tensor(env.sim)
    env.gym.refresh_rigid_body_state_tensor(env.sim)


def _make_reference_motion_clock(env, policy_env_id=1):
    policy_ids = torch.tensor([policy_env_id], device=env.device, dtype=torch.long)
    return {
        "motion_id": env._motion_ids[policy_ids].clone(),
        "start_time": env._motion_time_offsets[policy_ids].clone(),
        "origin_xy": env.episode_init_origin[policy_ids, :2].clone(),
    }


def _reference_motion_time(clock, step_idx, dt):
    return clock["start_time"] + step_idx * dt


def _set_reference_env_to_motion_clock(env, clock, step_idx, reference_env_id=0):
    env_ids = torch.tensor([reference_env_id], device=env.device, dtype=torch.long)
    motion_ids = clock["motion_id"]
    motion_times = _reference_motion_time(clock, step_idx, env.dt)

    root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, *_ = env._motion_lib.calc_motion_frame(motion_ids, motion_times)
    root_pos[:, 2] += env.cfg.motion.height_offset
    root_pos[:, :2] += clock["origin_xy"]

    env._motion_ids[env_ids] = motion_ids
    env._motion_time_offsets[env_ids] = motion_times - env.episode_length_buf[env_ids] * env.dt
    env.episode_init_origin[env_ids, :2] = clock["origin_xy"]
    env.root_states[env_ids, 0:3] = root_pos
    env.root_states[env_ids, 3:7] = root_rot
    env.root_states[env_ids, 7:10] = root_vel
    env.root_states[env_ids, 10:13] = root_ang_vel
    env.dof_pos[env_ids] = dof_pos
    env.dof_vel[env_ids] = dof_vel
    dof_state = env.dof_state.view(env.num_envs, env.num_dof, 2)
    dof_state[env_ids, :, 0] = dof_pos
    dof_state[env_ids, :, 1] = dof_vel
    env.last_actions[env_ids] = 0.
    env.actions[env_ids] = 0.
    env.torques[env_ids] = 0.

    env_ids_int32 = env_ids.to(dtype=torch.int32)
    env.gym.set_actor_root_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env.root_states),
        gymtorch.unwrap_tensor(env_ids_int32),
        len(env_ids_int32),
    )
    env.gym.set_dof_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env.dof_state),
        gymtorch.unwrap_tensor(env_ids_int32),
        len(env_ids_int32),
    )
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_dof_state_tensor(env.sim)
    env.gym.refresh_rigid_body_state_tensor(env.sim)


def get_load_path(root, load_run=-1, checkpoint=-1, model_name_include="jit"):
    if checkpoint==-1:
        models = [file for file in os.listdir(root) if model_name_include in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
        checkpoint = model.split("_")[-1].split(".")[0]
    return model, checkpoint

def set_play_cfg(env_cfg):
    env_cfg.env.num_envs = 2#2 if not args.num_envs else args.num_envs
    env_cfg.env.debug_viz = True
    env_cfg.env.episode_length_s = 60
    # env_cfg.commands.resampling_time = 60
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.max_difficulty = True

    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.push_interval_s = 5
    env_cfg.domain_rand.max_push_vel_xy = 2.5
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.action_delay = False

    if hasattr(env_cfg, "motion"):
        env_cfg.motion.motion_curriculum = False

    # Set evaluation mode with full masking for student future policies
    if hasattr(env_cfg.env, 'obs_type') and env_cfg.env.obs_type == 'student_future':
        # env_cfg.env.evaluation_mode = True
        # env_cfg.env.force_full_masking = True
        env_cfg.env.evaluation_mode = False
        env_cfg.env.force_full_masking = False


def play(args):
    faulthandler.enable()
    if args.jit_path is not None:
        args.use_jit = True
        args.proj_name = "g1_stu_future_single"
        args.exptid = "g1_stu_future_single"

    log_pth = "../../logs/{}/".format(args.proj_name) + args.exptid


    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    set_play_cfg(env_cfg)

    env_cfg.env.record_video = args.record_video
    env_cfg.env.rand_reset = args.eval_start == "random"

    if_normalize = env_cfg.env.normalize_obs
    cprint(f"if_normalize: {if_normalize}", "green")

    if env_cfg.env.record_video:
        env_cfg.env.episode_length_s = 10

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    # load policy
    if not args.use_jit:
        train_cfg.runner.resume = True
    else:
        train_cfg.runner.resume = False
    ppo_runner, train_cfg = task_registry.make_alg_runner(log_root = log_pth, env=env, name=args.task, args=args, train_cfg=train_cfg)

    if args.use_jit and args.jit_path is not None:
        print("Loading jit for policy: ", args.jit_path)
        policy_jit = torch.jit.load(args.jit_path, map_location=env.device)
    else:
        policy = ppo_runner.get_inference_policy(device=env.device)
        if if_normalize:
            try:
                normalizer = ppo_runner.get_normalizer(device=env.device)
                print("Normalizer found")
            except:
                print("No normalizer found")
                normalizer = None

    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device, requires_grad=False)

    if args.record_video:
        import imageio
        env.enable_viewer_sync = True
        # env.enable_viewer_sync = False
        video_name = args.proj_name + "-" + args.exptid +".mp4"
        run_name = log_pth.split("/")[-1]
        path = f"../../../logs/videos/motion_tracking/{run_name}"
        if not os.path.exists(path):
            os.makedirs(path)
        video_name = os.path.join(path, video_name)
        mp4_writer = imageio.get_writer(video_name, fps=int(1/env.dt))
        cprint(f"Recording side-by-side video to {video_name}", "green")
        if env.num_envs < 2:
            raise ValueError("Side-by-side recording requires at least 2 envs")

    if args.record_log:
        import json
        run_name = log_pth.split("/")[-1]
        logs_dict = []
        dict_name = args.proj_name + "-" + args.exptid + ".json"
        path = f"../../logs/env_logs/{run_name}"
        if not os.path.exists(path):
            os.makedirs(path)
        dict_name = os.path.join(path, dict_name)


    if args.eval_steps is not None:
        traj_length = args.eval_steps
    elif not (args.record_video or args.record_log):
        traj_length = 100*int(env.max_episode_length)
    else:
        traj_length = 1 * int(env.max_episode_length)

    # traj_length = 2000

    env_id = env.lookat_id

    completed_episode_lengths = []
    policy_env_id = 1 if args.record_video else 0
    if args.record_video:
        reference_clock = _make_reference_motion_clock(env, policy_env_id=policy_env_id)
        _set_reference_env_to_motion_clock(env, reference_clock, step_idx=0, reference_env_id=0)
    current_episode_lengths = torch.zeros(1, device=env.device, dtype=torch.float)
    for i in tqdm(range(traj_length)):
        if args.use_jit:
            actions = policy_jit(obs.detach())
        else:
            if if_normalize and normalizer is not None:
                normalized_obs = normalizer.normalize(obs.detach())
            else:
                normalized_obs = obs.detach()
            actions = policy(normalized_obs, hist_encoding=True)
        if args.record_video:
            actions[0] = 0.

        if "AMP" in env.__class__.__name__:
            obs, _, rews, dones, info0s, _, _ = env.step(actions.detach())
        else:
            obs, _, rews, dones, infos = env.step(actions.detach())

        if args.record_video:
            _set_reference_env_to_motion_clock(env, reference_clock, step_idx=i + 1, reference_env_id=0)
        policy_done = dones[policy_env_id:policy_env_id + 1]

        if policy_done.any():
            current_episode_lengths += 1
            done_ids = policy_done.bool()
            completed_episode_lengths.extend(current_episode_lengths[done_ids].detach().cpu().tolist())
            current_episode_lengths[done_ids] = 0
        else:
            current_episode_lengths += 1


        if args.record_video:
            imgs = env.render_record(mode='rgb_array')
            if imgs is not None:
                mp4_writer.append_data(_compose_side_by_side_frame(imgs[0], imgs[policy_env_id]))

        if args.record_log:
            log_dict = env.get_episode_log()
            logs_dict.append(log_dict)

        # Interaction
        if env.button_pressed:
            print(f"env_id: {env.lookat_id:<{5}}")

    if args.record_video:
        mp4_writer.close()

    if args.record_log:
        with open(dict_name, 'w') as f:
            json.dump(logs_dict, f)

    if completed_episode_lengths:
        lengths = torch.tensor(completed_episode_lengths, dtype=torch.float32)
        print(
            "Eval episode length: "
            f"mean={lengths.mean().item():.2f}, "
            f"min={lengths.min().item():.0f}, "
            f"max={lengths.max().item():.0f}, "
            f"n={lengths.numel()}"
        )
    else:
        print(f"Eval episode length: no completed episodes in {traj_length} steps")

    if args.record_video:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == '__main__':
    args = get_args()
    play(args)
