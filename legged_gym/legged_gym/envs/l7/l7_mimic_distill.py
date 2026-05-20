from legged_gym.envs.g1.g1_mimic_distill import G1MimicDistill
from legged_gym.envs.base.humanoid_char import convert_to_global_root_body_pos

import torch


def map_motion_body_pos_by_name(body_pos: torch.Tensor, motion_body_names, target_body_names) -> torch.Tensor:
    body_pos_mapped = torch.zeros(
        (body_pos.shape[0], len(target_body_names), body_pos.shape[-1]),
        dtype=body_pos.dtype,
        device=body_pos.device,
    )
    motion_body_idx = {name: idx for idx, name in enumerate(motion_body_names)}
    for target_idx, target_name in enumerate(target_body_names):
        source_idx = motion_body_idx.get(target_name)
        if source_idx is not None:
            body_pos_mapped[:, target_idx] = body_pos[:, source_idx]
    return body_pos_mapped


class L7MimicDistill(G1MimicDistill):
    def _map_body_pos_to_asset(self, body_pos):
        if body_pos.shape[1] == len(self.body_names):
            return body_pos
        return map_motion_body_pos_by_name(body_pos, self._motion_lib._body_link_list, self.body_names)

    def _reset_ref_motion(self, env_ids, motion_ids=None):
        n = len(env_ids)
        if motion_ids is None:
            motion_ids = self._motion_lib.sample_motions(n, motion_difficulty=self.motion_difficulty)

        if self._rand_reset:
            motion_times = self._motion_lib.sample_time(motion_ids)
        else:
            motion_times = torch.zeros(motion_ids.shape, device=self.device, dtype=torch.float)

        self._motion_ids[env_ids] = motion_ids
        self._motion_time_offsets[env_ids] = motion_times

        root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, body_pos, root_pos_delta_local, root_rot_delta_local = self._motion_lib.calc_motion_frame(motion_ids, motion_times)
        root_pos[:, 2] += self.cfg.motion.height_offset
        body_pos = self._map_body_pos_to_asset(body_pos)
        self._ref_root_pos[env_ids] = root_pos
        self._ref_root_rot[env_ids] = root_rot
        self._ref_root_vel[env_ids] = root_vel
        self._ref_root_ang_vel[env_ids] = root_ang_vel
        self._ref_dof_pos[env_ids] = dof_pos
        self._ref_dof_vel[env_ids] = dof_vel
        self._ref_root_pos_delta_local[env_ids] = root_pos_delta_local
        self._ref_root_rot_delta_local[env_ids] = root_rot_delta_local
        self._ref_body_pos[env_ids] = convert_to_global_root_body_pos(root_pos=root_pos, root_rot=root_rot, body_pos=body_pos)

    def _update_ref_motion(self):
        motion_ids = self._motion_ids
        motion_times = self._get_motion_times()
        root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, body_pos, root_pos_delta_local, root_rot_delta_local = self._motion_lib.calc_motion_frame(motion_ids, motion_times)
        root_pos[:, 2] += self.cfg.motion.height_offset
        root_pos[:, :2] += self.episode_init_origin[:, :2]
        body_pos = self._map_body_pos_to_asset(body_pos)

        self._ref_root_pos[:] = root_pos
        self._ref_root_rot[:] = root_rot
        self._ref_root_vel[:] = root_vel
        self._ref_root_ang_vel[:] = root_ang_vel
        self._ref_dof_pos[:] = dof_pos
        self._ref_dof_vel[:] = dof_vel
        self._ref_root_pos_delta_local[:] = root_pos_delta_local
        self._ref_root_rot_delta_local[:] = root_rot_delta_local
        self._ref_body_pos[:] = convert_to_global_root_body_pos(root_pos=root_pos, root_rot=root_rot, body_pos=body_pos)
