# L7 LAFAN Motion Training Notes

This note describes what is inside `datasets/lafan1_l7_filtered_pkl`, how the
data is connected to the L7 training code, and what the policy networks consume
and produce during training.

## Dataset

`datasets/lafan1_l7_filtered_pkl` contains 37 retargeted LAFAN-style motion
files for the L7 robot. The files are grouped by the action name embedded in the
filename:

| Group | Files |
| --- | ---: |
| `aiming` | 5 |
| `dance` | 7 |
| `fight` | 3 |
| `fightAndSports` | 2 |
| `jumps` | 3 |
| `push` | 1 |
| `run` | 4 |
| `sprint` | 2 |
| `walk` | 10 |

Each file is a Python pickle containing a dictionary. For example,
`robotera_l7_walk2_subject1_normal.pkl` has:

| Key | Shape / type | Meaning |
| --- | --- | --- |
| `fps` | `int`, normally 30 | Source motion frame rate. |
| `root_pos` | `(T, 3)` float32 | Root translation in world/reference coordinates. |
| `root_rot` | `(T, 4)` float32 | Root quaternion. The L7 converter stores it as `(w, x, y, z)`. |
| `dof_pos` | `(T, 29)` float32 | L7 joint positions for the 29 controlled DOFs. |
| `local_body_pos` | `(T, 51, 3)` float32 | Local positions for the 51 listed rigid/link bodies. |
| `link_body_list` | length 51 list | Body names matching the `local_body_pos` body axis. |

Important caveat for the current `lafan1_l7_filtered_pkl` files: the
`local_body_pos` field is not currently usable as a key-body tracking target.
For example, in
`robotera_l7_dance1_subject1_normal.pkl`, `local_body_pos` has shape
`(3945, 51, 3)`, but only the `pelvis` row has nonzero values. The L7 key-body
rows used by training are all zero for every frame:

```text
left_wrist_roll_link      maxnorm = 0
right_wrist_roll_link     maxnorm = 0
left_ankle_roll_link      maxnorm = 0
right_ankle_roll_link     maxnorm = 0
left_knee_link            maxnorm = 0
right_knee_link           maxnorm = 0
left_shoulder_roll_link   maxnorm = 0
right_shoulder_roll_link  maxnorm = 0
torso_link                maxnorm = 0
```

The `root_pos`, `root_rot`, and `dof_pos` fields can still initialize the robot
to a visually reasonable pose, because Isaac Gym state reset uses root state and
DOF state directly. `local_body_pos` is different: it is used as a reference
target for key-body tracking rewards, privileged motion observations, and pose
termination. If it is zero, the code interprets hands, feet, knees, shoulders,
and torso as being at the root.

The raw PKL does not store velocities. It stores positions/orientations over
time. `MotionLib` derives velocity-like tensors when each file is loaded:

- `root_vel` is `torch.gradient(root_pos, spacing=1/fps, dim=0)[0]`.
- `dof_vel` is `torch.gradient(dof_pos, spacing=1/fps, dim=0)[0]`.
- `root_ang_vel` is computed from neighboring root quaternions: adjacent
  quaternion differences are converted to exponential-map angular increments and
  divided by `dt`; interior frames use a central difference.
- `root_pos_delta_local` is the frame-to-frame root translation delta, rotated
  into the previous root frame.
- `root_rot_delta_local` is the frame-to-frame root rotation delta, represented
  as Euler-like local angular delta.

So the dataset is position-level motion data; velocity fields are cached
derivatives produced by `pose.utils.motion_lib_pkl.MotionLib`, not independent
measurements from the source dataset.

`dof_pos` has 29 entries because it only describes actuated/controlled joints:
the L7 policy action space is 29-dimensional. `local_body_pos` has 51 entries
because it describes rigid/link body positions in the robot kinematic tree,
including fixed links and intermediate bodies that are not independent
actuators. In short:

```text
29 DOFs
└── controllable generalized coordinates used for action, dof_pos, dof_vel

51 bodies/links
└── rigid/link positions used for body tracking, key-body lookup, and critic
    privileged reference features
```

The env usually does not feed all 51 bodies to the policy. For privileged
tracking features it selects `motion.key_bodies`, which for L7 contains 9 links:
left/right wrists, left/right ankles, left/right knees, left/right shoulders,
and torso.

The retargeted pickle is already in the format expected by
`pose.utils.motion_lib_pkl.MotionLib`: a YAML file points at one or more PKL
files, `HumanoidMimic._load_motions()` creates the motion library, and the env
samples reference root pose, joint pose, body positions, and velocities from that
library.

For L7 YAMLs in the main TWIST2 project, `root_rot_format: wxyz` is set
explicitly. `MotionLib` converts those quaternions to Isaac Gym's expected
`xyzw` order during load. Runtime diagnostics confirmed that the loaded
`motion_lib._motion_root_rot`, `env._ref_root_rot`, and the initialized
`env.root_states[:, 3:7]` all use the converted `xyzw` quaternion. The current
one-step termination problem is therefore not caused by the old sideways-root
quaternion issue.

After `MotionLib` loads the YAML/PKL files, each motion is converted from a
per-file dictionary into stacked tensors on the selected device:

```text
MotionLib
├── motion metadata
│   ├── _motion_names: list[str], one name per loaded/sub-motion
│   ├── _motion_files: list[str], source PKL path per motion
│   ├── _motion_weights: (M,), normalized sampling weights
│   ├── _motion_fps: (M,), source FPS
│   ├── _motion_dt: (M,), seconds per frame
│   ├── _motion_num_frames: (M,), frame count per motion
│   ├── _motion_lengths: (M,), motion length in seconds
│   └── _motion_start_idx: (M,), offset into the concatenated frame tensors
├── frame tensors, concatenated over all motions, N = sum(T_i)
│   ├── _motion_root_pos: (N, 3)
│   ├── _motion_root_rot: (N, 4)
│   ├── _motion_root_vel: (N, 3), finite-difference root linear velocity
│   ├── _motion_root_ang_vel: (N, 3), SO(3) finite-difference angular velocity
│   ├── _motion_dof_pos: (N, 29)
│   ├── _motion_dof_vel: (N, 29), finite-difference joint velocity
│   ├── _motion_local_body_pos: (N, 51, 3)
│   ├── _motion_root_pos_delta_local: (N, 3), local delta from previous frame
│   └── _motion_root_rot_delta_local: (N, 3), local Euler delta from previous frame
└── body metadata
    └── _body_link_list: length-51 body name list used by get_key_body_idx()
```

`calc_motion_frame(motion_ids, motion_times)` samples/interpolates these stacked
tensors and returns:

```text
root_pos                 (B, 3)
root_rot                 (B, 4)
root_vel                 (B, 3)
root_ang_vel             (B, 3)
dof_pos                  (B, 29)
dof_vel                  (B, 29)
local_body_pos           (B, 51, 3)
root_pos_delta_local     (B, 3)
root_rot_delta_local     (B, 3)
```

where `B = num_envs * num_sampled_motion_steps`. The env reshapes those tensors
back to `(num_envs, num_steps, ...)` before constructing observations.

## Current Pose-Termination Failure Mode

With pose termination enabled, `HumanoidMimic.check_termination()` compares the
current simulated key-body positions against the reference key-body positions
from `_ref_body_pos`:

```text
body_pos     = simulated key body position - simulated root position
tar_body_pos = reference key body position - reference root position
```

When `global_obs=False`, both are rotated into their root-local frames before
the distance is computed. The code then takes the maximum squared error over the
9 key bodies and terminates if it exceeds `pose_termination_dist ** 2`.

The current L7 key bodies are:

```text
left_wrist_roll_link
right_wrist_roll_link
left_ankle_roll_link
right_ankle_roll_link
left_knee_link
right_knee_link
left_shoulder_roll_link
right_shoulder_roll_link
torso_link
```

A 16-env runtime diagnostic on `l7_lafan1_dance.yaml` showed the reset source:

```text
contact      1 / 16
height       0 / 16
roll         0 / 16
pitch        0 / 16
motion_end   0 / 16
vel          0 / 16
pose        16 / 16
```

The root quaternions were already correct and matched between reference and sim,
but `tar_body_pos` for every key body was effectively zero because
`local_body_pos` in the PKL is zero. The real simulated feet and knees are not
at the root, so the pose error immediately exceeds the `0.7 m` threshold:

```text
left_ankle_roll_link   mean error ~= 0.84 m
right_ankle_roll_link  mean error ~= 0.85 m
left_knee_link         mean error ~= 0.58 m
right_knee_link        mean error ~= 0.58 m
```

`check_termination()` suppresses reset only on the initial `episode_length_buf ==
0` step. On the next step, `pose_fail=True`, so the episode ends at length 1.
This explains the observed `Mean episode length ~= 1`.

Fixing the root cause requires producing valid `local_body_pos` / key-body FK
targets for the L7 PKLs, or changing the TWIST2 training code so those targets
are computed from `root + dof_pos + L7 kinematics` instead of trusting the empty
PKL field. Disabling or relaxing pose termination only masks this failure; it
does not repair key-body tracking rewards or privileged key-body observations.

## Why TWIST1 Does Not Fail The Same Way

`TWIST1` uses the same `lafan1_l7_filtered_pkl` files, so the empty
`local_body_pos` data issue still exists there. It avoids the immediate
one-step reset for two main reasons:

1. `TWIST1/legged_gym/legged_gym/envs/l7/l7_mimic_distill_config.py` sets:

```python
pose_termination = False
```

So `check_termination()` does not use the bad key-body reference positions to
end the episode. Contacts, height, roll/pitch, velocity, and motion end can
still terminate, but the all-env pose failure is bypassed.

2. `TWIST1` also removes live key-body positions from part of the proprioceptive
privileged info path for L7:

```python
key_body_pos = torch.zeros((self.num_envs, len(self._key_body_ids) * 3), device=self.device)
```

This does not fix the PKL data, but it prevents one observation component from
depending on simulated key-body positions. TWIST1 still includes reference
key-body data in privileged mimic observations, and because the source PKL values
are zero, those reference key-body features are not semantically valid. The key
difference is that TWIST1 does not use them for pose termination, so training can
continue instead of resetting every step.

TWIST1 also has L7-specific body-count conversion helpers
(`l7_body_from_34_to_30` / `l7_body_from_34_to_51`) for datasets with 34 body
rows. They do not solve the current `lafan1_l7_filtered_pkl` issue in TWIST2:
the current PKLs expose 51 body rows, but the relevant key-body rows are zero.

## Training Config Adaptation

The main project now has dataset YAMLs under `legged_gym/motion_data_configs/`:

| YAML | Purpose |
| --- | --- |
| `l7_lafan1_all.yaml` | All 37 motions. |
| `l7_lafan1_dance.yaml` | 7 dance motions. |
| `l7_lafan1_walk.yaml` | 10 walk motions. |
| `l7_lafan1_run.yaml` | 4 run motions. |
| `l7_lafan1_*.yaml` | Other per-category subsets. |

The root training script passes arbitrary extra arguments through to
`legged_gym/legged_gym/scripts/train.py`. The `--motion_file` CLI option
overrides `env_cfg.motion.motion_file`, so L7 training can switch from the
default AMASS config to LAFAN without changing Python config classes.

## Main Project Policy I/O

The main L7 training entry point uses task `l7_stu_future`, implemented by
`L7MimicFuture` and configured by `L7MimicStuFutureCfgDAgger`. A smoke run with
the all-LAFAN YAML reports:

| Quantity | Value |
| --- | ---: |
| `num_actions` | 29 |
| actor observation width | 1432 |
| privileged critic observation width | 1734 |
| current observation width | 127 |
| history observation width | 1270 |
| future observation width | 35 |

The actor class is `ActorCriticFuture`. Its actor input is:

```text
obs = current + history + future
current = motion_obs + proprio = 35 + 92 = 127
history = 10 * current = 1270
future = 35
total = 127 + 1270 + 35 = 1432
```

Detailed actor observation layout:

```text
actor obs: 1432
├── current: 127
│   ├── motion_obs: 35
│   │   ├── root_vel_local_xy: 2
│   │   │   └── reference root linear velocity x/y, rotated into root-local frame
│   │   ├── root_pos_z: 1
│   │   │   └── reference root height
│   │   ├── roll_pitch: 2
│   │   │   └── reference root roll and pitch from root_rot
│   │   ├── yaw_ang_vel_local: 1
│   │   │   └── reference root yaw angular velocity, rotated into root-local frame
│   │   └── dof_pos: 29
│   │       └── reference L7 joint positions
│   └── proprio: 92
│       ├── base_ang_vel: 3
│       │   └── simulated robot base angular velocity, scaled by obs_scales.ang_vel
│       ├── imu_roll_pitch: 2
│       │   └── simulated robot roll and pitch
│       ├── dof_pos_error_to_default: 29
│       │   └── reindexed current joint position minus default joint position
│       ├── dof_vel: 29
│       │   └── reindexed current joint velocity; ankle velocity entries are zeroed
│       └── last_action: 29
│           └── reindexed previous policy action
├── history: 1270
│   └── 10 previous current observations
│       └── each history slot has the same 35 + 92 layout as current
└── future: 35
    └── one future target step with the same 35-dim layout as motion_obs
```

`motion_obs` and `future` both come from the sampled reference motion in
`MotionLib`; `proprio` comes from the live Isaac Gym robot state and the previous
action buffer. The actor outputs a Gaussian mean for 29 actions; sampled actions
and inference actions both have shape `(num_envs, 29)`.

The critic receives `privileged_obs_buf`, not the actor obs. For L7 future
training its raw env-side critic observation is 1734-dimensional:

```text
critic raw obs / privileged_obs_buf: 1734
├── priv_mimic_obs: 1540 = 20 * 77
│   └── 20 privileged reference frames at tar_motion_steps_priv
│       └── each frame: 77 = 21 + 29 + 3 * 9
│           ├── root_pos: 3
│           ├── root_pos_distance_to_target: 3
│           │   └── reference root position minus simulated robot root position
│           ├── roll_pitch_yaw: 3
│           ├── root_vel_local: 3
│           ├── root_ang_vel_local: 3
│           ├── root_pos_delta_local: 3
│           ├── root_rot_delta_local: 3
│           ├── dof_pos: 29
│           └── key_body_pos: 27 = 9 * 3
│               └── selected reference key body positions, local if global_obs=False
├── proprio: 92
│   └── same proprio layout as actor current obs
└── priv_info: 102
    ├── base_lin_vel: 3
    ├── simulated_root_pos: 3
    ├── simulated_root_quat: 4
    ├── simulated_key_body_pos: 27 = 9 * 3
    ├── foot_contact_mask: 2
    ├── mass_params_tensor: 4
    ├── friction_coeffs_tensor: 1
    ├── motor_strength_p_offset: 29
    └── motor_strength_d_offset: 29
```

The dimensions add up as `1540 + 92 + 102 = 1734`. This privileged motion frame
is richer than the actor's 35-dimensional `motion_obs`: the critic sees 20
reference frames, root pose/delta terms, local velocities, and selected key-body
positions.

Inside `ActorCriticFuture.evaluate()`, the critic does not pass the 1734 raw
vector straight into the MLP. It first splits and reassembles it:

```text
critic_observations
├── motion_obs = critic_observations[:, :n_priv_mimic_obs]
│   └── encoded by MotionEncoder: 1540 -> motion_latent_dim, normally 128
├── motion_single_obs = critic_observations[:, :77]
│   └── the first privileged reference frame is kept explicitly
└── critic_observations[:, n_priv_mimic_obs:]
    └── non-motion privileged part: proprio + priv_info

critic MLP input =
    non-motion privileged part
  + motion_single_obs
  + encoded motion latent
```

The critic output is a scalar value with shape `(num_envs, 1)`.

If no teacher run is passed, `OnPolicyDaggerRunner` prints that it is not loading
a teacher and disables the KL loss. That is still a valid future-policy RL run.
For distillation, pass a real teacher experiment name through the appropriate
script argument so the runner can load the teacher checkpoint.
