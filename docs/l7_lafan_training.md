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

The retargeted pickle is already in the format expected by
`pose.utils.motion_lib_pkl.MotionLib`: a YAML file points at one or more PKL
files, `HumanoidMimic._load_motions()` creates the motion library, and the env
samples reference root pose, joint pose, body positions, and velocities from that
library.

## Training Config Adaptation

The main project now has dataset YAMLs under `legged_gym/motion_data_configs/`:

| YAML | Purpose |
| --- | --- |
| `l7_lafan1_all.yaml` | All 37 motions. |
| `l7_lafan1_dance.yaml` | 7 dance motions. |
| `l7_lafan1_walk.yaml` | 10 walk motions. |
| `l7_lafan1_run.yaml` | 4 run motions. |
| `l7_lafan1_*.yaml` | Other per-category subsets. |

The root `train.sh` passes arbitrary extra arguments through to
`legged_gym/legged_gym/scripts/train.py`. The added `--motion_file` CLI option
overrides `env_cfg.motion.motion_file`, so L7 training can switch from the
default AMASS config to LAFAN without changing Python config classes:

```bash
bash train.sh l7_lafan_all cuda:0 \
  --motion_file /cephfs/hesixiao/TWIST2/legged_gym/motion_data_configs/l7_lafan1_all.yaml
```

The TWIST1 subtree has a matching all-motion YAML:

```bash
/cephfs/hesixiao/TWIST2/TWIST1/legged_gym/motion_data_configs/lafan1_l7_filtered.yaml
```

Its `train_teacher.sh` and `train_student.sh` now activate the `twist2` env when
available, set local `PYTHONPATH`, keep wandb enabled by default, and pass extra
arguments through to `train.py`.

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

For L7 future training:

- `motion_obs` is one target step of `6 + 29 = 35`.
- `proprio` is `3 + 2 + 3 * 29 = 92`.
- `future` is one future target step of `6 + 29 = 35`.
- The actor outputs a Gaussian mean for 29 actions; sampled actions and
  inference actions both have shape `(num_envs, 29)`.

The critic input is the privileged observation of width 1734. The first
`n_priv_mimic_obs` entries are privileged reference-motion observations. The
critic encodes those through `MotionEncoder`, then concatenates:

```text
critic_observations[:, n_priv_mimic_obs:]
current single motion observation
encoded motion latent
```

The critic output is a scalar value with shape `(num_envs, 1)`.

If no teacher run is passed, `OnPolicyDaggerRunner` prints that it is not loading
a teacher and disables the KL loss. That is still a valid future-policy RL run.
For distillation, pass a real teacher experiment name through the appropriate
script argument so the runner can load the teacher checkpoint.

## TWIST1 Teacher And Student I/O

The TWIST1 subtree uses `l7_*` task names, and the controlled action width is 29.

Teacher task:

```bash
bash train_teacher.sh wbc_lafan_teacher cuda:0 \
  --motion_file /cephfs/hesixiao/TWIST2/TWIST1/legged_gym/motion_data_configs/lafan1_l7_filtered.yaml
```

The teacher uses `l7_priv_mimic` with `obs_type = "priv"`:

```text
n_priv_mimic_obs = 20 * (8 + 29 + 3 * 9) = 1280
n_proprio = 3 + 2 + 3 * 29 = 92
n_priv_info = 3 + 1 + 3 * 9 + 2 + 4 + 1 + 2 * 29 = 96
joint_pos_error = 29
teacher obs = privileged obs = 1280 + 92 + 96 + 29 = 1497
teacher action = 29
```

Student task:

```bash
bash train_student.sh wbc_lafan_student <teacher_run_name> cuda:0 \
  --motion_file /cephfs/hesixiao/TWIST2/TWIST1/legged_gym/motion_data_configs/lafan1_l7_filtered.yaml
```

The student uses `l7_stu_rl` with `obs_type = "student"`:

```text
n_mimic_obs = 8 + 29 = 37
n_proprio = 92
joint_pos_error = 29
n_obs_single = 37 + 92 + 29 = 158
history_len = 10
student obs = 158 * (10 + 1) = 1738
student privileged obs = 1497
student action = 29
```

The student actor consumes the student observation/history and outputs 29
actions. The critic still receives the privileged observation. During DAgger
distillation the loaded teacher provides action targets for the KL/imitation
term.

## Wandb And Launch Commands

Do not pass `--no_wandb` for real runs. The scripts keep wandb enabled by
default.

Main all-motion run:

```bash
nohup bash train.sh l7_lafan_all_<timestamp> cuda:0 \
  --motion_file /cephfs/hesixiao/TWIST2/legged_gym/motion_data_configs/l7_lafan1_all.yaml \
  > logs/l7_lafan_all_<timestamp>.out 2>&1 &
```

Main per-category run:

```bash
nohup bash train.sh l7_lafan_dance_<timestamp> cuda:0 --num_envs 1024 \
  --motion_file /cephfs/hesixiao/TWIST2/legged_gym/motion_data_configs/l7_lafan1_dance.yaml \
  > logs/l7_lafan_dance_<timestamp>.out 2>&1 &
```

WBC teacher:

```bash
cd /cephfs/hesixiao/TWIST2/TWIST1
nohup bash train_teacher.sh wbc_lafan_teacher_<timestamp> cuda:0 \
  --motion_file /cephfs/hesixiao/TWIST2/TWIST1/legged_gym/motion_data_configs/lafan1_l7_filtered.yaml \
  > logs/wbc_lafan_teacher_<timestamp>.out 2>&1 &
```

WBC student after a teacher checkpoint exists:

```bash
cd /cephfs/hesixiao/TWIST2/TWIST1
nohup bash train_student.sh wbc_lafan_student_<timestamp> wbc_lafan_teacher_<timestamp> cuda:0 \
  --motion_file /cephfs/hesixiao/TWIST2/TWIST1/legged_gym/motion_data_configs/lafan1_l7_filtered.yaml \
  > logs/wbc_lafan_student_<timestamp>.out 2>&1 &
```
