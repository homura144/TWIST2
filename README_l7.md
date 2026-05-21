# L7 LAFAN Commands

This file lists the currently usable L7 LAFAN training and evaluation commands.

## Training

Run all 37 LAFAN motions in the main TWIST2 project:

```bash
bash train.sh l7_lafan_all cuda:0 \
  --motion_file all
```

Run a single motion category. Use the same `--num_envs` as the all-motion run
when comparing reward curves directly; otherwise each iteration contains a
different amount of data.

```bash
bash train.sh l7_lafan_dance cuda:0 --num_envs 4096 \
  --motion_file dance

bash train.sh l7_lafan_walk cuda:0 --num_envs 4096 \
  --motion_file walk

bash train.sh l7_lafan_run cuda:0 --num_envs 4096 \
  --motion_file run
```

Conservative concurrent single-category runs can use fewer envs:

```bash
bash train.sh l7_lafan_dance cuda:0 --num_envs 1024 \
  --motion_file dance
```

`train.sh` maps `--motion_file all`, `dance`, `walk`, `run`, etc. to the
matching `legged_gym/motion_data_configs/l7_lafan1_<name>.yaml` file. You can
also pass a YAML filename such as `l7_lafan1_dance.yaml`.

Available main-project YAMLs:

```text
legged_gym/motion_data_configs/l7_lafan1_all.yaml
legged_gym/motion_data_configs/l7_lafan1_aiming.yaml
legged_gym/motion_data_configs/l7_lafan1_dance.yaml
legged_gym/motion_data_configs/l7_lafan1_fight.yaml
legged_gym/motion_data_configs/l7_lafan1_fightAndSports.yaml
legged_gym/motion_data_configs/l7_lafan1_jumps.yaml
legged_gym/motion_data_configs/l7_lafan1_push.yaml
legged_gym/motion_data_configs/l7_lafan1_run.yaml
legged_gym/motion_data_configs/l7_lafan1_sprint.yaml
legged_gym/motion_data_configs/l7_lafan1_walk.yaml
```

Wandb project for the main project is:

```text
https://wandb.ai/homura_dev/twist2_l7_mimic
```

TWIST1 teacher and student:

```bash
cd /cephfs/hesixiao/TWIST2/TWIST1

bash train_teacher.sh lafan_all cuda:0 \
  --motion_file legged_gym/motion_data_configs/lafan1_l7_filtered.yaml

bash train_student.sh lafan_all lafan_all cuda:0 \
  --motion_file legged_gym/motion_data_configs/lafan1_l7_filtered.yaml
```

TWIST1 wandb projects:

```text
https://wandb.ai/homura_dev/twist1_l7_teacher
https://wandb.ai/homura_dev/twist1_l7_student
```

## Evaluation And Video

Evaluate the latest checkpoint of a run on the all-motion YAML and record video:

```bash
bash eval_l7.sh l7_lafan_all cuda:0
```

Evaluate a specific checkpoint on a specific motion YAML:

```bash
bash eval_l7.sh l7_lafan_dance cuda:0 \
  dance \
  model_500.pt \
  true \
  500
```

The L7 evaluator is isolated in:

```text
legged_gym/legged_gym/scripts/play_l7.py
```

The default `play.py` is kept as the generic project playback script. Use
`eval_l7.sh` for L7 reference-vs-policy videos; it calls `play_l7.py`, creates
two Isaac Gym envs when recording, and writes a single side-by-side MP4 with
the reference motion on the left and the policy rollout on the right.

Defaults:

```text
motion_yaml_or_alias: all
checkpoint_file_or_number: -1, meaning latest model_*.pt
record_video: true
eval_steps: 1000
eval_start: zero
```

Videos are written under:

```text
logs/videos/motion_tracking/<run_name>/
```

`record_video=true` uses Isaac Gym camera sensors in headless mode, so it does
not require opening the interactive GLFW viewer. The script exits immediately
after closing the MP4 writer to avoid an Isaac Gym graphics teardown crash seen
on this server.

The script loads checkpoints from:

```text
legged_gym/logs/l7_stu_future/<run_name>/model_<checkpoint>.pt
```

The motion argument accepts aliases such as `all`, `dance`, `walk`, and `run`,
or a YAML filename such as `l7_lafan1_dance.yaml`. The checkpoint argument
accepts `-1`, a checkpoint number, a filename like `model_500.pt`, or a full
checkpoint path.

```text
model_500.pt
```

`eval_start=zero` starts from the beginning of the selected motion, matching a
real rollout. Use `eval_start=random` only when comparing against training-time
random reference-state initialization.

## L7 Motion Quaternion Format

The L7 retargeted PKL files store `root_rot` as `(w, x, y, z)`, while Isaac Gym
root states and the project quaternion utilities expect `(x, y, z, w)`.
All L7 motion YAMLs therefore include:

```yaml
root_rot_format: wxyz
```

`pose.utils.motion_lib_pkl.MotionLib` converts those root quaternions to
`xyzw` on load. Without this conversion, the first component is interpreted as
the x component instead of w, which makes the robot appear sideways or flipped.

## Why Single-Category Curves Can Look Worse

The current logs show single-category runs have much lower `mean_reward` and
`mean_episode_length` than the all-motion run, but the comparison is not
one-to-one:

| Run | Recent iteration | Total timesteps | Recent mean reward | Recent mean episode length |
| --- | ---: | ---: | ---: | ---: |
| `l7_lafan_all` | 576 | 56.72M | 17.49 | 356.62 |
| `l7_lafan_dance` | 553 | 13.62M | 0.99 | 18.45 |
| `l7_lafan_walk` | 556 | 13.69M | 0.85 | 17.06 |
| `l7_lafan_run` | 557 | 13.71M | 1.19 | 22.95 |

The single-category runs were launched with `--num_envs 1024`, while the
all-motion run used the default L7 value of 4096. At a similar iteration count,
the all-motion run has collected about 4x more samples. It is therefore expected
to be much further along.

Another important detail is that `Mean reward (total)` is accumulated per
episode. When early termination produces short episodes, the total episode
reward is also small. In the current logs, the single-category policies are
still terminating after roughly 15-25 steps, while the all-motion policy often
survives hundreds of steps. The reward gap mostly reflects the episode-length
gap.

For a fair comparison, compare runs at similar total timesteps, or launch the
single-category runs with the same `--num_envs 4096`. Also compare normalized
tracking errors and reward components, not only total episode reward.
