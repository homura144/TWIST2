#!/bin/bash

# Evaluate an L7 policy checkpoint in Isaac Gym and optionally record a video.
#
# Usage:
#   bash eval_l7.sh <run_name> <device> [motion_yaml_or_alias] [checkpoint_file_or_number] [record_video] [eval_steps] [eval_start]
#
# Examples:
#   bash eval_l7.sh l7_lafan_all cuda:0
#   bash eval_l7.sh l7_lafan_dance cuda:0 dance model_2000.pt true 1000

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -f /root/miniconda3/etc/profile.d/conda.sh ]; then
    source /root/miniconda3/etc/profile.d/conda.sh
    conda activate twist2
fi

resolve_motion_file() {
    local value="$1"
    if [[ "${value}" == */* ]]; then
        printf '%s\n' "${value}"
    elif [[ "${value}" == *.yaml ]]; then
        printf '%s\n' "${SCRIPT_DIR}/legged_gym/motion_data_configs/${value}"
    else
        printf '%s\n' "${SCRIPT_DIR}/legged_gym/motion_data_configs/l7_lafan1_${value}.yaml"
    fi
}

resolve_checkpoint() {
    local value="$1"
    if [[ "${value}" == */* ]] || [[ "${value}" == "-1" ]] || [[ "${value}" =~ ^[0-9]+$ ]]; then
        printf '%s\n' "${value}"
    elif [[ "${value}" == model_*.pt ]]; then
        printf '%s\n' "${SCRIPT_DIR}/legged_gym/logs/l7_stu_future/${exptid}/${value}"
    else
        printf '%s\n' "${value}"
    fi
}

exptid=${1:?Usage: bash eval_l7.sh <run_name> <device> [motion_yaml_or_alias] [checkpoint_file_or_number] [record_video] [eval_steps] [eval_start]}
device=${2:?Usage: bash eval_l7.sh <run_name> <device> [motion_yaml_or_alias] [checkpoint_file_or_number] [record_video] [eval_steps] [eval_start]}
motion_file=$(resolve_motion_file "${3:-all}")
checkpoint=${4:--1}
checkpoint=$(resolve_checkpoint "${checkpoint}")
record_video=${5:-true}
eval_steps=${6:-1000}
eval_start=${7:-zero}
num_envs=1
if [ "${record_video}" = "true" ] || [ "${record_video}" = "1" ] || [ "${record_video}" = "yes" ]; then
    num_envs=2
fi

task_name="l7_stu_future"
proj_name="l7_stu_future"

cd "${SCRIPT_DIR}/legged_gym/legged_gym/scripts"

args=(
    --task "${task_name}"
    --proj_name "${proj_name}"
    --exptid "${exptid}"
    --resumeid "${exptid}"
    --checkpoint "${checkpoint}"
    --teacher_exptid "None"
    --num_envs "${num_envs}"
    --device "${device}"
    --motion_file "${motion_file}"
    --eval_steps "${eval_steps}"
    --eval_start "${eval_start}"
)

if [ "${record_video}" = "true" ] || [ "${record_video}" = "1" ] || [ "${record_video}" = "yes" ]; then
    args+=(--record_video --headless)
else
    args+=(--headless)
fi

echo "Evaluating L7 policy"
echo "  run: ${exptid}"
echo "  checkpoint: ${checkpoint}"
echo "  motion_file: ${motion_file}"
echo "  device: ${device}"
echo "  record_video: ${record_video}"
echo "  eval_steps: ${eval_steps}"
echo "  eval_start: ${eval_start}"
echo "  video_dir: ${SCRIPT_DIR}/logs/videos/motion_tracking/${exptid}"

python play_l7.py "${args[@]}"
