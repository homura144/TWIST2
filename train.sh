#!/bin/bash

# Usage: bash train.sh <experiment_id> <device>

# bash train.sh 1103_twist2 cuda:0

if [ -f /root/miniconda3/etc/profile.d/conda.sh ]; then
    source /root/miniconda3/etc/profile.d/conda.sh
    conda activate twist2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOTION_CONFIG_DIR="${SCRIPT_DIR}/legged_gym/motion_data_configs"

robot_name="l7"
exptid=$1
device=$2
shift 2
python_bin="${PYTHON:-python}"

task_name="${robot_name}_stu_future"
proj_name="${robot_name}_stu_future"

resolve_motion_file() {
    local value="$1"
    if [[ "${value}" == */* ]]; then
        printf '%s\n' "${value}"
    elif [[ "${value}" == *.yaml ]]; then
        printf '%s\n' "${MOTION_CONFIG_DIR}/${value}"
    else
        printf '%s\n' "${MOTION_CONFIG_DIR}/l7_lafan1_${value}.yaml"
    fi
}

args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --motion_file)
            args+=("$1" "$(resolve_motion_file "$2")")
            shift 2
            ;;
        --motion_file=*)
            args+=("--motion_file" "$(resolve_motion_file "${1#*=}")")
            shift
            ;;
        *)
            args+=("$1")
            shift
            ;;
    esac
done

cd "${SCRIPT_DIR}/legged_gym/legged_gym/scripts"

# Run the training script
"${python_bin}" train.py --task "${task_name}" \
                --proj_name "${proj_name}" \
                --exptid "${exptid}" \
                --device "${device}" \
                --teacher_exptid "None" \
                "${args[@]}"
