#!/bin/bash

# Usage: bash train.sh <experiment_id> <device>

# bash train.sh 1103_twist2 cuda:0

if [ -f /root/miniconda3/etc/profile.d/conda.sh ]; then
    source /root/miniconda3/etc/profile.d/conda.sh
    conda activate twist2
fi

cd legged_gym/legged_gym/scripts

robot_name="l7"
exptid=$1
device=$2
shift 2
python_bin="${PYTHON:-python}"

task_name="${robot_name}_stu_future"
proj_name="${robot_name}_stu_future"


# Run the training script
"${python_bin}" train.py --task "${task_name}" \
                --proj_name "${proj_name}" \
                --exptid "${exptid}" \
                --device "${device}" \
                --teacher_exptid "None" \
                "$@"
