from legged_gym.envs.g1.g1_mimic_distill_config import G1MimicPrivCfg, G1MimicPrivCfgPPO
from legged_gym import LEGGED_GYM_ROOT_DIR


class L7MimicPrivCfg(G1MimicPrivCfg):
    class env(G1MimicPrivCfg.env):
        num_envs = 4096
        num_actions = 29
        obs_type = 'priv'
        dof_err_w = [0.6, 0.8, 1.0, 1.0, 2.0, 0.6,
                     0.6, 0.8, 1.0, 1.0, 2.0, 0.6,
                     1.0, 2.0, 3.0,
                     1.6, 1.3, 0.8, 1.0, 0.8, 0.6, 0.6,
                     1.6, 1.3, 0.8, 1.0, 0.8, 0.6, 0.6]

    class init_state(G1MimicPrivCfg.init_state):
        pos = [0, 0, 1.1]
        default_joint_angles = {
            'left_hip_roll_joint': 0.0,
            'left_hip_yaw_joint': 0.0,
            'left_hip_pitch_joint': 0.0,
            'left_knee_joint': 0.0,
            'left_ankle_pitch_joint': 0.0,
            'left_ankle_roll_joint': 0.0,
            'right_hip_roll_joint': 0.0,
            'right_hip_yaw_joint': 0.0,
            'right_hip_pitch_joint': 0.0,
            'right_knee_joint': 0.0,
            'right_ankle_pitch_joint': 0.0,
            'right_ankle_roll_joint': 0.0,
            'waist_yaw_joint': 0.0,
            'waist_roll_joint': 0.0,
            'waist_pitch_joint': 0.0,
            'left_shoulder_pitch_joint': 0.0,
            'left_shoulder_roll_joint': 0.0,
            'left_arm_yaw_joint': 0.0,
            'left_elbow_pitch_joint': 0.0,
            'left_elbow_yaw_joint': 0.0,
            'left_wrist_pitch_joint': 0.0,
            'left_wrist_roll_joint': 0.0,
            'right_shoulder_pitch_joint': 0.0,
            'right_shoulder_roll_joint': 0.0,
            'right_arm_yaw_joint': 0.0,
            'right_elbow_pitch_joint': 0.0,
            'right_elbow_yaw_joint': 0.0,
            'right_wrist_pitch_joint': 0.0,
            'right_wrist_roll_joint': 0.0,
        }

    class control(G1MimicPrivCfg.control):
        stiffness = {'hip_yaw': 347.2,
                     'hip_roll': 647.1,
                     'hip_pitch': 381.9,
                     'knee': 381.9,
                     'ankle': 88.8,
                     'waist_roll': 394.7,
                     'waist_yaw': 197.35,
                     'waist_pitch': 394.7,
                     'shoulder': 39.47,
                     'elbow_pitch': 39.47,
                     'elbow_yaw': 39.47,
                     'wrist': 39.47,
                     'arm': 39.47}
        damping = {'hip_yaw': 22.15,
                   'hip_roll': 41.19,
                   'hip_pitch': 24.3,
                   'knee': 24.3,
                   'ankle': 5.65,
                   'waist_roll': 25.13,
                   'waist_yaw': 12.565,
                   'waist_pitch': 25.13,
                   'shoulder': 2.513,
                   'elbow_pitch': 2.513,
                   'elbow_yaw': 2.513,
                   'wrist': 2.513,
                   'arm': 2.513}
        action_scale = 0.25
        decimation = 10

    class asset(G1MimicPrivCfg.asset):
        file = f'{LEGGED_GYM_ROOT_DIR}/../assets/l7/l7_29dof_neck_fixed.urdf'
        torso_name = 'torso_link'
        chest_name = 'torso_link'
        thigh_name = 'hip'
        shank_name = 'knee'
        foot_name = 'ankle_roll_link'
        waist_name = ['waist_roll_link', 'waist_yaw_link']
        upper_arm_name = 'shoulder_pitch_link'
        lower_arm_name = 'elbow_pitch_link'
        hand_name = ['left_wrist_roll_link', 'right_wrist_roll_link']
        feet_bodies = ['left_ankle_roll_link', 'right_ankle_roll_link']
        n_lower_body_dofs = 12
        penalize_contacts_on = ["shoulder", "elbow", "hip", "knee", "ankle", "torso_link"]
        terminate_after_contacts_on = ['torso_link']
        dof_armature = [0.169610625, 0.085686055, 0.102749511, 0.102749511, 0.023523277, 0.023523277] * 2 + [0.137] * 7 + [0.01] * 3 + [0.137] * 4 + [0.01] * 3
        collapse_fixed_joints = True

    class motion(G1MimicPrivCfg.motion):
        key_bodies = ["left_wrist_roll_link", "right_wrist_roll_link", "left_ankle_roll_link", "right_ankle_roll_link", "left_knee_link", "right_knee_link", "left_shoulder_roll_link", "right_shoulder_roll_link", "torso_link"]
        upper_key_bodies = ["left_wrist_roll_link", "right_wrist_roll_link", "left_shoulder_roll_link", "right_shoulder_roll_link", "torso_link"]
        motion_file = f"{LEGGED_GYM_ROOT_DIR}/motion_data_configs/l7_amass_dataset.yaml"
        sample_ratio = 1.0
        motion_smooth = True
        motion_decompose = False


class L7MimicPrivCfgPPO(G1MimicPrivCfgPPO):
    pass
