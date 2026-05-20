from legged_gym.envs.l7.l7_mimic_distill_config import L7MimicPrivCfg, L7MimicPrivCfgPPO
from legged_gym.envs.g1.g1_mimic_future_config import G1MimicStuFutureCfg, G1MimicStuFutureCfgDAgger, TAR_MOTION_STEPS_FUTURE
from legged_gym.envs.base.humanoid_mimic_config import HumanoidMimicCfgPPO
from legged_gym import LEGGED_GYM_ROOT_DIR


class L7MimicStuFutureCfg(L7MimicPrivCfg):
    class env(L7MimicPrivCfg.env):
        obs_type = 'student_future'
        tar_motion_steps = [0]
        tar_motion_steps_future = TAR_MOTION_STEPS_FUTURE

        n_mimic_obs_single = 6 + 29
        n_mimic_obs = len(tar_motion_steps) * n_mimic_obs_single
        n_proprio = L7MimicPrivCfg.env.n_proprio
        n_future_obs_single = 6 + 29
        n_future_obs = len(tar_motion_steps_future) * n_future_obs_single
        n_obs_single = n_mimic_obs + n_proprio
        num_observations = n_obs_single * (L7MimicPrivCfg.env.history_len + 1) + n_future_obs

        enable_force_curriculum = False
        force_curriculum = G1MimicStuFutureCfg.env.force_curriculum

    class motion(L7MimicPrivCfg.motion):
        motion_file = f"{LEGGED_GYM_ROOT_DIR}/motion_data_configs/l7_amass_dataset.yaml"
        motion_curriculum = True
        motion_curriculum_gamma = 0.01
        motion_decompose = False
        motion_dr_enabled = False
        root_position_noise = [0.01, 0.05]
        root_orientation_noise = [0.1, 0.2]
        root_velocity_noise = [0.05, 0.1]
        joint_position_noise = [0.05, 0.1]
        motion_dr_resampling = True
        use_error_aware_sampling = False
        error_sampling_power = 5.0
        error_sampling_threshold = 0.15

    class rewards(L7MimicPrivCfg.rewards):
        scales = G1MimicStuFutureCfg.rewards.scales


class L7MimicStuFutureCfgDAgger(L7MimicStuFutureCfg):
    seed = 1

    class teachercfg(L7MimicPrivCfgPPO):
        pass

    class runner(L7MimicPrivCfgPPO.runner):
        policy_class_name = 'ActorCriticFuture'
        algorithm_class_name = 'DaggerPPO'
        runner_class_name = 'OnPolicyDaggerRunner'
        max_iterations = 30_001
        warm_iters = 100
        save_interval = 500
        experiment_name = 'test'
        run_name = ''
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
        teacher_experiment_name = 'test'
        teacher_proj_name = 'l7_priv_mimic'
        teacher_checkpoint = -1
        eval_student = False
        save_to_wandb = False

    class algorithm(HumanoidMimicCfgPPO.algorithm):
        grad_penalty_coef_schedule = [0.00, 0.00, 700, 1000]
        std_schedule = [1.0, 0.4, 4000, 1500]
        entropy_coef = 0.005
        dagger_coef_anneal_steps = 60000
        dagger_coef = 0.2
        dagger_coef_min = 0.1
        future_weight_decay = 0.95
        future_consistency_loss = 0.1

    class policy(G1MimicStuFutureCfgDAgger.policy):
        num_future_observations = L7MimicStuFutureCfg.env.n_future_obs
        num_future_steps = len(TAR_MOTION_STEPS_FUTURE)
