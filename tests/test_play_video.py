import numpy as np

from legged_gym.scripts.play_l7 import _compose_side_by_side_frame, _reference_motion_time
import torch


def test_compose_side_by_side_frame_keeps_reference_on_left_and_policy_on_right():
    reference = np.full((2, 3, 4), [255, 0, 0, 255], dtype=np.uint8)
    policy = np.full((2, 3, 4), [0, 255, 0, 255], dtype=np.uint8)

    frame = _compose_side_by_side_frame(reference, policy)

    assert frame.shape == (2, 6, 3)
    assert np.all(frame[:, :3] == [255, 0, 0])
    assert np.all(frame[:, 3:] == [0, 255, 0])


def test_reference_motion_clock_ignores_policy_episode_reset():
    clock = {"start_time": torch.tensor([0.4])}

    before_reset = _reference_motion_time(clock, step_idx=10, dt=0.02)
    after_policy_reset = _reference_motion_time(clock, step_idx=11, dt=0.02)

    assert torch.allclose(before_reset, torch.tensor([0.6]))
    assert torch.allclose(after_policy_reset, torch.tensor([0.62]))
