import numpy as np

from legged_gym.scripts.play_l7 import _compose_side_by_side_frame


def test_compose_side_by_side_frame_keeps_reference_on_left_and_policy_on_right():
    reference = np.full((2, 3, 4), [255, 0, 0, 255], dtype=np.uint8)
    policy = np.full((2, 3, 4), [0, 255, 0, 255], dtype=np.uint8)

    frame = _compose_side_by_side_frame(reference, policy)

    assert frame.shape == (2, 6, 3)
    assert np.all(frame[:, :3] == [255, 0, 0])
    assert np.all(frame[:, 3:] == [0, 255, 0])
