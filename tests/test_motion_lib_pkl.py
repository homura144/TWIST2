import torch

from pose.utils.motion_lib_pkl import _convert_root_rot_to_xyzw


def test_convert_root_rot_to_xyzw_converts_l7_wxyz_motion_quaternions():
    root_rot = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    converted = _convert_root_rot_to_xyzw(root_rot, "wxyz")

    assert torch.equal(converted, torch.tensor([[2.0, 3.0, 4.0, 1.0]]))
    assert torch.equal(_convert_root_rot_to_xyzw(root_rot, "xyzw"), root_rot)
