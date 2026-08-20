import pytest
import torch

from design_optimization.ik import (IKResult, concatenate_ik_branches,
                                    reverse_ik_time)


def _result(offset: float, *, times: int = 3, branches: int = 2) -> IKResult:
    q = torch.arange(times * branches * 6, dtype=torch.float32).reshape(
        1, times, branches, 6) + offset
    scalar = q[..., 0]
    return IKResult(q, scalar, scalar + 1, scalar + 2, scalar > -1)


def test_reverse_ik_time_flips_every_field_only_along_time():
    source = _result(0)
    reversed_result = reverse_ik_time(source)
    torch.testing.assert_close(reversed_result.q, source.q.flip(1))
    torch.testing.assert_close(reversed_result.position_error_m,
                               source.position_error_m.flip(1))
    torch.testing.assert_close(reversed_result.success, source.success.flip(1))


def test_concatenate_ik_branches_preserves_time_and_branch_order():
    first, second = _result(0), _result(100)
    combined = concatenate_ik_branches(first, second)
    assert combined.q.shape == (1, 3, 4, 6)
    torch.testing.assert_close(combined.q[:, :, :2], first.q)
    torch.testing.assert_close(combined.q[:, :, 2:], second.q)


def test_concatenate_ik_branches_rejects_mismatched_time():
    with pytest.raises(ValueError, match="matching design/time"):
        concatenate_ik_branches(_result(0), _result(0, times=4))

