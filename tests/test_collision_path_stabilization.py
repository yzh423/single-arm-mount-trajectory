import numpy as np

from scripts.solve_strict_urdf_task_cache import hold_invalid_frames


def test_hold_invalid_frames_reuses_last_safe_posture():
    q = np.asarray([[0.0], [0.1], [2.9], [-2.5], [0.2]])
    invalid = np.asarray([False, False, True, True, False])

    stable = hold_invalid_frames(q, invalid)

    np.testing.assert_allclose(stable[:, 0], [0.0, 0.1, 0.1, 0.1, 0.2])
