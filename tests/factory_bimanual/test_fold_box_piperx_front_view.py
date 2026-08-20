from scripts import render_fold_box_piperx_front_view as renderer


def test_true_front_view_is_normal_to_actual_mount_baseline():
    azimuth = renderer.front_azimuth_from_mount(
        [-0.2312103678324848, -0.0130977167988798],
        [0.0412103678324847, -0.3569022832011202])
    assert abs(azimuth - 218.392) < .01
    assert renderer.FRONT_ELEVATION_DEG == -12.0
    assert renderer.OUTPUT.name.endswith("true_front_720p.mp4")
