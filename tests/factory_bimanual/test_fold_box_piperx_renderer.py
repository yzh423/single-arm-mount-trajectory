from scripts import render_factory_dual_piperx_fold_box as renderer


def test_piperx_fold_box_wrapper_uses_refined_upright_mount_and_controller_limit(monkeypatch):
    observed = {}

    def fake_run(robot_name, mount, output, velocity_limit_rad_s):
        observed.update(robot=robot_name, mount=mount, output=output,
                        velocity=velocity_limit_rad_s)
        return observed

    monkeypatch.setattr(renderer, "run_fold_box", fake_run)
    renderer.main()
    assert observed["robot"] == "piperx"
    assert observed["velocity"] == 3.0
    assert observed["mount"]["shared_base_z_m"] == .81
    assert observed["mount"]["yaw"] == {"left": 15.0, "right": 45.0}
    assert observed["mount"]["xy"] == {
        "left": [-0.2312103678324848, -0.0130977167988798],
        "right": [0.0412103678324847, -0.3569022832011202],
    }
    assert observed["mount"]["xy"]["left"] != observed["mount"]["xy"]["right"]
    assert observed["output"].name.endswith("front_720p.mp4")
