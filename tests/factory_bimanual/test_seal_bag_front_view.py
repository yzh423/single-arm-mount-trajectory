from scripts.render_seal_bag_front_view import report_directory_for_robot


def test_front_view_uses_the_selected_robot_report_directory():
    assert report_directory_for_robot("piperx").name == "seal_bag_dual_piperx"
