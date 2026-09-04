from scripts import strict_urdf_model_audit as audit


def test_converted_mesh_key_is_content_based_and_line_ending_stable(tmp_path):
    left = tmp_path / "left" / "mesh.dae"
    right = tmp_path / "right" / "mesh.dae"
    left.parent.mkdir()
    right.parent.mkdir()
    content = "<COLLADA>\n<asset/>\n</COLLADA>\n"
    left.write_bytes(content.encode("utf-8"))
    right.write_bytes(content.replace("\n", "\r\n").encode("utf-8"))

    assert audit._stable_source_sha256(left) == audit._stable_source_sha256(right)


def test_resolved_mesh_cache_key_depends_on_content_not_checkout_path():
    xml = "<robot><mesh filename='official_mesh_0000.stl'/></robot>"
    assets = {"official_mesh_0000.stl": b"mesh-bytes"}

    first = audit._resolved_asset_cache_key(xml, assets)
    second = audit._resolved_asset_cache_key(xml, dict(assets))
    changed = audit._resolved_asset_cache_key(
        xml, {"official_mesh_0000.stl": b"changed"})

    assert first == second
    assert first != changed
    assert len(first) == 16
