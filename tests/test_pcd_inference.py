import numpy as np
import pytest

from vf6_adapter.pcd import prepare_model_points, read_binary_pcd


def write_pcd(path, points, fields=("x", "y", "z", "intensity"), data="binary"):
    points = np.asarray(points, dtype="<f4")
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "# timestamp 123.25\n"
        "VERSION 0.7\n"
        "FIELDS {}\n"
        "SIZE {}\n"
        "TYPE {}\n"
        "COUNT {}\n"
        "WIDTH {}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        "POINTS {}\n"
        "DATA {}\n"
    ).format(" ".join(fields), " ".join(["4"] * len(fields)),
             " ".join(["F"] * len(fields)), " ".join(["1"] * len(fields)),
             len(points), len(points), data)
    with path.open("wb") as stream:
        stream.write(header.encode("ascii"))
        stream.write(points.tobytes())


def test_read_binary_pcd_and_prepare_five_model_features(tmp_path):
    path = tmp_path / "frame.pcd"
    raw = np.array([[1, 2, 3, 4], [-2, 0, 1, 8]], dtype=np.float32)
    write_pcd(path, raw)
    points, metadata = read_binary_pcd(path)
    np.testing.assert_array_equal(points, raw)
    assert metadata == dict(fields=["x", "y", "z", "intensity"], points=2,
                            width=2, height=1, timestamp=123.25, data="binary")
    model = prepare_model_points(points, metadata["fields"], intensity_scale=.5)
    np.testing.assert_array_equal(model, [[1, 2, 3, 2, 0], [-2, 0, 1, 4, 0]])
    assert model.dtype == np.float32 and model.flags.c_contiguous


def test_prepare_applies_transform_then_range_filter():
    raw = np.array([[1, 2, 0, 5], [20, 0, 0, 6]], dtype=np.float32)
    transform = np.eye(4)
    transform[:3, 3] = [1, -1, 2]
    model = prepare_model_points(raw, ["x", "y", "z", "intensity"], transform,
                                 point_cloud_range=[-5, -5, -3, 5, 5, 3])
    np.testing.assert_array_equal(model, [[2, 1, 2, 5, 0]])


def test_reader_rejects_wrong_payload_and_ascii(tmp_path):
    path = tmp_path / "bad.pcd"
    write_pcd(path, [[1, 2, 3, 4]], data="ascii")
    with pytest.raises(ValueError, match="DATA binary"):
        read_binary_pcd(path)
    write_pcd(path, [[1, 2, 3, 4]])
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="payload bytes"):
        read_binary_pcd(path)


def test_prepare_rejects_missing_field_and_empty_range():
    raw = np.array([[1, 2, 3]], dtype=np.float32)
    with pytest.raises(ValueError, match="intensity"):
        prepare_model_points(raw, ["x", "y", "z"])
    raw = np.array([[100, 100, 100, 1]], dtype=np.float32)
    with pytest.raises(ValueError, match="No points remain"):
        prepare_model_points(raw, ["x", "y", "z", "intensity"],
                             point_cloud_range=[-1, -1, -1, 1, 1, 1])
