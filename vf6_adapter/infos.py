"""Validate the one supported VF6 NuScenes-info contract without MMCV/CUDA."""

from pathlib import Path

import numpy as np
from pyquaternion import Quaternion

from . import DATASET_VERSION, INFO_VERSION
from .cameras import camera_matrices


def validate_metadata(metadata):
    expected = dict(dataset="VF6", version=DATASET_VERSION, info_version=INFO_VERSION,
                    schema="bevfusion_nuscenes_infos", coordinate_frame="ego",
                    box_convention="bottom_lwh_cw", timestamp_unit="microseconds",
                    global_pose_status="unavailable_identity_placeholder", single_sweep=True)
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError("Incompatible VF6 {}. Regenerate raw data with tools/create_vf6_data.py (info v2)".format(key))
    classes = metadata.get("classes", [])
    if not classes or len(set(classes)) != len(classes):
        raise ValueError("VF6 classes must be nonempty and unique")
    names = metadata.get("camera_names", [])
    if len(set(names)) != len(names):
        raise ValueError("Camera names must be unique")
    if names and metadata.get("image_mode") != "rectified_pinhole":
        raise ValueError("Camera fusion requires rectified_pinhole metadata")


def validate_info(info, metadata):
    """Check fields actually consumed by inherited loading plus VF6 provenance."""
    if not info["token"] or not isinstance(info["timestamp"], (int, np.integer)):
        raise ValueError("Expected a token and integer microsecond timestamp")
    if info["timestamp"] != info["timestamp_ns"] // 1000 or info["sweeps"] != []:
        raise ValueError("VF6 requires synchronized integer timestamps and one sweep")
    if not info["lidar_path"]:
        raise ValueError("Missing point cloud path")
    for prefix in ("lidar2ego", "ego2global"):
        q = np.asarray(info[prefix + "_rotation"], dtype=float)
        t = np.asarray(info[prefix + "_translation"], dtype=float)
        if (q.shape != (4,) or t.shape != (3,) or not np.isfinite(q).all()
                or not np.isfinite(t).all() or not np.isclose(np.linalg.norm(q), 1, atol=1e-5)):
            raise ValueError("Invalid {} quaternion/translation".format(prefix))
        if not np.allclose(Quaternion(q).rotation_matrix, np.eye(3), atol=1e-6) or np.any(t != 0):
            raise ValueError("VF6 v2 expects identity {} with explicit local-only pose metadata".format(prefix))
    boxes = np.asarray(info["gt_boxes"])
    n = len(boxes)
    if boxes.shape != (n, 7) or not np.isfinite(boxes).all() or (boxes[:, 3:6] <= 0).any():
        raise ValueError("Invalid bottom-center/lwh/CW boxes")
    for name in ("gt_names", "num_lidar_pts", "valid_flag", "label_attributes", "track_ids"):
        if np.asarray(info[name]).shape != (n,):
            raise ValueError("{} does not align with boxes".format(name))
    if np.asarray(info["valid_flag"]).dtype != np.bool_:
        raise ValueError("valid_flag must be a boolean array")
    if not np.isin(info["gt_names"], metadata["classes"]).all():
        raise ValueError("Unexpected class name")
    if (np.asarray(info["num_lidar_pts"]) < 0).any():
        raise ValueError("Negative LiDAR point count")
    if not np.array_equal(info["valid_flag"], np.asarray(info["num_lidar_pts"]) > 0):
        raise ValueError("Export valid_flag must match num_lidar_pts > 0")
    names = metadata.get("camera_names", [])
    if list(info.get("cams", {})) != names:
        raise ValueError("Camera order/count differs from metadata")
    for name in names:
        camera = info["cams"][name]
        if camera["timestamp"] != camera["timestamp_ns"] // 1000:
            raise ValueError("Invalid camera timestamp unit")
        delta = camera["timestamp_ns"] - info["timestamp_ns"]
        if camera["time_offset_ns"] != delta or abs(delta) / 1e6 > metadata["max_camera_dt_ms"]:
            raise ValueError("Camera timestamp exceeds synchronization tolerance")
        matrices = camera_matrices(camera)
        if not camera["data_path"] or not np.allclose(matrices["camera2ego"], matrices["camera2lidar"], atol=1e-5):
            raise ValueError("Camera transforms must agree with ego-frame exported points")
        if not np.allclose(matrices["lidar2camera"], camera["ego2camera"], atol=1e-3):
            raise ValueError("Camera extrinsics disagree with raw calibration")


def resolve_info_paths(info, dataset_root, camera_names):
    """Return a shallow record copy; never persist host paths back into the PKL."""
    def absolute(path):
        return str((Path(dataset_root) / path).resolve())
    resolved = dict(info, lidar_path=absolute(info["lidar_path"]))
    if camera_names:
        resolved["cams"] = {name: dict(info["cams"][name], data_path=absolute(info["cams"][name]["data_path"]))
                            for name in camera_names}
    return resolved
