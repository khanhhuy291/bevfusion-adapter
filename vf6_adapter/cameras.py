"""Synchronize VF6 pinhole cameras and export portable rectified images."""

from bisect import bisect_left
from pathlib import Path
import json

import numpy as np
from pyquaternion import Quaternion

from .converter import load_extrinsics, timestamp_ns, validate_rigid_transform

DEFAULT_CAMERAS = ("CAM_P_F", "CAM_P_FL", "CAM_P_FR", "CAM_P_L",
                   "CAM_P_R", "CAM_P_B", "CAM_P_LB", "CAM_P_RB")


def camera_matrices(camera):
    """Offline projection from the same sensor2lidar fields as NuScenesDataset.

    Training uses the inherited loader, not this helper. Exported LiDAR axes
    coincide with ego axes, so sensor2lidar and sensor2ego describe the same RT.
    """
    camera_to_lidar = np.eye(4, dtype=np.float32)
    camera_to_lidar[:3, :3] = camera["sensor2lidar_rotation"]
    camera_to_lidar[:3, 3] = camera["sensor2lidar_translation"]
    validate_rigid_transform(camera_to_lidar, "sensor2lidar")
    # Match the stock loader's 3x3 inverse and row-vector translation arithmetic
    # exactly, including float32 rounding before K @ RT.
    inverse_rotation = np.linalg.inv(camera["sensor2lidar_rotation"])
    inverse_translation = camera["sensor2lidar_translation"] @ inverse_rotation.T
    ego_to_camera = np.eye(4, dtype=np.float32)
    ego_to_camera[:3, :3] = inverse_rotation
    ego_to_camera[:3, 3] = -inverse_translation
    camera_to_ego = np.eye(4, dtype=np.float32)
    camera_to_ego[:3, :3] = Quaternion(camera["sensor2ego_rotation"]).rotation_matrix
    camera_to_ego[:3, 3] = camera["sensor2ego_translation"]
    intrinsic = np.eye(4, dtype=np.float32)
    matrix = np.asarray(camera["cam_intrinsic"], dtype=np.float32)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
        raise ValueError("Invalid rectified camera intrinsics")
    intrinsic[:3, :3] = matrix
    return dict(camera_intrinsics=intrinsic, camera2ego=camera_to_ego,
                camera2lidar=camera_to_lidar, lidar2camera=ego_to_camera,
                lidar2image=intrinsic @ ego_to_camera)


class CameraProcessor:
    def __init__(self, root, clip, camera_names, output_size, max_dt_ms):
        try:
            import cv2
        except ImportError as error:
            raise ImportError("Camera export requires opencv-python-headless; install requirements-vf6.txt") from error
        self.cv2 = cv2
        self.names = list(camera_names)
        if len(output_size) != 2 or any(int(value) != value or value <= 0 for value in output_size):
            raise ValueError("camera_size must contain positive integer width and height")
        self.size = tuple(map(int, output_size))
        if not np.isfinite(max_dt_ms) or max_dt_ms <= 0:
            raise ValueError("max_camera_dt_ms must be positive and finite")
        self.max_dt_ns = int(max_dt_ms * 1e6)
        self.clip = clip
        sensors, profile = load_extrinsics(root, clip["session"])
        intrinsics_path = Path(root) / (profile["vehicle"] + "_Intrinsics.json")
        intrinsics = json.loads(intrinsics_path.read_text())
        self.calibrations, self.indices, self.maps = {}, {}, {}
        for name in self.names:
            if not name.startswith("CAM_P_"):
                raise ValueError("{}: this exporter supports CAM_P_* pinhole cameras; fisheye models must be confirmed separately".format(name))
            if name not in sensors or name not in intrinsics:
                raise ValueError("{} missing calibration in vehicle/profile {}/{}; select cameras explicitly".format(name, profile["vehicle"], profile["profile"]))
            transform = validate_rigid_transform(sensors[name], profile["source"] + ":" + name)
            values = intrinsics[name]
            k = np.asarray(values["camera_matrix"], dtype=np.float64)
            d = np.asarray(values["distortion_coefficients"], dtype=np.float64)
            if (k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0
                    or d.ndim != 1 or len(d) not in (5, 8) or not np.isfinite(d).all()):
                raise ValueError("{}: expected valid pinhole K and 5/8 distortion coefficients".format(name))
            source_size = (int(values["image_width"]), int(values["image_height"]))
            if min(source_size) <= 0:
                raise ValueError("{}: invalid image size".format(name))
            # alpha=0 crops to a valid pinhole field; K' is in exported pixels.
            new_k, _ = cv2.getOptimalNewCameraMatrix(k, d, source_size, 0, self.size)
            if not np.isfinite(new_k).all() or new_k[0, 0] <= 0 or new_k[1, 1] <= 0:
                raise ValueError("{}: failed to produce rectified intrinsics".format(name))
            camera_to_ego = np.linalg.inv(transform)
            quaternion = Quaternion(matrix=camera_to_ego[:3, :3], atol=1e-3, rtol=1e-3).normalised
            self.calibrations[name] = dict(
                cam_intrinsic=new_k.astype(np.float32), ego2camera=transform.astype(np.float32),
                sensor2lidar_rotation=quaternion.rotation_matrix.astype(np.float32),
                sensor2lidar_translation=camera_to_ego[:3, 3].astype(np.float32),
                sensor2ego_rotation=quaternion.elements.tolist(),
                sensor2ego_translation=camera_to_ego[:3, 3].tolist(),
                original_intrinsics=k, original_distortion=d,
                source_image_size=source_size, image_size=self.size,
                image_mode="rectified_pinhole", calibration_profile=profile["profile"],
            )
            directory = clip["path"] / "CAMERA" / name
            files = sorted(path for path in directory.iterdir()
                           if path.is_file() and path.suffix.lower() in (".jpg", ".jpeg", ".png"))
            indexed = sorted((timestamp_ns(path.stem), path) for path in files)
            if not indexed:
                raise ValueError("No camera images in {}".format(directory))
            times = [time for time, _ in indexed]
            if len(set(times)) != len(times):
                raise ValueError("Duplicate image timestamps in {}".format(directory))
            self.indices[name] = (times, [path for _, path in indexed])

    def match(self, target_ns):
        cameras = {}
        for name in self.names:
            times, files = self.indices[name]
            position = bisect_left(times, target_ns)
            candidates = [index for index in (position - 1, position) if 0 <= index < len(times)]
            best = min(candidates, key=lambda index: (abs(times[index] - target_ns), times[index]))
            offset = times[best] - target_ns
            if abs(offset) > self.max_dt_ns:
                raise ValueError("{} {}: nearest image offset {:.3f} ms exceeds {:.3f} ms".format(
                    self.clip["clip"], name, offset / 1e6, self.max_dt_ns / 1e6))
            source = files[best]
            relative = Path("images") / self.clip["session"] / self.clip["clip"] / name / (source.stem + ".jpg")
            cameras[name] = dict(self.calibrations[name], data_path=relative.as_posix(),
                                 timestamp=times[best] // 1000,
                                 raw_data_path=str(source.resolve()), timestamp_ns=times[best], time_offset_ns=offset)
        return cameras

    def export(self, name, camera, output_root):
        cv2 = self.cv2
        calibration = self.calibrations[name]
        image = cv2.imread(camera["raw_data_path"], cv2.IMREAD_COLOR)
        if image is None:
            raise OSError("Cannot decode image: {}".format(camera["raw_data_path"]))
        if (image.shape[1], image.shape[0]) != calibration["source_image_size"]:
            raise ValueError("Image size does not match calibration: {}".format(camera["raw_data_path"]))
        if name not in self.maps:
            self.maps[name] = cv2.initUndistortRectifyMap(
                calibration["original_intrinsics"], calibration["original_distortion"], None,
                calibration["cam_intrinsic"], self.size, cv2.CV_32FC1,
            )
        rectified = cv2.remap(image, *self.maps[name], interpolation=cv2.INTER_LINEAR)
        destination = Path(output_root) / camera["data_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(destination), rectified, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise OSError("Cannot write rectified image: {}".format(destination))

    def clear_cache(self):
        self.maps.clear()
