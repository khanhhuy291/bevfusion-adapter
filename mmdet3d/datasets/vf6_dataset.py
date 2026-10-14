"""Single-sweep VF6 camera/LiDAR detection without nuScenes database tables."""

from pathlib import Path
import tempfile

import mmcv
import numpy as np
from mmdet.datasets import DATASETS

from mmdet3d.core.bbox import LiDARInstance3DBoxes
from vf6_adapter import CLASSES, INFO_VERSION
from vf6_adapter.evaluation import evaluate_detection
from vf6_adapter.geometry import from_repo_boxes, in_bev_range, to_repo_boxes
from vf6_adapter.cameras import camera_matrices

from .custom_3d import Custom3DDataset


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


@DATASETS.register_module()
class VF6Dataset(Custom3DDataset):
    """Points are already in ego axes (x forward, y left, z up).

    Metadata boxes are geometric-center/lwh/CCW. Convert to this checkout's
    bottom-center/lwh/CW convention at the Dataset boundary, exactly once.
    Unknown label attributes and object IDs remain in metadata, not targets.
    """

    CLASSES = CLASSES

    def __init__(self, ann_file, dataset_root, pipeline=None, object_classes=None,
                 modality=None, box_type_3d="LiDAR", filter_empty_gt=True,
                 test_mode=False, min_lidar_points=1, point_cloud_range=None,
                 load_interval=1, allow_smoke_test=False, camera_names=None):
        if box_type_3d != "LiDAR":
            raise ValueError("VF6Dataset currently supports LiDAR box axes only")
        modality = modality or dict(use_lidar=True, use_camera=False, use_radar=False)
        if modality.get("use_radar"):
            raise ValueError("VF6 radar fusion is not implemented")
        if not modality.get("use_lidar", True):
            raise ValueError("VF6Dataset requires use_lidar=True")
        if min_lidar_points < 0 or load_interval < 1:
            raise ValueError("min_lidar_points must be >=0 and load_interval must be >=1")
        self.min_lidar_points = min_lidar_points
        self.point_cloud_range = point_cloud_range
        self.load_interval = load_interval
        self.allow_smoke_test = allow_smoke_test
        self.camera_names = camera_names
        super().__init__(dataset_root=dataset_root, ann_file=ann_file, pipeline=pipeline,
                         classes=object_classes, modality=modality, box_type_3d=box_type_3d,
                         filter_empty_gt=filter_empty_gt, test_mode=test_mode)

    def load_annotations(self, ann_file):
        data = mmcv.load(ann_file)
        self.metadata = data["metadata"]
        if (self.metadata.get("dataset") != "VF6"
                or self.metadata.get("info_version") != INFO_VERSION
                or self.metadata.get("coordinate_frame") != "ego"
                or self.metadata.get("box_convention") != "center_lwh_ccw"):
            raise ValueError("Incompatible VF6 metadata; regenerate using tools/create_vf6_data.py")
        if list(self.CLASSES) != self.metadata["classes"]:
            raise ValueError("Config class order must match the converter metadata")
        if self.metadata.get("smoke_test") and not self.allow_smoke_test:
            raise ValueError("Smoke metadata has overlapping train/val. Use an explicit smoke config only for debugging")
        if self.modality.get("use_camera"):
            if self.metadata.get("image_mode") != "rectified_pinhole" or not self.metadata.get("camera_names"):
                raise ValueError("Camera fusion requires metadata exported with --with-camera")
            self.camera_names = list(self.camera_names or self.metadata["camera_names"])
            if (len(set(self.camera_names)) != len(self.camera_names)
                    or not set(self.camera_names).issubset(self.metadata["camera_names"])):
                raise ValueError("Requested camera_names do not match exported metadata")
            for info in data["infos"]:
                if not set(self.camera_names).issubset(info.get("cams", {})):
                    raise ValueError("Missing camera records in sample {}".format(info["token"]))
        infos = sorted(data["infos"], key=lambda frame: (frame["session_id"], frame["timestamp_ns"], frame["token"]))
        infos = infos[::self.load_interval]
        if self.filter_empty_gt and not self.test_mode:
            infos = [frame for frame in infos if self._annotation_mask(frame).any()]
        if not infos:
            raise ValueError("VF6 dataset has no usable samples after annotation filtering")
        return infos

    def _annotation_mask(self, info):
        return ((np.asarray(info["num_lidar_pts"]) >= self.min_lidar_points)
                & np.isin(info["gt_names"], self.CLASSES)
                & in_bev_range(info["gt_boxes"], self.point_cloud_range))

    def get_cat_ids(self, index):
        info = self.data_infos[index]
        names = np.asarray(info["gt_names"])[self._annotation_mask(info)]
        return [self.cat2id[name] for name in sorted(set(names))]

    def get_data_info(self, index):
        info = self.data_infos[index]
        data = dict(
            token=info["token"], sample_idx=info["token"], timestamp=info["timestamp"],
            lidar_path=str(Path(self.dataset_root) / info["lidar_path"]), sweeps=[],
            # Exported points are already ego-frame points. Do not apply the raw
            # LiDAR extrinsics a second time.
            lidar2ego=np.eye(4, dtype=np.float32),
            lidar_aug_matrix=np.eye(4, dtype=np.float32),
        )
        if not self.test_mode:
            data["ann_info"] = self.get_ann_info(index)
        if self.modality.get("use_camera"):
            data["image_paths"] = []
            for key in ("camera_intrinsics", "camera2ego", "camera2lidar", "lidar2camera", "lidar2image"):
                data[key] = []
            for name in self.camera_names:
                camera = info["cams"][name]
                data["image_paths"].append(str(Path(self.dataset_root) / camera["data_path"]))
                for key, matrix in camera_matrices(camera).items():
                    data[key].append(matrix)
            data["img_aug_matrix"] = [np.eye(4, dtype=np.float32) for _ in self.camera_names]
        return data

    def get_ann_info(self, index):
        info = self.data_infos[index]
        mask = self._annotation_mask(info)
        names = np.asarray(info["gt_names"])[mask]
        boxes = LiDARInstance3DBoxes(to_repo_boxes(info["gt_boxes"][mask]), box_dim=7)
        return dict(gt_bboxes_3d=boxes,
                    gt_labels_3d=np.asarray([self.cat2id[name] for name in names], dtype=np.int64),
                    gt_names=names)

    def evaluate(self, results, metric="bbox", iou_threshold=0.5, logger=None, **kwargs):
        metrics = [metric] if isinstance(metric, str) else list(metric)
        if not metrics or any(name not in ("bbox", "iou3d") for name in metrics):
            raise ValueError("VF6 supports metric='bbox' or 'iou3d' (custom rotated 3D IoU AP)")
        if len(results) != len(self):
            raise ValueError("Expected {} predictions, got {}".format(len(self), len(results)))
        ground_truth, predictions = [], []
        for info, result in zip(self.data_infos, results):
            mask = self._annotation_mask(info)
            ground_truth.append(dict(
                boxes=info["gt_boxes"][mask],
                labels=np.asarray([self.cat2id[name] for name in info["gt_names"][mask]], dtype=np.int64),
            ))
            result = result.get("pts_bbox", result)
            boxes = result["boxes_3d"]
            boxes = from_repo_boxes(_numpy(boxes.tensor if hasattr(boxes, "tensor") else boxes))
            mask = in_bev_range(boxes, self.point_cloud_range)
            predictions.append(dict(boxes=boxes[mask], scores=_numpy(result["scores_3d"])[mask],
                                    labels=_numpy(result["labels_3d"])[mask]))
        return evaluate_detection(ground_truth, predictions, self.CLASSES, iou_threshold)

    def format_results(self, outputs, pklfile_prefix=None, **kwargs):
        temporary = tempfile.TemporaryDirectory() if pklfile_prefix is None else None
        prefix = Path(temporary.name) / "results" if temporary else Path(pklfile_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        mmcv.dump(outputs, str(prefix) + ".pkl")
        return outputs, temporary
