"""Thin VF6 policy layer over this checkout's NuScenesDataset info loader."""

from pathlib import Path
import tempfile

import mmcv
import numpy as np
from mmdet.datasets import DATASETS

from vf6_adapter import CLASSES
from vf6_adapter.evaluation import evaluate_detection
from vf6_adapter.geometry import from_repo_boxes, in_bev_range
from vf6_adapter.infos import validate_metadata, validate_info, resolve_info_paths

from .nuscenes_dataset import NuScenesDataset


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


@DATASETS.register_module()
class VF6Dataset(NuScenesDataset):
    """Inherit camera matrices, box loading, class mapping and sample pipeline.

    PKL gt_boxes already use bottom-center/lwh/CW, so get_ann_info is inherited
    without conversion. Global poses are explicitly unavailable placeholders;
    only single-frame local detection is supported. Evaluation uses VF6 IoU AP.
    """

    CLASSES = CLASSES

    def __init__(self, ann_file, dataset_root, pipeline=None, object_classes=None,
                 modality=None, box_type_3d="LiDAR", filter_empty_gt=True,
                 test_mode=False, min_lidar_points=1, point_cloud_range=None,
                 load_interval=1, allow_smoke_test=False, camera_names=None,
                 with_velocity=False, use_valid_flag=True):
        if box_type_3d != "LiDAR" or with_velocity or not use_valid_flag:
            raise ValueError("VF6 requires LiDAR boxes, with_velocity=False and use_valid_flag=True")
        modality = modality or dict(use_lidar=True, use_camera=False, use_radar=False)
        if (not modality.get("use_lidar", True) or modality.get("use_radar")
                or modality.get("use_map")):
            raise ValueError("VF6 supports single-frame LiDAR/camera detection, without radar/maps")
        if min_lidar_points < 1 or load_interval < 1:
            raise ValueError("min_lidar_points and load_interval must be >=1")
        self.min_lidar_points = min_lidar_points
        self.point_cloud_range = point_cloud_range
        self.allow_smoke_test = allow_smoke_test
        self.camera_names = camera_names
        super().__init__(ann_file=ann_file, dataset_root=dataset_root, pipeline=pipeline,
                         object_classes=object_classes, modality=modality, box_type_3d=box_type_3d,
                         filter_empty_gt=filter_empty_gt, test_mode=test_mode,
                         load_interval=load_interval, with_velocity=False, use_valid_flag=True)

    def load_annotations(self, ann_file):
        # Reuse nuScenes deserialization, timestamp sorting and load_interval.
        try:
            infos = super().load_annotations(ann_file)
        except KeyError as error:
            raise ValueError("Incompatible VF6 PKL. Regenerate with tools/create_vf6_data.py (info v2)") from error
        validate_metadata(self.metadata)
        if list(self.CLASSES) != self.metadata["classes"]:
            raise ValueError("Config class order must match converter metadata")
        if self.metadata.get("smoke_test") and not self.allow_smoke_test:
            raise ValueError("Smoke metadata has overlapping train/val; use an explicit smoke config")
        exported_names = self.metadata.get("camera_names", [])
        self.camera_names = list(self.camera_names or exported_names) if self.modality.get("use_camera") else []
        if self.modality.get("use_camera") and not self.camera_names:
            raise ValueError("Camera fusion requires an export made with --with-camera")
        if (len(set(self.camera_names)) != len(self.camera_names)
                or not set(self.camera_names).issubset(exported_names)):
            raise ValueError("Requested camera_names do not match exported metadata")
        prepared = []
        for info in infos:
            validate_info(info, self.metadata)
            frame = resolve_info_paths(info, self.dataset_root, self.camera_names)
            # Both inherited get_ann_info/get_cat_ids and our evaluator use this
            # same mask. No additional coordinate conversion occurs here.
            frame["valid_flag"] = (info["valid_flag"]
                                   & (info["num_lidar_pts"] >= self.min_lidar_points)
                                   & in_bev_range(info["gt_boxes"], self.point_cloud_range))
            if self.filter_empty_gt and not self.test_mode and not frame["valid_flag"].any():
                continue
            prepared.append(frame)
        if not prepared:
            raise ValueError("VF6 dataset has no usable samples after annotation filtering")
        if len({frame["token"] for frame in prepared}) != len(prepared):
            raise ValueError("Duplicate VF6 sample tokens")
        return prepared

    def evaluate(self, results, metric="bbox", iou_threshold=0.5, logger=None, **kwargs):
        metrics = [metric] if isinstance(metric, str) else list(metric)
        if not metrics or any(name not in ("bbox", "iou3d") for name in metrics):
            raise ValueError("VF6 supports metric='bbox' or 'iou3d' (custom rotated 3D IoU AP)")
        if len(results) != len(self):
            raise ValueError("Expected {} predictions, got {}".format(len(self), len(results)))
        ground_truth, predictions = [], []
        for info, result in zip(self.data_infos, results):
            mask = info["valid_flag"]
            ground_truth.append(dict(
                boxes=from_repo_boxes(info["gt_boxes"][mask]),
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
