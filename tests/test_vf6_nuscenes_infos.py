"""Exercise the actual inherited loader code on CPU without importing CUDA ops.

Only framework services (MMCV I/O/registry, Compose, box-mode dispatch) are
substituted. Box constructors, camera/annotation loading and VF6 policy execute
from the repository sources. Full framework integration remains a GPU check.
"""

import ast
from abc import abstractmethod
import copy
import importlib.util
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np
import pytest
from pyquaternion import Quaternion

from vf6_adapter import CLASSES, DATASET_VERSION
from vf6_adapter.cameras import camera_matrices
from vf6_adapter.converter import create_vf6_infos, read_labels
from vf6_adapter.evaluation import evaluate_detection
from vf6_adapter.geometry import from_repo_boxes, in_bev_range, to_repo_boxes
from vf6_adapter.infos import resolve_info_paths, validate_info, validate_metadata
from test_vf6_adapter import raw_dataset, make_splits
from test_vf6_fusion import camera_dataset, CAMERAS

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def loader_classes():
    torch = pytest.importorskip("torch")
    pytest.importorskip("nuscenes")
    spec = importlib.util.spec_from_file_location("vf6_box_utils", ROOT / "mmdet3d/core/bbox/structures/utils.py")
    utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils)
    def load(path):
        with open(path, "rb") as stream:
            return pickle.load(stream)
    def dump(value, path):
        with open(path, "wb") as stream:
            pickle.dump(value, stream)
    namespace = dict(np=np, torch=torch, Quaternion=Quaternion, Any=Any, Dict=Dict,
                     abstractmethod=abstractmethod, Path=Path, tempfile=tempfile,
                     mmcv=SimpleNamespace(load=load, dump=dump), Dataset=torch.utils.data.Dataset,
                     limit_period=utils.limit_period, rotation_3d_in_axis=utils.rotation_3d_in_axis,
                     CLASSES=CLASSES, evaluate_detection=evaluate_detection,
                     from_repo_boxes=from_repo_boxes, in_bev_range=in_bev_range,
                     validate_info=validate_info, validate_metadata=validate_metadata,
                     resolve_info_paths=resolve_info_paths)
    for relative in ["core/bbox/structures/base_box3d.py", "core/bbox/structures/lidar_box3d.py",
                     "datasets/custom_3d.py", "datasets/nuscenes_dataset.py", "datasets/vf6_dataset.py"]:
        path = ROOT / "mmdet3d" / relative
        nodes = [node for node in ast.parse(path.read_text()).body
                 if isinstance(node, ast.ClassDef) or isinstance(node, ast.FunctionDef) and node.name == "_numpy"]
        for node in nodes:
            node.decorator_list = []
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
        if relative.endswith("lidar_box3d.py"):
            # This suite only requests the unchanged LiDAR mode. Keep the real
            # constructor/corners, avoid loading the framework's mode dispatcher.
            namespace["LiDARInstance3DBoxes"].convert_to = lambda self, mode: self
            namespace["get_box_type"] = lambda name: (namespace["LiDARInstance3DBoxes"], "LiDAR")
    return namespace["NuScenesDataset"], namespace["VF6Dataset"]


@pytest.fixture
def exported(camera_dataset, tmp_path):
    train, val = make_splits(tmp_path)
    output = tmp_path / "processed"
    create_vf6_infos(camera_dataset, output, train, val, camera_names=CAMERAS, camera_size=(32, 24))
    return output


def build(cls, root, split="train", **kwargs):
    options = dict(ann_file=str(root / ("vf6_infos_" + split + ".pkl")), dataset_root=str(root),
                   object_classes=list(CLASSES), modality=dict(use_lidar=True, use_camera=True))
    options.update(kwargs)
    return cls(**options)


def test_stock_loader_consumes_new_pkl_and_vf6_inherits_methods(loader_classes, exported):
    Stock, VF6 = loader_classes
    stock = build(Stock, exported, with_velocity=False, use_valid_flag=True)
    vf6 = build(VF6, exported)
    assert VF6.get_data_info is Stock.get_data_info
    assert VF6.get_ann_info is Stock.get_ann_info
    assert VF6.get_cat_ids is Stock.get_cat_ids
    assert vf6.version == DATASET_VERSION
    # Paths in the PKL are portable. Stock loader returns them as-is; the thin
    # subclass resolves only these paths and applies VF6 policy.
    raw_sample, sample = stock.get_data_info(0), vf6.get_data_info(0)
    assert not Path(raw_sample["lidar_path"]).is_absolute()
    assert Path(sample["lidar_path"]).is_file()
    assert all(Path(path).is_file() for path in sample["image_paths"])
    expected = to_repo_boxes(read_labels(vf6.data_infos[0]["label_path"])["gt_boxes"])
    np.testing.assert_allclose(sample["ann_info"]["gt_bboxes_3d"].tensor.numpy(), expected)
    np.testing.assert_array_equal(sample["ann_info"]["gt_labels_3d"], [0])
    # Prove no second z/yaw conversion at the Dataset boundary.
    assert sample["ann_info"]["gt_bboxes_3d"].box_dim == 7
    np.testing.assert_allclose(raw_sample["ann_info"]["gt_bboxes_3d"].tensor.numpy(), expected)
    for key in ("lidar2ego", "ego2global"):
        np.testing.assert_array_equal(sample[key], np.eye(4))
    for index, camera in enumerate(vf6.data_infos[0]["cams"].values()):
        offline = camera_matrices(camera)
        for key, value in offline.items():
            np.testing.assert_allclose(sample[key][index], value, atol=1e-5)
        np.testing.assert_allclose(sample["camera2lidar"][index] @ sample["lidar2camera"][index], np.eye(4), atol=1e-5)
    result = dict(boxes_3d=sample["ann_info"]["gt_bboxes_3d"], labels_3d=np.array([0]), scores_3d=np.array([1.]))
    assert vf6.evaluate([result])["vf6/mAP_3d@0.50"] == 1


def test_relocated_pkl_and_reordered_camera_subset(loader_classes, exported, tmp_path):
    _, VF6 = loader_classes
    moved = tmp_path / "company-machine"
    exported.rename(moved)
    dataset = build(VF6, moved, camera_names=list(reversed(CAMERAS)))
    assert list(dataset.data_infos[0]["cams"]) == list(reversed(CAMERAS))
    assert all(Path(p).is_file() for p in dataset.get_data_info(0)["image_paths"])
    subset = build(VF6, moved, camera_names=[CAMERAS[0]])
    assert len(subset.get_data_info(0)["camera2ego"]) == 1


def test_vf6_filter_masks_train_and_eval_consistently(loader_classes, exported):
    _, VF6 = loader_classes
    dataset = build(VF6, exported, min_lidar_points=4, test_mode=True)
    assert dataset.get_cat_ids(0) == []
    assert dataset.get_ann_info(0)["gt_bboxes_3d"].tensor.shape == (0, 7)
    with pytest.raises(ValueError, match="no usable samples"):
        build(VF6, exported, min_lidar_points=4)
    with pytest.raises(ValueError, match="no usable samples"):
        build(VF6, exported, point_cloud_range=[10, 10, -3, 20, 20, 5])


@pytest.mark.parametrize("changes", [dict(with_velocity=True), dict(use_valid_flag=False),
                                     dict(modality=dict(use_lidar=True, use_map=True))])
def test_unsupported_nuscenes_assumptions_fail_explicitly(loader_classes, exported, changes):
    _, VF6 = loader_classes
    with pytest.raises(ValueError):
        build(VF6, exported, **changes)


@pytest.mark.parametrize("keep_version", [True, False])
def test_old_pkl_is_rejected_instead_of_double_converting(loader_classes, exported, keep_version):
    _, VF6 = loader_classes
    path = exported / "vf6_infos_train.pkl"
    with path.open("rb") as stream:
        data = pickle.load(stream)
    data["metadata"].update(info_version=1, box_convention="center_lwh_ccw")
    if not keep_version:
        data["metadata"].pop("version")
    with path.open("wb") as stream:
        pickle.dump(data, stream)
    with pytest.raises(ValueError, match="Regenerate"):
        build(VF6, exported)


def test_smoke_guard_and_validation_on_empty_boxes(loader_classes, camera_dataset, tmp_path):
    _, VF6 = loader_classes
    for path in (camera_dataset / "label").rglob("*.txt"):
        path.write_text("")
    output = tmp_path / "smoke"
    create_vf6_infos(camera_dataset, output, smoke_test=True, camera_names=CAMERAS, camera_size=(32, 24))
    with pytest.raises(ValueError, match="Smoke"):
        build(VF6, output, test_mode=True)
    dataset = build(VF6, output, allow_smoke_test=True, test_mode=True)
    assert dataset.get_data_info(0)["ann_info"]["gt_bboxes_3d"].tensor.shape == (0, 7)


@pytest.mark.parametrize("mutate", [
    lambda frame: frame.pop("sensor2lidar_rotation", None),
])
def test_missing_camera_schema_field_is_rejected(exported, mutate):
    with (exported / "vf6_infos_train.pkl").open("rb") as stream:
        data = pickle.load(stream)
    mutate(data["infos"][0]["cams"][CAMERAS[0]])
    with pytest.raises(KeyError):
        validate_info(data["infos"][0], data["metadata"])


def test_timestamp_or_pose_mismatch_is_rejected(exported):
    with (exported / "vf6_infos_train.pkl").open("rb") as stream:
        data = pickle.load(stream)
    frame = copy.deepcopy(data["infos"][0])
    frame["timestamp"] /= 1e6
    with pytest.raises(ValueError, match="microsecond"):
        validate_info(frame, data["metadata"])
    frame = copy.deepcopy(data["infos"][0])
    frame["ego2global_translation"] = [1, 2, 3]
    with pytest.raises(ValueError, match="local-only"):
        validate_info(frame, data["metadata"])
