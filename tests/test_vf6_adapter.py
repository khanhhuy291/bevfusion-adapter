"""Geometry, split isolation and real LAZ-to-metadata integration checks."""

import importlib.util
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from vf6_adapter import CLASSES
from vf6_adapter.converter import (
    create_vf6_infos, discover_clips, load_lidar_calibration, read_labels,
    read_split, timestamp_ns,
)
from vf6_adapter.evaluation import evaluate_detection
from vf6_adapter.geometry import bev_corners, from_repo_boxes, iou_3d, to_repo_boxes

SESSION_A = "20260228_1105_VF6_01_1772251511_1772254880"
SESSION_B = "20260301_1105_VF6_01_1772337911_1772341280"
CLIP = "37_1772252610_1772252644"
STAMP = "1772252610-099740267"


@pytest.fixture
def raw_dataset(tmp_path):
    laspy = pytest.importorskip("laspy")
    pytest.importorskip("lazrs")
    root = tmp_path / "raw"
    root.mkdir()
    matrix = np.eye(4)
    matrix[:3, 3] = [1, 0, 2]
    (root / "VF6_01_Extrinsics_By_Dates.json").write_text(json.dumps(dict(periods=[dict(
        id="first", valid_from="2026-02-28", valid_to="2026-03-20",
        extrinsics={"LIDAR_TOP": matrix.tolist()},
    )])))
    for session in [SESSION_A, SESSION_B]:
        lidar_dir = root / "data" / session / CLIP / "LIDAR" / "LIDAR_TOP"
        labels = root / "label" / session / CLIP
        lidar_dir.mkdir(parents=True)
        labels.mkdir(parents=True)
        cloud = laspy.create(point_format=0, file_version="1.2")
        cloud.header.scales = [0.005, 0.005, 0.005]
        cloud.x = [0, 1, 2]
        cloud.y = [0, 0, 1]
        cloud.z = [0, 0.5, 1]
        cloud.intensity = [1, 128, 255]
        cloud.write(lidar_dir / (STAMP + ".laz"))
        labels.joinpath(STAMP + ".txt").write_text("1 0 2 4 2 2 0.6 Car 3 3 42\n")
    return root


def make_splits(tmp_path, train=SESSION_A, val=SESSION_B):
    a, b = tmp_path / "train.txt", tmp_path / "val.txt"
    a.write_text(train + "\n")
    b.write_text(val + "\n")
    return a, b


def test_timestamp_uses_integer_nanoseconds():
    assert timestamp_ns("1772252610-099740267") == 1772252610099740267
    assert timestamp_ns("1772252610-099740268") - timestamp_ns(STAMP) == 1


@pytest.mark.parametrize("stem", ["1772252610.099740267", "1772252610-99740267", "abc", "1772252610--099740267"])
def test_invalid_timestamp(stem):
    with pytest.raises(ValueError):
        timestamp_ns(stem)


def test_labels_preserve_unknown_attributes_and_object_ids(tmp_path):
    path = tmp_path / "labels.txt"
    path.write_text("1 2 3 4 2 1 0.6 Rider 3 0 42\n")
    data = read_labels(path)
    assert data["gt_boxes"].shape == (1, 7)
    assert data["gt_names"].tolist() == ["Rider"]
    assert data["label_attributes"].tolist() == [3]
    assert data["num_lidar_pts"].tolist() == [0]
    assert data["track_ids"].tolist() == [42]


@pytest.mark.parametrize("line", [
    "1 2 3 4 2 1 0 Car 1 2", "nan 2 3 4 2 1 0 Car 1 2 3",
    "1 2 3 -4 2 1 0 Car 1 2 3", "1 2 3 4 2 1 0 Unknown 1 2 3",
    "1 2 3 4 2 1 0 Car 1 -2 3", "1 2 3 4 2 1 0 Car 1 2.5 3",
])
def test_bad_labels_fail_with_location(tmp_path, line):
    path = tmp_path / "bad.txt"
    path.write_text(line)
    with pytest.raises(ValueError, match="bad.txt:1"):
        read_labels(path)


def test_empty_label_is_a_valid_annotated_negative(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("")
    assert read_labels(path)["gt_boxes"].shape == (0, 7)


def test_repo_yaw_matches_actual_checkout_rotation_helper():
    # Load the REAL upstream utility without importing MMDetection/CUDA modules.
    torch = pytest.importorskip("torch")
    path = Path(__file__).resolve().parents[1] / "mmdet3d/core/bbox/structures/utils.py"
    spec = importlib.util.spec_from_file_location("checkout_box_utils", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = np.array([[3, 4, 2, 4, 2, 2, 0.6]], dtype=np.float32)
    converted = to_repo_boxes(original)
    assert converted[0, 2] == 1  # Geometric center -> bottom center, once.
    unrotated = np.array([[[-2, -1, 0], [2, -1, 0], [2, 1, 0], [-2, 1, 0]]], dtype=np.float32)
    actual = module.rotation_3d_in_axis(torch.from_numpy(unrotated), torch.from_numpy(converted[:, 6]), axis=2)
    actual = actual.numpy()[0, :, :2] + converted[0, :2]
    np.testing.assert_allclose(actual, bev_corners(original[0]), atol=1e-6)
    np.testing.assert_allclose(from_repo_boxes(converted), original, atol=1e-6)


def test_rotated_3d_iou_has_analytic_values():
    box = np.array([0, 0, 0, 4, 2, 2, 0.0])
    others = np.array([box, box + [1, 0, 0, 0, 0, 0, 0],
                       box + [0, 0, 0, 0, 0, 0, np.pi / 2],
                       box + [0, 0, 3, 0, 0, 0, 0]])
    np.testing.assert_allclose(iou_3d(box, others), [1, 0.6, 1 / 3, 0], atol=1e-8)


def test_rotated_iou_is_symmetric_at_oblique_angles():
    a = np.array([1, 2, 3, 4, 2, 1, 0.2])
    b = np.array([2, 2, 3.2, 3, 2, 2, -0.7])
    assert 0 < iou_3d(a, [b])[0] < 1
    np.testing.assert_allclose(iou_3d(a, [b]), iou_3d(b, [a]))


def test_evaluator_perfect_empty_wrong_class_and_score_order():
    box = np.array([[0, 0, 0, 4, 2, 2, 0.6]])
    gt = [dict(boxes=box, labels=np.array([0]))]
    key = "vf6/mAP_3d@0.50"
    assert evaluate_detection(gt, [dict(boxes=box, labels=np.array([0]), scores=np.array([0.9]))], CLASSES)[key] == 1
    assert evaluate_detection(gt, [dict(boxes=np.empty((0, 7)), labels=np.array([], int), scores=np.array([]))], CLASSES)[key] == 0
    assert evaluate_detection(gt, [dict(boxes=box, labels=np.array([1]), scores=np.array([0.9]))], CLASSES)[key] == 0
    # A high-scoring false positive preceding a true positive halves AP.
    boxes = np.concatenate([box + [20, 0, 0, 0, 0, 0, 0], box])
    pred = [dict(boxes=boxes, labels=np.array([0, 0]), scores=np.array([0.9, 0.8]))]
    assert evaluate_detection(gt, pred, CLASSES)[key] == 0.5


def test_evaluator_does_not_match_one_gt_twice():
    boxes = np.array([[0, 0, 0, 2, 2, 2, 0], [10, 0, 0, 2, 2, 2, 0]], float)
    gt = [dict(boxes=boxes, labels=np.array([0, 0]))]
    pred = [dict(boxes=np.repeat(boxes[:1], 2, axis=0), labels=np.array([0, 0]), scores=np.array([1, .9]))]
    result = evaluate_detection(gt, pred, CLASSES)
    assert result["vf6/Car/recall_3d@0.50"] == .5
    assert result["vf6/mAP_3d@0.50"] == .5


def test_real_laz_conversion_and_explicit_split_isolation(raw_dataset, tmp_path):
    train, val = make_splits(tmp_path)
    output = tmp_path / "processed"
    report = create_vf6_infos(raw_dataset, output, train, val)
    assert report["converted_unique_frames"] == 2
    assert report["splits"]["train"]["samples"] == 1
    with (output / "vf6_infos_train.pkl").open("rb") as stream:
        data = pickle.load(stream)
    frame = data["infos"][0]
    points = np.load(output / frame["lidar_path"])
    np.testing.assert_allclose(points[:, :3], [[1, 0, 2], [2, 0, 2.5], [3, 1, 3]])
    np.testing.assert_array_equal(points[:, 3], [1, 128, 255])
    np.testing.assert_array_equal(points[:, 4], 0)
    assert data["metadata"]["smoke_test"] is False
    assert frame["session_id"] == SESSION_A
    assert frame["gt_boxes"][0, 2] == 2  # Metadata keeps center; Dataset converts.
    assert len(list(raw_dataset.rglob("*.npy"))) == 0
    with (output / "vf6_infos_val.pkl").open("rb") as stream:
        assert pickle.load(stream)["infos"][0]["session_id"] == SESSION_B


def test_flattened_clip_recovers_session_from_kml(raw_dataset, tmp_path):
    import shutil

    shutil.rmtree(raw_dataset / "data" / SESSION_B)
    shutil.rmtree(raw_dataset / "label" / SESSION_B)
    source = raw_dataset / "data" / SESSION_A / CLIP
    source.rename(raw_dataset / "data" / CLIP)
    (raw_dataset / "data" / CLIP / (SESSION_A + ".kml")).write_text("unused KML contents")
    clips = discover_clips(raw_dataset)
    assert list(clips) == [SESSION_A + "/" + CLIP]


def test_prefixed_and_windows_manifest_paths(raw_dataset, tmp_path):
    train, _ = make_splits(tmp_path, train="PNKX-26_09_2026\\data\\" + SESSION_A + " # selected session")
    assert read_split(train, discover_clips(raw_dataset)) == {SESSION_A + "/" + CLIP}


def test_ambiguous_clip_requires_session(raw_dataset, tmp_path):
    train, _ = make_splits(tmp_path, train=CLIP)
    with pytest.raises(ValueError, match="ambiguous"):
        read_split(train, discover_clips(raw_dataset))


def test_train_val_session_overlap_is_rejected_before_writing(raw_dataset, tmp_path):
    train, val = make_splits(tmp_path, val=SESSION_A + "/" + CLIP)
    output = tmp_path / "processed"
    with pytest.raises(ValueError, match="share recording sessions"):
        create_vf6_infos(raw_dataset, output, train, val)
    assert not output.exists()


def test_unknown_or_empty_split_does_not_become_validation(raw_dataset, tmp_path):
    train, val = make_splits(tmp_path, train="unknown_session")
    with pytest.raises(ValueError, match="matches no"):
        create_vf6_infos(raw_dataset, tmp_path / "processed", train, val)
    train.write_text("# nothing selected\n")
    with pytest.raises(ValueError, match="empty"):
        create_vf6_infos(raw_dataset, tmp_path / "processed", train, val)


def test_missing_lidar_is_not_an_empty_negative(raw_dataset, tmp_path):
    train, val = make_splits(tmp_path)
    next((raw_dataset / "data" / SESSION_A).rglob("*.laz")).unlink()
    with pytest.raises(FileNotFoundError, match="matching LIDAR_TOP"):
        create_vf6_infos(raw_dataset, tmp_path / "processed", train, val)


def test_date_interval_is_half_open_and_has_no_silent_fallback(raw_dataset):
    _, profile = load_lidar_calibration(raw_dataset, SESSION_A)
    assert profile["profile"] == "first"
    with pytest.raises(ValueError, match="calibration period"):
        load_lidar_calibration(raw_dataset, SESSION_A.replace("20260228", "20260320"))


def test_smoke_mode_is_explicit_and_marks_both_metadata_files(raw_dataset, tmp_path):
    output = tmp_path / "smoke"
    report = create_vf6_infos(raw_dataset, output, smoke_test=True)
    assert report["metadata"]["smoke_test"] is True
    tokens = []
    for split in ["train", "val"]:
        with (output / ("vf6_infos_" + split + ".pkl")).open("rb") as stream:
            data = pickle.load(stream)
        assert data["metadata"]["smoke_test"] is True
        tokens.append(data["infos"][0]["token"])
    assert tokens[0] == tokens[1]
    with pytest.raises(FileExistsError):
        create_vf6_infos(raw_dataset, output, smoke_test=True)


@pytest.mark.parametrize("smoke,fusion,debug", [
    (False, False, False), (True, False, False),
    (False, True, False), (True, True, False), (False, True, True),
])
def test_config_resolves_without_nuscenes_targets(smoke, fusion, debug):
    import copy
    yaml = pytest.importorskip("yaml")

    root = Path(__file__).resolve().parents[1]
    def merge(destination, source):
        for key, value in source.items():
            if isinstance(value, dict) and isinstance(destination.get(key), dict):
                merge(destination[key], value)
            else:
                destination[key] = copy.deepcopy(value)
    config = {}
    paths = ["configs/default.yaml", "configs/vf6/default.yaml"]
    if fusion:
        paths.append("configs/vf6/fusion/default.yaml")
    if smoke:
        paths.append("configs/vf6/fusion/smoke.yaml" if fusion else "configs/vf6/smoke.yaml")
    if debug:
        paths.append("configs/vf6/fusion/debug.yaml")
    for relative in paths:
        merge(config, yaml.safe_load((root / relative).read_text()))
    spec = importlib.util.spec_from_file_location("checkout_config_utils", root / "mmdet3d/utils/config.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = module.recursive_eval(config)
    if debug:
        assert config["max_epochs"] == 2
        assert config["fp16"] is None
        assert config["train_augmentation"] is False
    head = config["model"]["heads"]["object"]
    assert head["num_classes"] == 3
    assert "vel" not in head["common_heads"]
    assert head["bbox_coder"]["code_size"] == len(head["train_cfg"]["code_weights"]) == 8
    assert head["test_cfg"]["dataset"] == "VF6"
    assert head["test_cfg"]["nms_type"] is None
    for split in ["train", "val", "test"]:
        dataset = config["data"][split]
        assert dataset["allow_smoke_test"] is smoke
        expected = "data/vf6" + ("-fusion" if fusion else "") + ("-smoke" if smoke else "") + "/"
        assert dataset["ann_file"].startswith(expected)
        assert dataset["modality"]["use_camera"] is fusion
        assert not dataset["modality"]["use_radar"]
        types = {stage["type"] for stage in dataset["pipeline"]}
        assert not types.intersection({"GTDepth", "ObjectPaste", "LoadBEVSegmentation", "LoadRadarPointsMultiSweeps"})
        assert ("LoadMultiViewImageFromFiles" in types) is fusion
        if fusion:
            assert "img" in dataset["pipeline"][-1]["keys"]
    if fusion:
        camera = config["model"]["encoders"]["camera"]
        assert camera["vtransform"]["feature_size"] == [48, 60]
        assert config["model"]["fuser"]["in_channels"] == [80, 256]
        assert camera["backbone"]["init_cfg"] is None
        assert camera["backbone"]["out_indices"] == [1, 2, 3]
        # Both encoders must produce the SAME BEV spatial dimensions.
        transform = camera["vtransform"]
        for axis, key in enumerate(["xbound", "ybound"]):
            bound = transform[key]
            camera_cells = round((bound[1] - bound[0]) / bound[2]) // transform["downsample"]
            lidar_cells = head["train_cfg"]["grid_size"][axis] // head["train_cfg"]["out_size_factor"]
            assert camera_cells == lidar_cells == 128


def test_model_forward_accepts_lidar_without_camera_arguments():
    import ast
    from types import SimpleNamespace

    # Exercise the actual modified call boundary without loading CUDA operators.
    path = Path(__file__).resolve().parents[1] / "mmdet3d/models/fusion_models/bevfusion.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "BEVFusion")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "forward")
    method.decorator_list = []
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    model = SimpleNamespace(forward_single=lambda *args: args)
    points = [object()]
    result = namespace["forward"](model, points=points, metas=[{}])
    assert result[0] is None
    assert result[1] is points
    assert result[12] is None  # Optional depths.
