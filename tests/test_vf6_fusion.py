"""Camera synchronization, rectification and depth-transform regression tests."""

import ast
import json
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest

from vf6_adapter.cameras import CameraProcessor, camera_matrices
from vf6_adapter.converter import create_vf6_infos, discover_clips, timestamp_ns
from test_vf6_adapter import raw_dataset, make_splits, SESSION_A, SESSION_B, CLIP, STAMP

CAMERAS = ("CAM_P_F", "CAM_P_B")


def test_fusion_cli_with_real_train_val_manifests(camera_dataset, tmp_path):
    """Exercise the documented executable, not only the converter Python API."""
    import subprocess
    import sys

    train, val = make_splits(tmp_path)
    output = tmp_path / "cli-fusion"
    repository = Path(__file__).resolve().parents[1]
    command = [sys.executable, str(repository / "tools/create_vf6_data.py"),
               "--root-path", str(camera_dataset), "--out-dir", str(output),
               "--train-split", str(train), "--val-split", str(val),
               "--with-camera", "--cameras", *CAMERAS,
               "--camera-size", "32", "24", "--max-camera-dt-ms", "2"]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    for split, expected_session in [("train", SESSION_A), ("val", SESSION_B)]:
        with (output / ("vf6_infos_" + split + ".pkl")).open("rb") as stream:
            data = pickle.load(stream)
        assert data["metadata"]["smoke_test"] is False
        assert data["metadata"]["camera_names"] == list(CAMERAS)
        assert data["metadata"]["image_size"] == [32, 24]
        assert {info["session_id"] for info in data["infos"]} == {expected_session}
        checked = subprocess.run([sys.executable, str(repository / "tools/check_vf6.py"),
                                  "--data-root", str(output), "--split", split],
                                 cwd=tmp_path, capture_output=True, text=True)
        assert checked.returncode == 0, checked.stderr
        assert json.loads(checked.stdout)["offline"]["rectified_images"] == 2


def test_camera_cli_requires_explicit_fusion_mode(tmp_path):
    import subprocess
    import sys

    executable = Path(__file__).resolve().parents[1] / "tools/create_vf6_data.py"
    result = subprocess.run([sys.executable, str(executable), "--root-path", str(tmp_path),
                             "--out-dir", str(tmp_path / "output"), "--cameras", "CAM_P_F"],
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert "--cameras requires --with-camera" in result.stderr


@pytest.fixture
def camera_dataset(raw_dataset):
    cv2 = pytest.importorskip("cv2")
    path = raw_dataset / "VF6_01_Extrinsics_By_Dates.json"
    data = json.loads(path.read_text())
    transforms = {
        "CAM_P_F": [[0, -1, 0, 0], [0, 0, -1, 1.5], [1, 0, 0, -1], [0, 0, 0, 1]],
        "CAM_P_B": [[0, 1, 0, 0], [0, 0, -1, 1.5], [-1, 0, 0, 0], [0, 0, 0, 1]],
    }
    data["periods"][0]["extrinsics"].update(transforms)
    path.write_text(json.dumps(data))
    intrinsics = {name: dict(image_width=64, image_height=48,
                            camera_matrix=[[40, 0, 32], [0, 40, 24], [0, 0, 1]],
                            distortion_coefficients=[-.05, .01, 0, 0, 0]) for name in CAMERAS}
    (raw_dataset / "VF6_01_Intrinsics.json").write_text(json.dumps(intrinsics))
    y, x = np.mgrid[:48, :64]
    image = np.stack([x * 3, y * 4, np.full_like(x, 128)], axis=-1).astype(np.uint8)
    for session in [SESSION_A, SESSION_B]:
        for name in CAMERAS:
            directory = raw_dataset / "data" / session / CLIP / "CAMERA" / name
            directory.mkdir(parents=True)
            assert cv2.imwrite(str(directory / "1772252610-100740267.jpg"), image)
    return raw_dataset


def test_fusion_export_has_relative_paths_rectified_images_and_sync(camera_dataset, tmp_path):
    from PIL import Image
    from tools.check_vf6 import check_export

    train, val = make_splits(tmp_path)
    output = tmp_path / "fusion"
    report = create_vf6_infos(camera_dataset, output, train, val,
                              camera_names=CAMERAS, camera_size=(32, 24))
    assert report["converted_unique_images"] == 4
    assert report["splits"]["train"]["camera_sync"]["CAM_P_F"]["max_abs_dt_ms"] == 1
    with (output / "vf6_infos_train.pkl").open("rb") as stream:
        data = pickle.load(stream)
    assert data["metadata"]["camera_names"] == list(CAMERAS)
    assert data["metadata"]["image_mode"] == "rectified_pinhole"
    cameras = data["infos"][0]["cams"]
    for camera in cameras.values():
        assert not Path(camera["data_path"]).is_absolute()
        assert camera["time_offset_ns"] == 1_000_000
        with Image.open(output / camera["data_path"]) as image:
            assert image.size == (32, 24)
    # The processed dataset remains usable after moving it to another machine.
    moved = tmp_path / "copied-to-gpu-machine"
    output.rename(moved)
    result = check_export(moved, "train")
    assert result["rectified_images"] == 2
    assert result["perfect_prediction_mAP"] == 1


def test_camera_projection_matches_undistorted_pixel_coordinates(camera_dataset):
    import cv2

    clip = discover_clips(camera_dataset)[SESSION_A + "/" + CLIP]
    processor = CameraProcessor(camera_dataset, clip, CAMERAS, (32, 24), 50)
    camera = processor.match(timestamp_ns(STAMP))["CAM_P_F"]
    matrices = camera_matrices(camera)
    ego_point = np.array([10., 2., 1., 1.])
    camera_point = matrices["lidar2camera"] @ ego_point
    assert camera_point[2] > 0
    pixel = matrices["lidar2image"] @ ego_point
    pixel = pixel[:2] / pixel[2]
    distorted = cv2.projectPoints(camera_point[:3].reshape(1, 3), np.zeros(3), np.zeros(3),
                                 camera["original_intrinsics"], camera["original_distortion"])[0]
    undistorted = cv2.undistortPoints(distorted, camera["original_intrinsics"],
                                     camera["original_distortion"], P=camera["cam_intrinsic"])
    np.testing.assert_allclose(pixel, undistorted.reshape(2), atol=1e-4)
    np.testing.assert_allclose(matrices["camera2lidar"] @ matrices["lidar2camera"], np.eye(4), atol=1e-6)
    np.testing.assert_array_equal(matrices["camera2lidar"], matrices["camera2ego"])


def test_camera_sync_excess_is_rejected_before_writing(camera_dataset, tmp_path):
    output = tmp_path / "fusion"
    with pytest.raises(ValueError, match="offset.*exceeds"):
        create_vf6_infos(camera_dataset, output, smoke_test=True, camera_names=CAMERAS,
                         max_camera_dt_ms=.5, camera_size=(32, 24))
    assert not output.exists()


def test_nearest_camera_timestamp_tie_uses_earlier_frame(camera_dataset):
    import shutil

    directory = camera_dataset / "data" / SESSION_A / CLIP / "CAMERA" / CAMERAS[0]
    shutil.copyfile(next(directory.glob("*.jpg")), directory / "1772252610-098740267.jpg")
    clip = discover_clips(camera_dataset)[SESSION_A + "/" + CLIP]
    processor = CameraProcessor(camera_dataset, clip, CAMERAS, (32, 24), 50)
    camera = processor.match(timestamp_ns(STAMP))[CAMERAS[0]]
    assert camera["time_offset_ns"] == -1_000_000


@pytest.mark.parametrize("names,match", [
    (["CAM_F_F"], "pinhole cameras"),
    (["CAM_P_UNKNOWN"], "missing calibration"),
    (["CAM_P_F", "CAM_P_F"], "unique"),
])
def test_camera_selection_is_explicit(camera_dataset, tmp_path, names, match):
    with pytest.raises(ValueError, match=match):
        create_vf6_infos(camera_dataset, tmp_path / "fusion", smoke_test=True, camera_names=names)


def test_raw_image_size_must_match_calibration(camera_dataset, tmp_path):
    import cv2

    image = next((camera_dataset / "data" / SESSION_A / CLIP / "CAMERA" / CAMERAS[0]).glob("*.jpg"))
    cv2.imwrite(str(image), np.zeros((20, 20, 3), np.uint8))
    with pytest.raises(ValueError, match="size does not match"):
        create_vf6_infos(camera_dataset, tmp_path / "fusion", smoke_test=True,
                         camera_names=CAMERAS, camera_size=(32, 24))


def test_invalid_camera_intrinsics_are_rejected(camera_dataset, tmp_path):
    path = camera_dataset / "VF6_01_Intrinsics.json"
    data = json.loads(path.read_text())
    data[CAMERAS[0]]["camera_matrix"][0][0] = 0
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="valid pinhole"):
        create_vf6_infos(camera_dataset, tmp_path / "fusion", smoke_test=True, camera_names=CAMERAS)


def test_dataset_camera_fields_match_exported_ego_frame(camera_dataset, tmp_path):
    from types import SimpleNamespace

    output = tmp_path / "fusion"
    create_vf6_infos(camera_dataset, output, smoke_test=True, camera_names=CAMERAS, camera_size=(32, 24))
    with (output / "vf6_infos_train.pkl").open("rb") as stream:
        info = pickle.load(stream)["infos"][0]
    path = Path(__file__).resolve().parents[1] / "mmdet3d/datasets/vf6_dataset.py"
    cls = next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef))
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "get_data_info")
    namespace = dict(np=np, Path=Path, camera_matrices=camera_matrices)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    dataset = SimpleNamespace(data_infos=[info], dataset_root=str(output), test_mode=True,
                              modality=dict(use_camera=True), camera_names=list(CAMERAS))
    result = namespace["get_data_info"](dataset, 0)
    assert len(result["image_paths"]) == 2
    assert all(Path(path).is_file() for path in result["image_paths"])
    np.testing.assert_array_equal(result["lidar2ego"], np.eye(4))
    for index in range(2):
        np.testing.assert_allclose(result["camera2lidar"][index] @ result["lidar2camera"][index], np.eye(4), atol=1e-5)
        np.testing.assert_array_equal(result["lidar2image"][index],
                                      result["camera_intrinsics"][index] @ result["lidar2camera"][index])


def test_object_range_filter_supports_numpy_without_np_bool():
    torch = pytest.importorskip("torch")

    class Boxes:
        def __init__(self, xy):
            self.xy = xy
        def in_range_bev(self, limits):
            return torch.tensor((self.xy[:, 0] >= limits[0]) & (self.xy[:, 0] < limits[2]))
        def __getitem__(self, mask):
            return Boxes(self.xy[mask.numpy()])
        def limit_yaw(self, **kwargs):
            pass

    path = Path(__file__).resolve().parents[1] / "mmdet3d/datasets/pipelines/transforms_3d.py"
    cls = next(node for node in ast.parse(path.read_text()).body
               if isinstance(node, ast.ClassDef) and node.name == "ObjectRangeFilter")
    cls.decorator_list = []
    namespace = dict(np=np, LiDARInstance3DBoxes=Boxes, DepthInstance3DBoxes=Boxes, CameraInstance3DBoxes=type("CameraBoxes", (), {}))
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace)
    transform = namespace["ObjectRangeFilter"]([-10, -10, -3, 10, 10, 5])
    sample = dict(gt_bboxes_3d=Boxes(np.array([[1, 2], [100, 2]])), gt_labels_3d=np.array([0, 2]))
    assert transform(sample)["gt_labels_3d"].tolist() == [0]


def test_depth_lss_contract_gradients_and_no_point_mutation_without_cuda_pool():
    torch = pytest.importorskip("torch")
    # Execute the actual projection and depth CNN on CPU. Only the unavailable
    # CUDA BEV pool and framework decorators are substituted in this unit test.
    root = Path(__file__).resolve().parents[1]
    namespace = dict(torch=torch, nn=torch.nn, Tuple=Tuple,
                     force_fp32=lambda *args, **kwargs: (lambda function: function))
    base_path = root / "mmdet3d/models/vtransforms/base.py"
    nodes = [node for node in ast.parse(base_path.read_text()).body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef))
             and node.name in ("gen_dx_bx", "boolmask2idx", "BaseTransform", "BaseDepthTransform")]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(base_path), "exec"), namespace)
    lss_path = root / "mmdet3d/models/vtransforms/depth_lss.py"
    cls = next(node for node in ast.parse(lss_path.read_text()).body if isinstance(node, ast.ClassDef))
    cls.decorator_list = []
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(lss_path), "exec"), namespace)
    model = namespace["DepthLSSTransform"](
        in_channels=8, out_channels=4, image_size=(32, 32), feature_size=(4, 4),
        xbound=(-10, 10, 1), ybound=(-10, 10, 1), zbound=(-10, 10, 20), dbound=(1, 4, 1))
    assert model.height_expand is False
    assert model.add_depth_features is False
    captured = []
    model.dtransform.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach().clone()))
    model.bev_pool = lambda geometry, features: features.sum(dim=(1, 2)).permute(0, 3, 1, 2)
    images = torch.randn(1, 2, 8, 4, 4, requires_grad=True)
    points = [torch.tensor([[1., 0, 2, 20, 0], [1., 0, -2, 20, 0]])]
    before = points[0].clone()
    camera_matrix = torch.eye(4).reshape(1, 1, 4, 4).repeat(1, 2, 1, 1)
    lidar_matrix = torch.eye(4).reshape(1, 4, 4)
    augmentation = lidar_matrix.clone()
    augmentation[:, 0, 3] = .5
    output = model(images, points, None, camera_matrix, lidar_matrix, camera_matrix,
                   camera_matrix, camera_matrix, camera_matrix, camera_matrix, augmentation, [{}])
    assert tuple(output.shape) == (1, 4, 4, 4)
    assert tuple(captured[0].shape) == (2, 1, 32, 32)
    assert captured[0].min() >= 0  # Behind-camera point is excluded.
    assert captured[0].max() == 2
    assert torch.equal(points[0], before)
    output.sum().backward()
    assert images.grad is not None and torch.isfinite(images.grad).all()
    assert images.grad.abs().sum() > 0
