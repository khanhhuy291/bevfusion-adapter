#!/usr/bin/env python3
"""Check exported VF6 data offline, optionally run the real training pipeline."""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vf6_adapter.infos import validate_metadata, validate_info
from vf6_adapter.evaluation import evaluate_detection
from vf6_adapter.geometry import from_repo_boxes, to_repo_boxes
from vf6_adapter.cameras import camera_matrices


def check_export(root, split):
    root = Path(root).resolve()
    with (root / ("vf6_infos_" + split + ".pkl")).open("rb") as stream:
        data = pickle.load(stream)
    metadata, infos = data["metadata"], data["infos"]
    validate_metadata(metadata)
    if not infos:
        raise ValueError("Empty VF6 export")
    classes = metadata["classes"]
    if len({frame["token"] for frame in infos}) != len(infos):
        raise ValueError("Duplicate sample IDs")
    point_count, annotation_count = 0, 0
    camera_names = metadata.get("camera_names", [])
    image_paths = set()
    max_camera_dt_ms = 0.0
    # Validate all exported frames, then exercise evaluator on three samples.
    for info in infos:
        validate_info(info, metadata)
        points = np.load(root / info["lidar_path"], mmap_mode="r", allow_pickle=False)
        if points.ndim != 2 or points.shape[1] != 5 or points.dtype != np.float32:
            raise ValueError("Expected float32 points with shape (N,5): {}".format(info["lidar_path"]))
        if not len(points) or not np.isfinite(points).all() or np.any(points[:, 4] != 0):
            raise ValueError("Single-sweep points must be finite, nonempty, with zero sweep lag")
        boxes = info["gt_boxes"]
        if boxes.shape != (len(info["gt_names"]), 7) or not np.isfinite(boxes).all() or (boxes[:, 3:6] <= 0).any():
            raise ValueError("Invalid box array: {}".format(info["token"]))
        if not np.isin(info["gt_names"], classes).all():
            raise ValueError("Unexpected classes in {}".format(info["token"]))
        for name in ("label_attributes", "num_lidar_pts", "track_ids"):
            if np.asarray(info[name]).shape != (len(boxes),):
                raise ValueError("{} does not align with boxes".format(name))
        if not np.allclose(to_repo_boxes(from_repo_boxes(boxes)), boxes, atol=1e-5):
            raise ValueError("Box center/yaw conversion does not round trip")
        point_count += len(points)
        annotation_count += len(boxes)
        if camera_names:
            if metadata.get("image_mode") != "rectified_pinhole":
                raise ValueError("Fusion metadata must describe rectified pinhole images")
            if list(info.get("cams", {})) != camera_names:
                raise ValueError("Camera order/count differs between samples")
            from PIL import Image

            for name in camera_names:
                camera = info["cams"][name]
                matrices = camera_matrices(camera)
                np.testing.assert_allclose(matrices["lidar2camera"] @ matrices["camera2lidar"], np.eye(4), atol=1e-4)
                delta = abs(camera["timestamp_ns"] - info["timestamp_ns"]) / 1e6
                if delta > metadata["max_camera_dt_ms"]:
                    raise ValueError("Camera timestamp exceeds synchronization tolerance")
                max_camera_dt_ms = max(delta, max_camera_dt_ms)
                path = root / camera["data_path"]
                if str(path) not in image_paths:
                    with Image.open(path) as image:
                        if image.size != tuple(camera["image_size"]):
                            raise ValueError("Rectified image size mismatch: {}".format(path))
                        image.verify()
                    image_paths.add(str(path))
    gt, perfect = [], []
    for index in sorted({0, len(infos) // 2, len(infos) - 1}):
        frame = infos[index]
        mask = frame["num_lidar_pts"] >= 1
        boxes = from_repo_boxes(frame["gt_boxes"][mask])
        labels = np.asarray([classes.index(name) for name in frame["gt_names"][mask]], dtype=np.int64)
        gt.append(dict(boxes=boxes, labels=labels))
        perfect.append(dict(boxes=from_repo_boxes(to_repo_boxes(boxes)), labels=labels, scores=np.ones(len(boxes))))
    score = evaluate_detection(gt, perfect, classes)["vf6/mAP_3d@0.50"]
    if any(len(frame["boxes"]) for frame in gt) and not np.isclose(score, 1):
        raise ValueError("Perfect-prediction evaluation did not return mAP=1: {}".format(score))
    return dict(samples=len(infos), points=point_count, annotations=annotation_count,
                classes=classes, smoke_test=metadata["smoke_test"], perfect_prediction_mAP=score,
                camera_names=camera_names, rectified_images=len(image_paths), max_camera_dt_ms=max_camera_dt_ms)


def check_runtime(config_path, data_root, split, forward=False, steps=1, checkpoint_out=None):
    from mmcv import Config
    from mmcv.parallel import collate, scatter
    import torch
    from torchpack.utils.config import configs

    from mmdet3d.datasets import build_dataset
    from mmdet3d.utils import recursive_eval

    configs.load(config_path, recursive=True)
    configs.update(["--dataset_root", str(Path(data_root).resolve())])
    cfg = Config(recursive_eval(configs), filename=config_path)
    dataset = build_dataset(cfg.data[split])
    sample = dataset[0]
    batch = collate([sample], samples_per_gpu=1)
    summary = dict(dataset_samples=len(dataset), batch_keys=sorted(batch), pipeline="passed",
                   info_schema=dataset.metadata["schema"],
                   inherited_loader="NuScenesDataset.get_data_info/get_ann_info",
                   global_pose_status=dataset.metadata["global_pose_status"])
    if "img" in sample:
        summary["sample_image_shape"] = list(sample["img"].data.shape)
        for key in ("camera_intrinsics", "camera2ego", "camera2lidar", "lidar2camera", "lidar2image", "img_aug_matrix"):
            shape = tuple(sample[key].data.shape)
            if shape != (len(dataset.camera_names), 4, 4):
                raise ValueError("{}: expected (N,4,4), got {}".format(key, shape))
    if forward:
        if split != "train":
            raise ValueError("--forward requires --split train to check loss and backward")
        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA GPU and compiled BEVFusion operators are required for --forward")
        from mmdet3d.models import build_model
        from mmcv.runner import build_optimizer, save_checkpoint, load_checkpoint

        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats()
        summary["cuda_device"] = torch.cuda.get_device_name(0)
        model = build_model(cfg.model)
        model.init_weights()
        model = model.cuda().train()
        batch = scatter(batch, [0])[0]
        optimizer = build_optimizer(model, cfg.optimizer)
        original_points = [points.clone() for points in batch["points"]]
        summary["step_losses"] = []
        branches = ["encoders." + name for name in model.encoders]
        if model.fuser is not None:
            branches.append("fuser")
        for step in range(steps):
            optimizer.zero_grad()
            losses = model(**batch)
            loss = sum(value for key, value in losses.items() if key.startswith("loss/"))
            if not isinstance(loss, torch.Tensor) or not torch.isfinite(loss):
                raise ValueError("Training did not produce a finite loss")
            loss.backward()
            gradients = {prefix: 0.0 for prefix in branches}
            for name, parameter in model.named_parameters():
                if parameter.grad is None:
                    continue
                if not torch.isfinite(parameter.grad).all():
                    raise ValueError("Nonfinite gradient in {}".format(name))
                for prefix in branches:
                    if name.startswith(prefix + "."):
                        gradients[prefix] = max(gradients[prefix], float(parameter.grad.detach().abs().max().cpu()))
            if any(value == 0 for value in gradients.values()):
                raise ValueError("A sensor/fusion branch received no nonzero gradient: {}".format(gradients))
            for before, after in zip(original_points, batch["points"]):
                if not torch.equal(before, after):
                    raise ValueError("Forward modified input points in place")
            optimizer.step()
            value = float(loss.detach().cpu())
            summary["step_losses"].append(value)
            summary["branch_max_abs_gradient"] = gradients
            print("GPU train step {}/{}: loss={:.6f}".format(step + 1, steps, value), flush=True)
        torch.cuda.synchronize()
        if checkpoint_out:
            destination = Path(checkpoint_out).resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            save_checkpoint(model, str(destination), optimizer=optimizer, meta=dict(CLASSES=list(dataset.CLASSES)))
            load_checkpoint(model, str(destination), map_location="cpu", strict=True)
            summary["checkpoint_save_reload"] = str(destination)
        model.eval()
        with torch.no_grad():
            predictions = model(**batch)
        for key in ("scores_3d", "labels_3d"):
            if not torch.isfinite(predictions[0][key]).all():
                raise ValueError("Inference returned nonfinite {}".format(key))
        if not torch.isfinite(predictions[0]["boxes_3d"].tensor).all():
            raise ValueError("Inference returned nonfinite boxes")
        summary["predicted_boxes"] = len(predictions[0]["boxes_3d"])
        summary["peak_cuda_memory_mb"] = torch.cuda.max_memory_allocated() / (1024 ** 2)
        summary["forward_backward_optimizer_inference"] = "passed"
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--runtime", action="store_true", help="Build the actual Dataset, pipeline and collated batch")
    parser.add_argument("--forward", action="store_true", help="Also run CUDA loss/backward/inference; implies --runtime")
    parser.add_argument("--config", default="configs/vf6/lidar.yaml")
    parser.add_argument("--steps", type=int, default=1, help="Optimizer steps on one batch for --forward")
    parser.add_argument("--checkpoint-out", help="With --forward, save and strictly reload a checkpoint")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.checkpoint_out and not args.forward:
        parser.error("--checkpoint-out requires --forward")
    try:
        report = dict(offline=check_export(args.data_root, args.split))
        if args.runtime or args.forward:
            report["runtime"] = check_runtime(args.config, args.data_root, args.split, args.forward,
                                               args.steps, args.checkpoint_out)
    except (OSError, ValueError, KeyError, ImportError, RuntimeError, AssertionError) as error:
        parser.exit(2, "VF6 check failed: {}\n".format(error))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
