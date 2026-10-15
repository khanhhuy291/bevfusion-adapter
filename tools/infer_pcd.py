#!/usr/bin/env python3
"""Run the official BEVFusion LiDAR detector on binary PCD frames.

This is an inference-only adapter for unlabelled single-frame point clouds. It
does not pretend that arbitrary PCD data is a nuScenes dataset and it does not
compute evaluation metrics.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vf6_adapter.pcd import prepare_model_points, read_binary_pcd


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("--pcd-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--score-threshold", type=float, default=0.2)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--intensity-scale", type=float, default=1.0)
    parser.add_argument(
        "--transform", type=float, nargs=16, metavar="M",
        help="Optional row-major 4x4 transform from PCD coordinates to nuScenes LiDAR axes",
    )
    parser.add_argument("--no-bev", action="store_true", help="Do not save BEV PNG previews")
    args = parser.parse_args()
    if not 0 <= args.score_threshold <= 1:
        parser.error("--score-threshold must be in [0,1]")
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be positive")
    if args.stride <= 0:
        parser.error("--stride must be positive")
    return args


def load_config(path):
    from mmcv import Config
    from torchpack.utils.config import configs
    from mmdet3d.utils import recursive_eval

    configs.load(path, recursive=True)
    return Config(recursive_eval(configs), filename=path)


def build_detector(cfg, checkpoint_path, device):
    import torch
    from mmcv.runner import load_checkpoint, wrap_fp16_model
    from mmdet3d.models import build_model

    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("BEVFusion sparse voxel inference requires a CUDA device")
    index = int(device.split(":", 1)[1]) if ":" in device else 0
    torch.cuda.set_device(index)
    cfg.model.train_cfg = None
    if "pretrained" in cfg.model:
        cfg.model.pretrained = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    if cfg.get("fp16") is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, checkpoint_path, map_location="cpu", strict=True)
    return model.to(device).eval()


def prediction_record(source, metadata, input_count, points, boxes, scores, labels, classes,
                      elapsed_ms):
    detections = []
    for box, score, label in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
        detections.append(dict(class_id=int(label), class_name=classes[int(label)],
                               score=float(score), box=box))
    return dict(
        source=str(source), timestamp=metadata.get("timestamp"),
        input_points=int(input_count), model_range_points=int(len(points)),
        inference_ms=float(elapsed_ms), detections=detections,
    )


def main():
    args = parse_args()
    try:
        import torch
        from mmdet3d.core import LiDARInstance3DBoxes
        from mmdet3d.core.utils import visualize_lidar
    except ImportError as error:
        raise SystemExit("Missing BEVFusion runtime dependencies; use the repository Docker/legacy CUDA environment: {}".format(error))

    root = Path(args.pcd_dir).resolve()
    files = sorted(root.glob("*.pcd"))[::args.stride]
    if args.max_frames is not None:
        files = files[:args.max_frames]
    if not files:
        raise SystemExit("No .pcd files found under {}".format(root))
    output = Path(args.out_dir).resolve()
    predictions_dir, bev_dir = output / "predictions", output / "bev"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_bev:
        bev_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    expected_classes = ["car", "truck", "construction_vehicle", "bus", "trailer",
                        "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone"]
    classes = list(cfg.object_classes)
    if classes != expected_classes:
        raise SystemExit("This checkpoint adapter expects the official 10-class nuScenes order, got {}".format(classes))
    point_range = np.asarray(cfg.point_cloud_range, dtype=np.float64)
    transform = np.asarray(args.transform, dtype=np.float64).reshape(4, 4) if args.transform else None
    model = build_detector(cfg, args.checkpoint, args.device)

    summary = dict(config=str(Path(args.config).resolve()), checkpoint=str(Path(args.checkpoint).resolve()),
                   pcd_dir=str(root), classes=classes, point_cloud_range=point_range.tolist(),
                   score_threshold=args.score_threshold, single_frame=True,
                   intensity_scale=args.intensity_scale,
                   coordinate_transform=transform.tolist() if transform is not None else None,
                   frames=[])
    warned_zero_intensity = False
    for index, path in enumerate(files, 1):
        raw, metadata = read_binary_pcd(path)
        intensity_index = metadata["fields"].index("intensity")
        if not warned_zero_intensity and np.all(raw[:, intensity_index] == 0):
            print("WARNING: intensity is identically zero; inference can run but is out-of-domain.", flush=True)
            warned_zero_intensity = True
        points = prepare_model_points(raw, metadata["fields"], transform=transform,
                                      intensity_scale=args.intensity_scale,
                                      point_cloud_range=point_range)
        tensor = torch.from_numpy(points).to(args.device)
        metas = [dict(box_type_3d=LiDARInstance3DBoxes, token=path.stem,
                      timestamp=metadata.get("timestamp"))]
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            result = model(points=[tensor], metas=metas)[0]
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000
        scores_tensor = result["scores_3d"]
        labels_tensor = result["labels_3d"]
        keep = scores_tensor >= args.score_threshold
        boxes = result["boxes_3d"][keep]
        scores = scores_tensor[keep].numpy()
        labels = labels_tensor[keep].numpy()
        box_array = boxes.tensor.numpy()
        record = prediction_record(path, metadata, len(raw), points, box_array, scores,
                                   labels, classes, elapsed_ms)
        destination = predictions_dir / (path.stem + ".json")
        destination.write_text(json.dumps(record, indent=2) + "\n")
        if not args.no_bev:
            visualize_lidar(str(bev_dir / (path.stem + ".png")), points,
                            bboxes=boxes, labels=labels, classes=classes,
                            xlim=tuple(point_range[[0, 3]]), ylim=tuple(point_range[[1, 4]]),
                            radius=1, thickness=3)
        summary["frames"].append(dict(frame=path.name, input_points=len(raw),
                                      model_range_points=len(points), detections=len(scores),
                                      inference_ms=elapsed_ms))
        print("[{}/{}] {}: {} points, {} detections >= {:.2f}, {:.1f} ms".format(
            index, len(files), path.name, len(points), len(scores), args.score_threshold, elapsed_ms), flush=True)

    values = [frame["inference_ms"] for frame in summary["frames"]]
    summary["mean_inference_ms"] = float(np.mean(values))
    summary["median_inference_ms"] = float(np.median(values))
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("Saved predictions and previews to {}".format(output))


if __name__ == "__main__":
    main()
