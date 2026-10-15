"""Export VF6 NuScenes-info v2 and LiDAR/camera payloads from raw and split manifests."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vf6_adapter import CLASSES
from vf6_adapter.cameras import DEFAULT_CAMERAS
from vf6_adapter.converter import create_vf6_infos


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-path", required=True, help="Raw root containing data/, label/ and extrinsics JSON")
    parser.add_argument("--out-dir", required=True, help="New processed-data directory outside the raw root")
    parser.add_argument("--train-split", help="Text file: one session or session/clip per line")
    parser.add_argument("--val-split", help="Text file: one session or session/clip per line")
    parser.add_argument("--smoke-test", action="store_true", help="Use the first clip for BOTH sets; debugging only")
    parser.add_argument("--max-samples", type=int, help="Limit labelled frames per split for debugging")
    parser.add_argument("--classes", nargs="+", default=list(CLASSES))
    parser.add_argument("--with-camera", action="store_true", help="Export rectified pinhole images for camera + LiDAR fusion")
    parser.add_argument("--cameras", nargs="+", help="Camera names in model order; defaults to all eight CAM_P_* cameras")
    parser.add_argument("--max-camera-dt-ms", type=float, default=50.0, help="Maximum nearest-image offset from the labelled LiDAR timestamp")
    parser.add_argument("--camera-size", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"), default=(960, 768), help="Rectified image output size, before the training pipeline")
    parser.add_argument("--session-map", help="JSON mapping flattened clip IDs to session IDs when KML/video names are absent")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite generated files in out-dir")
    args = parser.parse_args()
    if args.cameras and not args.with_camera:
        parser.error("--cameras requires --with-camera")
    session_map = json.loads(Path(args.session_map).read_text()) if args.session_map else None
    try:
        report = create_vf6_infos(
            args.root_path, args.out_dir, args.train_split, args.val_split,
            args.smoke_test, args.max_samples, args.classes, session_map, args.overwrite,
            camera_names=(args.cameras or DEFAULT_CAMERAS) if args.with_camera else None,
            max_camera_dt_ms=args.max_camera_dt_ms, camera_size=args.camera_size,
        )
    except (ValueError, OSError, KeyError, ImportError) as error:
        parser.exit(2, "VF6 conversion failed: {}\n".format(error))
    if args.smoke_test:
        print("SMOKE TEST: train and val intentionally contain the same clip. Metrics do not measure generalization.")
    print(json.dumps(report["splits"], indent=2))
    print("Report: {}".format(Path(args.out_dir) / "conversion_report.json"))


if __name__ == "__main__":
    main()
