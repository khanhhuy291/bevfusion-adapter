"""Export VF6 payloads and infos consumed by this checkout's NuScenesDataset."""

import json
import pickle
import re
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np

from . import CLASSES, INFO_VERSION, DATASET_VERSION
from .geometry import transform_points, to_repo_boxes

SESSION_PATTERN = re.compile(r"\d{8}_\d{4}_VF6_\d{2}_\d+_\d+")
CLIP_PATTERN = re.compile(r"\d+_\d+_\d+")
TIMESTAMP_PATTERN = re.compile(r"(\d{10,})-(\d{9})")


def timestamp_ns(stem):
    match = TIMESTAMP_PATTERN.fullmatch(stem)
    if match is None:
        raise ValueError("Expected seconds-nanoseconds timestamp, got {!r}".format(stem))
    return int(match[1]) * 1_000_000_000 + int(match[2])


def read_labels(path, classes=CLASSES):
    """Keep unknown attribute semantics explicit; parse, never remap, classes."""
    boxes, names, attributes, counts, track_ids = [], [], [], [], []
    for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        context = "{}:{}".format(path, line_number)
        if len(fields) != 11:
            raise ValueError("{}: expected 11 label columns, got {}".format(context, len(fields)))
        try:
            box = np.asarray(fields[:7], dtype=np.float64)
            attribute, count, track_id = map(int, fields[8:11])
        except ValueError as error:
            raise ValueError("{}: invalid numeric label".format(context)) from error
        if not np.isfinite(box).all() or (box[3:6] <= 0).any():
            raise ValueError("{}: box must be finite with positive dimensions".format(context))
        if fields[7] not in classes:
            raise ValueError("{}: unknown class {!r}; configure --classes explicitly".format(context, fields[7]))
        if count < 0 or track_id < 0:
            raise ValueError("{}: point count and object ID must be nonnegative".format(context))
        boxes.append(box)
        names.append(fields[7])
        attributes.append(attribute)
        counts.append(count)
        track_ids.append(track_id)
    return dict(
        gt_boxes=np.asarray(boxes, dtype=np.float32).reshape(-1, 7),
        gt_names=np.asarray(names, dtype=str),
        label_attributes=np.asarray(attributes, dtype=np.int64),
        num_lidar_pts=np.asarray(counts, dtype=np.int64),
        track_ids=np.asarray(track_ids, dtype=np.int64),
    )


def _session_for_clip(clip, session_map):
    if SESSION_PATTERN.fullmatch(clip.parent.name):
        return clip.parent.name
    candidates = set()
    for path in clip.iterdir():
        name = path.stem
        if name.startswith("summary_"):
            name = name[len("summary_"):]
        if path.suffix.lower() in (".kml", ".mp4") and SESSION_PATTERN.fullmatch(name):
            candidates.add(name)
    if clip.name in session_map:
        candidates.add(session_map[clip.name])
    if len(candidates) != 1:
        raise ValueError("{}: cannot determine a unique session; supply --session-map JSON".format(clip))
    session = candidates.pop()
    if not SESSION_PATTERN.fullmatch(session):
        raise ValueError("{}: invalid session ID {!r}".format(clip, session))
    return session


def discover_clips(root, session_map=None):
    """Support data/session/clip and flattened data/clip sample exports."""
    root = Path(root).resolve()
    clips = {}
    if not (root / "data").is_dir() or not (root / "label").is_dir():
        raise ValueError("root must contain data/ and label/: {}".format(root))
    for lidar_dir in sorted((root / "data").rglob("LIDAR_TOP")):
        if not lidar_dir.is_dir() or lidar_dir.parent.name != "LIDAR":
            continue
        clip = lidar_dir.parent.parent
        if not CLIP_PATTERN.fullmatch(clip.name):
            raise ValueError("Unexpected clip directory: {}".format(clip))
        session = _session_for_clip(clip, session_map or {})
        key = session + "/" + clip.name
        if key in clips:
            raise ValueError("Duplicate session/clip: {}".format(key))
        clips[key] = dict(session=session, clip=clip.name, path=clip, lidar_dir=lidar_dir)
    if not clips:
        raise ValueError("No data/**/LIDAR/LIDAR_TOP directories found")
    return clips


def read_split(path, clips):
    """Resolve IDs or prefixed paths such as PNKX-.../data/<session>."""
    selected = set()
    for line_number, raw in enumerate(Path(path).read_text().splitlines(), 1):
        entry = raw.split("#", 1)[0].strip().replace("\\", "/").rstrip("/")
        if not entry:
            continue
        parts = entry.split("/")
        last = parts[-1]
        if SESSION_PATTERN.fullmatch(last):
            matched = {key for key, value in clips.items() if value["session"] == last}
        elif CLIP_PATTERN.fullmatch(last):
            sessions = [part for part in parts[:-1] if SESSION_PATTERN.fullmatch(part)]
            matched = {key for key, value in clips.items() if value["clip"] == last
                       and (not sessions or value["session"] == sessions[-1])}
            if not sessions and len(matched) > 1:
                raise ValueError("{}:{}: ambiguous clip ID; use session/clip".format(path, line_number))
        else:
            matched = set()
        if not matched:
            raise ValueError("{}:{}: split entry matches no discovered clip: {}".format(path, line_number, entry))
        selected.update(matched)
    if not selected:
        raise ValueError("Split is empty: {}".format(path))
    return selected


def load_extrinsics(root, session):
    parts = session.split("_")
    vehicle = "_".join(parts[2:4])
    capture_date = date(int(parts[0][:4]), int(parts[0][4:6]), int(parts[0][6:8]))
    dated = Path(root) / (vehicle + "_Extrinsics_By_Dates.json")
    static = Path(root) / (vehicle + "_Extrinsics.json")
    if dated.is_file():
        data = json.loads(dated.read_text())
        periods = [period for period in data["periods"]
                   if date.fromisoformat(period["valid_from"]) <= capture_date
                   < date.fromisoformat(period["valid_to"])]
        if len(periods) != 1:
            raise ValueError("{}: expected one calibration period for {}, found {}".format(dated, capture_date, len(periods)))
        sensors = periods[0]["extrinsics"]
        profile = periods[0]["id"]
        source = dated
    elif static.is_file():
        sensors = json.loads(static.read_text())
        profile, source = "static", static
    else:
        raise FileNotFoundError("No extrinsics for {} under {}".format(vehicle, root))
    return sensors, dict(vehicle=vehicle, date=str(capture_date), profile=profile, source=str(source))


def validate_rigid_transform(value, context):
    transform = np.asarray(value, dtype=np.float64)
    if (transform.shape != (4, 4) or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0, 0, 0, 1])
            or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-3)
            or not np.isclose(np.linalg.det(transform[:3, :3]), 1, atol=1e-3)):
        raise ValueError("Invalid rigid calibration: {}".format(context))
    return transform


def load_lidar_calibration(root, session):
    sensors, profile = load_extrinsics(root, session)
    transform = validate_rigid_transform(sensors["LIDAR_TOP"], profile["source"] + ":LIDAR_TOP")
    return transform, profile


def read_lidar_points(path, lidar_to_ego):
    try:
        import laspy
    except ImportError as error:
        raise ImportError('Install preparation dependencies: pip install -r requirements-vf6.txt') from error
    cloud = laspy.read(path)
    # Single-sweep features: xyz in ego, original intensity, sweep time lag=0.
    # Ring and absolute point_timestamp are NOT the fifth model feature.
    points = np.empty((len(cloud.points), 5), dtype=np.float32)
    points[:, :3] = transform_points(cloud.xyz, lidar_to_ego)
    points[:, 3] = np.asarray(cloud.intensity, dtype=np.float32)
    points[:, 4] = 0.0
    if not len(points) or not np.isfinite(points).all():
        raise ValueError("Empty or nonfinite point cloud: {}".format(path))
    return points


def _label_directory(root, clip):
    candidates = [root / "label" / clip["session"] / clip["clip"], root / "label" / clip["clip"]]
    found = [path for path in candidates if path.is_dir()]
    if len(found) != 1:
        raise ValueError("{}: expected exactly one matching label directory, found {}".format(clip["path"], found))
    return found[0]


def create_vf6_infos(root_path, out_dir, train_split=None, val_split=None,
                     smoke_test=False, max_samples=None, classes=CLASSES,
                     session_map=None, overwrite=False, camera_names=None,
                     max_camera_dt_ms=50.0, camera_size=(960, 768)):
    root, output = Path(root_path).resolve(), Path(out_dir).resolve()
    if output == root or root in output.parents:
        raise ValueError("out_dir must be outside the raw dataset root")
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive")
    if not classes or len(set(classes)) != len(classes):
        raise ValueError("classes must be nonempty and unique")
    if camera_names is not None and (not camera_names or len(set(camera_names)) != len(camera_names)):
        raise ValueError("camera_names must be nonempty and unique when camera fusion is enabled")
    if not overwrite and output.exists() and any(output.iterdir()):
        raise FileExistsError("Output is not empty; use a new directory or --overwrite: {}".format(output))
    clips = discover_clips(root, session_map)
    if smoke_test:
        if train_split is not None or val_split is not None:
            raise ValueError("--smoke-test cannot be combined with split files")
        # Intentional same-data train/val, isolated behind explicit smoke mode.
        selected = {"train": {sorted(clips)[0]}, "val": {sorted(clips)[0]}}
    else:
        if train_split is None or val_split is None:
            raise ValueError("Provide both --train-split and --val-split, or explicitly use --smoke-test")
        selected = {"train": read_split(train_split, clips), "val": read_split(val_split, clips)}
        train_sessions = {clips[key]["session"] for key in selected["train"]}
        val_sessions = {clips[key]["session"] for key in selected["val"]}
        overlap = train_sessions & val_sessions
        if overlap:
            raise ValueError("Train/val share recording sessions (leakage): {}".format(sorted(overlap)))

    # Preflight every selected label/calibration before creating output files.
    prepared, clip_report, camera_processors = {}, {}, {}
    for key in sorted(selected["train"] | selected["val"]):
        clip = clips[key]
        transform, profile = load_lidar_calibration(root, clip["session"])
        if camera_names is not None:
            from .cameras import CameraProcessor

            camera_processors[key] = CameraProcessor(root, clip, camera_names, camera_size, max_camera_dt_ms)
        label_dir = _label_directory(root, clip)
        files = sorted(label_dir.glob("*.txt"), key=lambda path: timestamp_ns(path.stem))
        if not files:
            raise ValueError("No label files in {}".format(label_dir))
        frames, timestamps = [], set()
        for label in files:
            stamp = timestamp_ns(label.stem)
            if stamp in timestamps:
                raise ValueError("Duplicate label timestamp: {}".format(label))
            timestamps.add(stamp)
            lidar = clip["lidar_dir"] / (label.stem + ".laz")
            if not lidar.is_file():
                raise FileNotFoundError("Label has no matching LIDAR_TOP file: {}".format(label))
            frame = dict(
                token=key + "/" + label.stem, session_id=clip["session"], clip_id=clip["clip"],
                # nuScenes info timestamps are integer microseconds. Keep ns
                # separately for exact raw matching and synchronization audits.
                timestamp_ns=stamp, timestamp=stamp // 1000,
                sweeps=[],
                # The exported point file already uses the ego axes.
                lidar2ego_rotation=[1., 0., 0., 0.], lidar2ego_translation=[0., 0., 0.],
                # Structural placeholder ONLY: no measured global pose is
                # claimed. Single-frame detection never consumes ego2global.
                ego2global_rotation=[1., 0., 0., 0.], ego2global_translation=[0., 0., 0.],
                raw_lidar_path=str(lidar), label_path=str(label),
                raw_lidar2ego=transform.astype(np.float32), calibration=profile,
                **read_labels(label, classes),
            )
            # Normalize once at export. The inherited get_ann_info() consumes
            # bottom-center/lwh/CW boxes without another VF6 conversion.
            frame["gt_boxes"] = to_repo_boxes(frame["gt_boxes"])
            frame["valid_flag"] = frame["num_lidar_pts"] > 0
            if camera_names is not None:
                frame["cams"] = camera_processors[key].match(stamp)
            frames.append(frame)
        prepared[key] = frames
        clip_report[key] = dict(label_frames=len(frames), raw_lidar_frames=len(list(clip["lidar_dir"].glob("*.laz"))), calibration=profile)
    by_split = {}
    for split, keys in selected.items():
        frames = [frame for key in sorted(keys) for frame in prepared[key]]
        frames.sort(key=lambda frame: (frame["session_id"], frame["timestamp_ns"], frame["token"]))
        by_split[split] = frames[:max_samples] if max_samples is not None else frames

    output.mkdir(parents=True, exist_ok=True)
    converted = set()
    total_points = 0
    converted_images = set()
    active_processor = None
    for split in ("train", "val"):
        for frame in by_split[split]:
            relative = Path("points") / frame["session_id"] / frame["clip_id"] / (Path(frame["label_path"]).stem + ".npy")
            frame["lidar_path"] = relative.as_posix()
            if frame["token"] in converted:
                continue
            points = read_lidar_points(frame["raw_lidar_path"], frame["raw_lidar2ego"])
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.save(destination, points, allow_pickle=False)
            if camera_names is not None:
                processor = camera_processors[frame["session_id"] + "/" + frame["clip_id"]]
                if active_processor is not None and processor is not active_processor:
                    active_processor.clear_cache()
                active_processor = processor
                for name, camera in frame["cams"].items():
                    if camera["data_path"] not in converted_images:
                        processor.export(name, camera, output)
                        converted_images.add(camera["data_path"])
            total_points += len(points)
            converted.add(frame["token"])
            if len(converted) % 25 == 0:
                print("Converted {} unique labelled frames".format(len(converted)), flush=True)

    metadata = dict(
        dataset="VF6", version=DATASET_VERSION, info_version=INFO_VERSION,
        schema="bevfusion_nuscenes_infos", classes=list(classes), coordinate_frame="ego",
        box_convention="bottom_lwh_cw", timestamp_unit="microseconds",
        global_pose_status="unavailable_identity_placeholder",
        global_pose_usage="single_frame_detection_only; no global evaluation, maps or temporal alignment",
        point_features=["x", "y", "z", "intensity", "sweep_time_lag"],
        raw_root=str(root), smoke_test=bool(smoke_test), single_sweep=True,
        calibration_convention="LIDAR_TOP matrix interpreted as lidar_to_ego; verified on the supplied VF6_01 clip",
        label_convention="Columns 9 and 11 preserved; column 10 interpreted as annotated LiDAR point count",
        camera_names=list(camera_names or []),
        image_mode="rectified_pinhole" if camera_names is not None else None,
        image_size=list(camera_size) if camera_names is not None else None,
        max_camera_dt_ms=float(max_camera_dt_ms) if camera_names is not None else None,
    )
    report = dict(metadata=metadata, clips=clip_report, discovered_clips=len(clips),
                  ignored_clips=sorted(set(clips) - selected["train"] - selected["val"]),
                  converted_unique_frames=len(converted), converted_points=total_points,
                  converted_unique_images=len(converted_images),
                  max_samples_per_split=max_samples, splits={})
    split_dir = output / "splits"
    split_dir.mkdir(exist_ok=True)
    for split, frames in by_split.items():
        with (output / ("vf6_infos_" + split + ".pkl")).open("wb") as stream:
            pickle.dump(dict(infos=frames, metadata=dict(metadata, split=split)), stream, protocol=4)
        entries = sorted(selected[split])
        (split_dir / (split + ".txt")).write_text("\n".join(entries) + "\n")
        report["splits"][split] = dict(
            clips=entries, samples=len(frames), annotations=sum(len(frame["gt_boxes"]) for frame in frames),
            classes=dict(Counter(name for frame in frames for name in frame["gt_names"].tolist())),
            zero_point_annotations=sum(int((frame["num_lidar_pts"] == 0).sum()) for frame in frames),
        )
        if camera_names is not None:
            report["splits"][split]["camera_sync"] = {}
            for name in camera_names:
                deltas = [abs(frame["cams"][name]["time_offset_ns"]) / 1e6 for frame in frames]
                report["splits"][split]["camera_sync"][name] = dict(
                    samples=len(deltas), median_abs_dt_ms=float(np.median(deltas)),
                    max_abs_dt_ms=float(max(deltas)),
                )
    (output / "conversion_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
