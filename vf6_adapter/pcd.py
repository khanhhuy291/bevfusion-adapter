"""Small, dependency-free PCD reader used by single-frame inference tools."""

from pathlib import Path

import numpy as np


_NUMPY_TYPES = {
    ("F", 4): "<f4",
    ("F", 8): "<f8",
    ("I", 1): "<i1",
    ("I", 2): "<i2",
    ("I", 4): "<i4",
    ("I", 8): "<i8",
    ("U", 1): "<u1",
    ("U", 2): "<u2",
    ("U", 4): "<u4",
    ("U", 8): "<u8",
}


def _single(header, key):
    values = header.get(key, [])
    if len(values) != 1:
        raise ValueError("PCD header must contain exactly one {} line".format(key))
    return values[0]


def read_binary_pcd(path):
    """Read scalar fields from an uncompressed binary PCD file.

    Returns ``(points, metadata)``. ``points`` is float32 with columns ordered
    exactly as ``metadata['fields']``. The deliberately small reader rejects
    ASCII, binary-compressed and vector-valued fields instead of guessing.
    """
    path = Path(path)
    header, comments = {}, []
    with path.open("rb") as stream:
        for _ in range(128):
            raw = stream.readline()
            if not raw:
                raise ValueError("{}: truncated PCD header".format(path))
            try:
                line = raw.decode("ascii").strip()
            except UnicodeDecodeError as error:
                raise ValueError("{}: PCD header is not ASCII".format(path)) from error
            if not line:
                continue
            if line.startswith("#"):
                comments.append(line[1:].strip())
                continue
            parts = line.split()
            header.setdefault(parts[0].upper(), []).append(parts[1:])
            if parts[0].upper() == "DATA":
                break
        else:
            raise ValueError("{}: PCD header is too long".format(path))

        data_mode = _single(header, "DATA")
        if data_mode != ["binary"]:
            raise ValueError("{}: only DATA binary PCD is supported".format(path))
        fields = _single(header, "FIELDS")
        sizes = [int(value) for value in _single(header, "SIZE")]
        types = [value.upper() for value in _single(header, "TYPE")]
        counts = [int(value) for value in _single(header, "COUNT")]
        if not fields or len(set(fields)) != len(fields):
            raise ValueError("{}: PCD fields must be nonempty and unique".format(path))
        if not (len(fields) == len(sizes) == len(types) == len(counts)):
            raise ValueError("{}: inconsistent FIELDS/SIZE/TYPE/COUNT lengths".format(path))
        if any(count != 1 for count in counts):
            raise ValueError("{}: vector-valued PCD fields are not supported".format(path))
        try:
            dtype = np.dtype([(name, _NUMPY_TYPES[(kind, size)])
                              for name, kind, size in zip(fields, types, sizes)])
        except KeyError as error:
            raise ValueError("{}: unsupported PCD TYPE/SIZE {}".format(path, error.args[0])) from error

        points = int(_single(header, "POINTS")[0])
        width = int(_single(header, "WIDTH")[0])
        height = int(_single(header, "HEIGHT")[0])
        if points <= 0 or width <= 0 or height <= 0 or width * height != points:
            raise ValueError("{}: invalid POINTS/WIDTH/HEIGHT".format(path))
        payload = stream.read()

    expected = points * dtype.itemsize
    if len(payload) != expected:
        raise ValueError("{}: expected {} payload bytes, found {}".format(path, expected, len(payload)))
    records = np.frombuffer(payload, dtype=dtype, count=points)
    array = np.column_stack([records[name] for name in fields]).astype(np.float32, copy=False)
    if not np.isfinite(array).all():
        raise ValueError("{}: point cloud contains NaN or Inf".format(path))

    timestamp = None
    for comment in comments:
        if comment.startswith("timestamp "):
            try:
                timestamp = float(comment.split(None, 1)[1])
            except ValueError as error:
                raise ValueError("{}: invalid timestamp comment".format(path)) from error
    return array, dict(fields=fields, points=points, width=width, height=height,
                       timestamp=timestamp, data="binary")


def prepare_model_points(points, fields, transform=None, intensity_scale=1.0,
                         point_cloud_range=None):
    """Convert named PCD columns to BEVFusion ``x,y,z,intensity,time`` input."""
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != len(fields):
        raise ValueError("points shape must match fields")
    indices = {name: index for index, name in enumerate(fields)}
    missing = [name for name in ("x", "y", "z", "intensity") if name not in indices]
    if missing:
        raise ValueError("PCD is missing required fields: {}".format(", ".join(missing)))
    if not np.isfinite(intensity_scale):
        raise ValueError("intensity_scale must be finite")

    xyz = points[:, [indices["x"], indices["y"], indices["z"]]].astype(np.float64)
    if transform is not None:
        transform = np.asarray(transform, dtype=np.float64)
        if (transform.shape != (4, 4) or not np.isfinite(transform).all()
                or not np.allclose(transform[3], [0, 0, 0, 1])):
            raise ValueError("transform must be a finite homogeneous 4x4 matrix")
        xyz = xyz @ transform[:3, :3].T + transform[:3, 3]
    intensity = points[:, indices["intensity"]].astype(np.float64) * intensity_scale
    model_points = np.column_stack([xyz, intensity, np.zeros(len(points))]).astype(np.float32)

    if point_cloud_range is not None:
        limits = np.asarray(point_cloud_range, dtype=np.float64)
        if limits.shape != (6,) or not np.isfinite(limits).all() or np.any(limits[:3] >= limits[3:]):
            raise ValueError("point_cloud_range must be [xmin,ymin,zmin,xmax,ymax,zmax]")
        mask = ((model_points[:, :3] >= limits[:3])
                & (model_points[:, :3] < limits[3:])).all(axis=1)
        model_points = model_points[mask]
    if not len(model_points):
        raise ValueError("No points remain after coordinate transform/range filtering")
    return np.ascontiguousarray(model_points)
