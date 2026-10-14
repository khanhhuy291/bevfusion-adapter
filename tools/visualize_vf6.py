#!/usr/bin/env python3
"""Project exported ego-frame points and GT boxes onto rectified VF6 images."""

import argparse
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vf6_adapter.cameras import camera_matrices


def box_corners(box):
    signs = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                      [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]])
    local = signs * box[3:6] / 2
    c, s = np.cos(box[6]), np.sin(box[6])
    return local @ np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]]) + box[:3]


def render(root, output, split="train", index=0):
    root = Path(root)
    with (root / ("vf6_infos_" + split + ".pkl")).open("rb") as stream:
        data = pickle.load(stream)
    frame = data["infos"][index]
    names = data["metadata"].get("camera_names", [])
    if not names:
        raise ValueError("This export has no camera records; run converter with --with-camera")
    points = np.load(root / frame["lidar_path"])[:, :3]
    points = points[::max(1, len(points) // 12000)]
    colors = {"Car": (40, 220, 40), "Rider": (0, 180, 255), "Pedestrian": (230, 50, 230)}
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    width = 640
    source_width, source_height = data["metadata"]["image_size"]
    height = round(width * source_height / source_width)
    columns = min(2, len(names))
    rows = (len(names) + columns - 1) // columns
    canvas = np.full((rows * (height + 34), columns * width, 3), 24, np.uint8)
    for number, name in enumerate(names):
        camera = frame["cams"][name]
        image = cv2.imread(str(root / camera["data_path"]))
        if image is None:
            raise OSError("Cannot read image: {}".format(camera["data_path"]))
        matrices = camera_matrices(camera)
        projection = matrices["lidar2image"]
        pixels = points @ projection[:3, :3].T + projection[:3, 3]
        distance = pixels[:, 2].copy()
        pixels[:, :2] /= np.maximum(distance[:, None], 1e-5)
        visible = ((distance > 0) & (pixels[:, 0] >= 0) & (pixels[:, 0] < source_width)
                   & (pixels[:, 1] >= 0) & (pixels[:, 1] < source_height))
        for pixel in pixels[visible, :2].astype(int):
            cv2.circle(image, tuple(pixel), 1, (255, 220, 0), -1)
        for box, label in zip(frame["gt_boxes"], frame["gt_names"]):
            corners = box_corners(box)
            coordinates = corners @ projection[:3, :3].T + projection[:3, 3]
            if (coordinates[:, 2] < .5).any():
                continue
            coordinates = coordinates[:, :2] / coordinates[:, 2:3]
            if not np.isfinite(coordinates).all() or (np.abs(coordinates) > 1e5).any():
                continue
            color = colors.get(label, (230, 230, 230))
            for a, b in edges:
                cv2.line(image, tuple(coordinates[a].round().astype(int)),
                         tuple(coordinates[b].round().astype(int)), color, 2)
        image = cv2.resize(image, (width, height))
        x, y = number % columns * width, number // columns * (height + 34)
        canvas[y + 34:y + 34 + height, x:x + width] = image
        text = "{}  dt={:+.2f}ms".format(name, camera["time_offset_ns"] / 1e6)
        cv2.putText(canvas, text, (x + 8, y + 23), cv2.FONT_HERSHEY_SIMPLEX, .55, (240, 240, 240), 1)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), canvas):
        raise OSError("Cannot write {}".format(output))
    print("Projection preview: {}".format(output.resolve()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    render(args.data_root, args.out, args.split, args.index)


if __name__ == "__main__":
    main()
