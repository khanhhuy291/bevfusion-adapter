"""Geometry in VF6's ego frame: center xyz, length/width/height, CCW yaw.

The converter normalizes raw boxes to bottom centers and CLOCKWISE yaw
for this checkout's NuScenesDataset/LiDARInstance3DBoxes. The evaluator uses
center/CCW geometry after converting both GT and predictions back.
Do not copy conversion rules from newer MMDetection3D versions.
"""

import numpy as np


def to_repo_boxes(boxes):
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 7).copy()
    boxes[:, 2] -= boxes[:, 5] / 2
    boxes[:, 6] = -boxes[:, 6]
    return boxes


def from_repo_boxes(boxes):
    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1] < 7:
        raise ValueError("Predicted boxes must have shape (N, >=7)")
    boxes = boxes[:, :7].copy()
    boxes[:, 2] += boxes[:, 5] / 2
    boxes[:, 6] = -boxes[:, 6]
    return boxes


def transform_points(xyz, transform):
    transform = np.asarray(transform, dtype=np.float64)
    return np.asarray(xyz) @ transform[:3, :3].T + transform[:3, 3]


def bev_corners(box):
    """Counterclockwise polygon for a center/lwh/CCW box."""
    x, y, _, length, width, _, yaw = box
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float)
    corners *= [length / 2, width / 2]
    c, s = np.cos(yaw), np.sin(yaw)
    return corners @ np.array([[c, s], [-s, c]]) + [x, y]


def _cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def _intersection_area(subject, clip):
    # Sutherland-Hodgman clipping of convex CCW polygons.
    polygon = list(subject)
    for start, end in zip(clip, np.roll(clip, -1, axis=0)):
        previous_polygon, polygon = polygon, []
        if not previous_polygon:
            return 0.0
        edge = end - start
        previous = previous_polygon[-1]
        previous_distance = _cross(edge, previous - start)
        for current in previous_polygon:
            distance = _cross(edge, current - start)
            inside, previous_inside = distance >= -1e-10, previous_distance >= -1e-10
            if inside != previous_inside:
                fraction = previous_distance / (previous_distance - distance)
                polygon.append(previous + fraction * (current - previous))
            if inside:
                polygon.append(current)
            previous, previous_distance = current, distance
    if len(polygon) < 3:
        return 0.0
    polygon = np.asarray(polygon)
    return float(abs(np.dot(polygon[:, 0], np.roll(polygon[:, 1], -1))
                     - np.dot(polygon[:, 1], np.roll(polygon[:, 0], -1))) / 2)


def iou_3d(box, others):
    """CPU rotated 3D IoU; all inputs use VF6 center/lwh/CCW convention."""
    others = np.asarray(others, dtype=float).reshape(-1, 7)
    result = np.zeros(len(others), dtype=float)
    if len(others) == 0:
        return result
    bottom = np.maximum(box[2] - box[5] / 2, others[:, 2] - others[:, 5] / 2)
    top = np.minimum(box[2] + box[5] / 2, others[:, 2] + others[:, 5] / 2)
    radius = np.linalg.norm(box[3:5]) / 2 + np.linalg.norm(others[:, 3:5], axis=1) / 2
    possible = (top > bottom) & (np.linalg.norm(others[:, :2] - box[:2], axis=1) <= radius)
    polygon = bev_corners(box)
    volume = np.prod(box[3:6])
    for index in np.flatnonzero(possible):
        area = _intersection_area(polygon, bev_corners(others[index]))
        intersection = area * (top[index] - bottom[index])
        union = volume + np.prod(others[index, 3:6]) - intersection
        result[index] = np.clip(intersection / max(union, 1e-12), 0, 1)
    return result


def in_bev_range(boxes, point_cloud_range):
    boxes = np.asarray(boxes).reshape(-1, 7)
    if point_cloud_range is None:
        return np.ones(len(boxes), dtype=bool)
    low, high = np.asarray(point_cloud_range[:2]), np.asarray(point_cloud_range[3:5])
    return ((boxes[:, :2] >= low) & (boxes[:, :2] < high)).all(axis=1)
