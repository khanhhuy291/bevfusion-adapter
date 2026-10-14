"""VF6 diagnostic detection AP, independent of nuScenes and GPU operators."""

import numpy as np

from .geometry import iou_3d


def evaluate_detection(ground_truth, predictions, classes, iou_threshold=0.5):
    """Score-greedy, one-to-one matching, continuous precision-envelope AP.

    Each frame has `boxes` (VF6 center/lwh/CCW) and integer `labels`.
    Predictions additionally have `scores`. mAP excludes classes without GT.
    This is a custom 3D IoU metric, NOT nuScenes mAP or NDS.
    """
    if len(ground_truth) != len(predictions):
        raise ValueError("Prediction count must equal the number of frames")
    if not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold must be in (0, 1]")
    metrics, average_precisions = {}, []
    for category, name in enumerate(classes):
        gt = [np.asarray(frame["boxes"], dtype=float).reshape(-1, 7)[
            np.asarray(frame["labels"]) == category] for frame in ground_truth]
        matches = [np.zeros(len(boxes), dtype=bool) for boxes in gt]
        total = sum(len(boxes) for boxes in gt)
        ranked = []
        for frame_index, frame in enumerate(predictions):
            boxes = np.asarray(frame["boxes"], dtype=float).reshape(-1, 7)
            scores = np.asarray(frame["scores"], dtype=float)
            labels = np.asarray(frame["labels"])
            if scores.shape != (len(boxes),) or labels.shape != (len(boxes),):
                raise ValueError("Prediction boxes, scores and labels must have equal lengths")
            if not np.isfinite(boxes).all() or not np.isfinite(scores).all() or (boxes[:, 3:6] <= 0).any():
                raise ValueError("Predictions must be finite with positive box dimensions")
            if ((labels < 0) | (labels >= len(classes))).any():
                raise ValueError("Prediction contains an unknown class ID")
            for index in np.flatnonzero(labels == category):
                ranked.append((float(scores[index]), frame_index, boxes[index]))
        ranked.sort(key=lambda value: -value[0])
        true_positives = np.zeros(len(ranked), dtype=float)
        for index, (_, frame_index, box) in enumerate(ranked):
            overlaps = iou_3d(box, gt[frame_index])
            overlaps[matches[frame_index]] = -1
            if len(overlaps):
                best = int(overlaps.argmax())
                if overlaps[best] >= iou_threshold:
                    matches[frame_index][best] = True
                    true_positives[index] = 1
        accumulated = true_positives.cumsum()
        precision = accumulated / np.arange(1, len(ranked) + 1)
        recall = accumulated / max(total, 1)
        padded_recall = np.r_[0.0, recall, 1.0]
        envelope = np.r_[0.0, precision, 0.0]
        envelope = np.maximum.accumulate(envelope[::-1])[::-1]
        changes = np.flatnonzero(padded_recall[1:] != padded_recall[:-1])
        ap = float(np.sum((padded_recall[changes + 1] - padded_recall[changes]) * envelope[changes + 1]))
        prefix = "vf6/{}/".format(name)
        suffix = "3d@{:.2f}".format(iou_threshold)
        metrics[prefix + "AP_" + suffix] = ap
        metrics[prefix + "recall_" + suffix] = float(recall[-1]) if len(recall) else 0.0
        metrics[prefix + "gt_count"] = total
        if total:
            average_precisions.append(ap)
    metrics["vf6/mAP_3d@{:.2f}".format(iou_threshold)] = float(np.mean(average_precisions)) if average_precisions else 0.0
    return metrics
