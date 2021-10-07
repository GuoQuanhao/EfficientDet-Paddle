import numpy as np
from numpy import ndarray


def nms(dets, scores, thresh):
    '''
    dets is a numpy array : num_dets, 4
    scores ia  nump array : num_dets,
    '''
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1] # get boxes with more ious first

    keep = []
    while order.size > 0:
        i = order[0] # pick maxmum iou box
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1) # maximum width
        h = np.maximum(0.0, yy2 - yy1 + 1) # maxiumum height
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def _batched_nms_vanilla(
    boxes: ndarray,
    scores: ndarray,
    idxs: ndarray,
    iou_threshold: float,
) -> ndarray:
    # Based on Detectron2 implementation, just manually call nms() on each class independently
    keep_mask = np.zeros_like(scores, dtype=np.bool)
    for class_id in np.unique(idxs):
        curr_indices = np.where(idxs == class_id)[0]
        curr_keep_indices = nms(boxes[curr_indices], scores[curr_indices], iou_threshold)
        keep_mask[curr_indices[curr_keep_indices]] = True
    keep_indices = np.where(keep_mask)[0]

    return keep_indices[np.argsort(-scores[keep_indices])]


def batched_nms(
    boxes: ndarray,
    scores: ndarray,
    idxs: ndarray,
    iou_threshold: float,
) -> ndarray:
    """
    Performs non-maximum suppression in a batched fashion.

    Each index value correspond to a category, and NMS
    will not be applied between elements of different categories.

    Args:
        boxes (ndarray[N, 4]): boxes where NMS will be performed. They
            are expected to be in ``(x1, y1, x2, y2)`` format with ``0 <= x1 < x2`` and
            ``0 <= y1 < y2``.
        scores (ndarray[N]): scores for each one of the boxes
        idxs (ndarray[N]): indices of the categories for each one of the boxes.
        iou_threshold (float): discards all overlapping boxes with IoU > iou_threshold

    Returns:
        ndarray: int64 tensor with the indices of the elements that have been kept by NMS, sorted
        in decreasing order of scores
    """
    # Benchmarks that drove the following thresholds are at
    # Ideally for GPU we'd use a higher threshold
    if len(boxes.flatten()) > 4_000:
        return _batched_nms_vanilla(boxes, scores, idxs, iou_threshold)
    else:
        return _batched_nms_coordinate_trick(boxes, scores, idxs, iou_threshold)


def _batched_nms_coordinate_trick(
    boxes: ndarray,
    scores: ndarray,
    idxs: ndarray,
    iou_threshold: float,
) -> ndarray:
    # strategy: in order to perform NMS independently per class,
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap
    if len(boxes.flatten()) == 0:
        return np.empty((0,), dtype=np.int64)
    max_coordinate = boxes.max()
    offsets = idxs * (max_coordinate + np.array([1]))
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep
