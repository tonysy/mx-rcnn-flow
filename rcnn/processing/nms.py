import numpy as np
from ..cython.cpu_nms import cpu_nms
from ..cython.gpu_nms import gpu_nms


def py_nms_wrapper(thresh):
    def _nms(dets):
        return nms(dets, thresh)
    return _nms


def cpu_nms_wrapper(thresh):
    def _nms(dets):
        return cpu_nms(dets, thresh)
    return _nms


def gpu_nms_wrapper(thresh, device_id):
    def _nms(dets):
        return gpu_nms(dets, thresh, device_id)
    return _nms


def wnms_wrapper(thresh_lo, thresh_hi):
    def _nms(dets):
        return py_weighted_nms(dets, thresh_lo, thresh_hi)
    return _nms


def nms(dets, thresh):
    """
    greedily select boxes with high confidence and overlap with current maximum <= thresh
    rule out overlap >= thresh
    :param dets: [[x1, y1, x2, y2 score]]
    :param thresh: retain overlap < thresh
    :return: indexes to keep
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def py_weighted_nms(dets, thresh_lo, thresh_hi):
    """
    voting boxes with confidence > thresh_hi
    keep boxes overlap <= thresh_lo
    rule out overlap > thresh_hi
    :param dets: [[x1, y1, x2, y2 score]]
    :param thresh_lo: retain overlap <= thresh_lo
    :param thresh_hi: vote overlap > thresh_hi
    :return: indexes to keep
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        xx1 = np.maximum(x1[i], x1[order])
        yy1 = np.maximum(y1[i], y1[order])
        xx2 = np.minimum(x2[i], x2[order])
        yy2 = np.minimum(y2[i], y2[order])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order] - inter)

        inds = np.where(ovr <= thresh_lo)[0]
        inds_keep = np.where(ovr > thresh_hi)[0]
        if len(inds_keep) == 0:
            break

        order_keep = order[inds_keep]
        x1_avg = np.sum(scores[order_keep] * x1[order_keep]) / np.sum(scores[order_keep])
        y1_avg = np.sum(scores[order_keep] * y1[order_keep]) / np.sum(scores[order_keep])
        x2_avg = np.sum(scores[order_keep] * x2[order_keep]) / np.sum(scores[order_keep])
        y2_avg = np.sum(scores[order_keep] * y2[order_keep]) / np.sum(scores[order_keep])

        keep.append([x1_avg, y1_avg, x2_avg, y2_avg, scores[i]])
        order = order[inds]
    return np.array(keep)
