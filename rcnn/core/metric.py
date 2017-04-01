import mxnet as mx
import numpy as np

from rcnn.config import config


def get_rpn_names():
    if config.TRAIN.RPN_OHEM:
        pred = ['rpn_cls_prob', 'rpn_bbox_loss', 'rpn_cls_mask', 'rpn_bbox_mask']
    else:
        pred = ['rpn_cls_prob', 'rpn_bbox_loss']
    label = ['rpn_label', 'rpn_bbox_target', 'rpn_bbox_weight']
    return pred, label


def get_rcnn_names():
    if config.TRAIN.RCNN_OHEM:
        pred = ['rcnn_cls_prob', 'rcnn_bbox_loss', 'rcnn_cls_mask', 'rcnn_bbox_mask']
    else:
        pred = ['rcnn_cls_prob', 'rcnn_bbox_loss']
    label = ['rcnn_label', 'rcnn_bbox_target', 'rcnn_bbox_weight']
    if config.TRAIN.END2END:
        pred.append('rcnn_label')
        rpn_pred, rpn_label = get_rpn_names()
        pred = rpn_pred + pred
        label = rpn_label
    return pred, label


class RPNAccMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RPNAccMetric, self).__init__('RPNAcc')
        self.rpn_ohem = config.TRAIN.RPN_OHEM
        self.pred, self.label = get_rpn_names()

    def update(self, labels, preds):
        pred = preds[self.pred.index('rpn_cls_prob')]
        label = labels[self.label.index('rpn_label')]

        # pred (b, c, p) or (b, c, h, w)
        pred_label = mx.ndarray.argmax_channel(pred).asnumpy().astype('int32')
        pred_label = pred_label.reshape((pred_label.shape[0], -1))
        # label (b, p)
        label = label.asnumpy().astype('int32')

        # filter with keep_inds
        if self.rpn_ohem:
            cls_mask = preds[self.pred.index('rpn_cls_mask')].asnumpy()[:, 0, :].reshape(label.shape)
            keep_inds = np.where(cls_mask > 0)
        else:
            keep_inds = np.where(label != -1)
        pred_label = pred_label[keep_inds]
        label = label[keep_inds]

        self.sum_metric += np.sum(pred_label.flat == label.flat)
        self.num_inst += len(pred_label.flat)


class RCNNAccMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RCNNAccMetric, self).__init__('RCNNAcc')
        self.e2e = config.TRAIN.END2END
        self.rcnn_ohem = config.TRAIN.RCNN_OHEM
        self.pred, self.label = get_rcnn_names()

    def update(self, labels, preds):
        pred = preds[self.pred.index('rcnn_cls_prob')]
        if self.e2e:
            label = preds[self.pred.index('rcnn_label')]
        else:
            label = labels[self.label.index('rcnn_label')]

        last_dim = pred.shape[-1]
        pred_label = pred.asnumpy().reshape(-1, last_dim).argmax(axis=1).astype('int32')
        label = label.asnumpy().reshape(-1,).astype('int32')

        if self.rcnn_ohem:
            cls_mask = preds[self.pred.index('rcnn_cls_mask')].asnumpy().reshape(-1, last_dim)
            keep_inds = np.where(cls_mask.sum(axis=1) > 0)[0]
            pred_label = pred_label[keep_inds]
            label = label[keep_inds]

        self.sum_metric += np.sum(pred_label.flat == label.flat)
        self.num_inst += len(pred_label.flat)


class RPNLogLossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RPNLogLossMetric, self).__init__('RPNLogLoss')
        self.rpn_ohem = config.TRAIN.RPN_OHEM
        self.pred, self.label = get_rpn_names()

    def update(self, labels, preds):
        pred = preds[self.pred.index('rpn_cls_prob')]
        label = labels[self.label.index('rpn_label')]

        # label (b, p)
        label = label.asnumpy().astype('int32').reshape((-1))
        # pred (b, c, p) or (b, c, h, w) --> (b, p, c) --> (b*p, c)
        pred = pred.asnumpy().reshape((pred.shape[0], pred.shape[1], -1)).transpose((0, 2, 1))
        pred = pred.reshape((label.shape[0], -1))

        # filter with keep_inds
        if self.rpn_ohem:
            # induce the keep_inds from argmax_channel to cls_prob
            cls_mask = preds[self.pred.index('rpn_cls_mask')].asnumpy()
            cls_mask = cls_mask[:, 0, :].reshape((-1))
            keep_inds = np.where(cls_mask > 0)[0]
        else:
            keep_inds = np.where(label != -1)[0]
        label = label[keep_inds]
        cls = pred[keep_inds, label]

        cls += 1e-14
        cls_loss = -1 * np.log(cls)
        cls_loss = np.sum(cls_loss)
        self.sum_metric += cls_loss
        self.num_inst += label.shape[0]


class RCNNLogLossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RCNNLogLossMetric, self).__init__('RCNNLogLoss')
        self.e2e = config.TRAIN.END2END
        self.rcnn_ohem = config.TRAIN.RCNN_OHEM
        self.pred, self.label = get_rcnn_names()

    def update(self, labels, preds):
        pred = preds[self.pred.index('rcnn_cls_prob')]
        if self.e2e:
            label = preds[self.pred.index('rcnn_label')]
        else:
            label = labels[self.label.index('rcnn_label')]

        last_dim = pred.shape[-1]
        pred = pred.asnumpy().reshape(-1, last_dim)
        label = label.asnumpy().reshape(-1,).astype('int32')

        if self.rcnn_ohem:
            cls_mask = preds[self.pred.index('rcnn_cls_mask')].asnumpy().reshape(-1, last_dim)
            keep_inds = np.where(cls_mask.sum(axis=1) > 0)[0]
            pred = pred[keep_inds, :]
            label = label[keep_inds]

        cls = pred[np.arange(label.shape[0]), label]

        cls += 1e-14
        cls_loss = -1 * np.log(cls)
        cls_loss = np.sum(cls_loss)
        self.sum_metric += cls_loss
        self.num_inst += label.shape[0]


class RPNRegLossMetric(mx.metric.EvalMetric):
    def __init__(self):
        name = 'RPNIoULoss' if config.RPN_IOU_LOSS else 'RPNL1Loss'
        super(RPNRegLossMetric, self).__init__(name)
        self.rpn_ohem = config.TRAIN.RPN_OHEM
        self.pred, self.label = get_rpn_names()

    def update(self, labels, preds):
        bbox_loss = preds[self.pred.index('rpn_bbox_loss')].asnumpy()
        bbox_weight = labels[self.label.index('rpn_bbox_weight')].asnumpy()

        # calculate num_inst (average on those fg anchors)
        if self.rpn_ohem:
            bbox_mask = preds[self.pred.index('rpn_bbox_mask')].asnumpy()
            bbox_loss *= bbox_mask
            num_inst = np.sum((bbox_mask > 0) & (bbox_weight > 0)) / 4
        else:
            num_inst = np.sum(bbox_weight > 0) / 4

        self.sum_metric += np.sum(bbox_loss)
        self.num_inst += num_inst


class RCNNRegLossMetric(mx.metric.EvalMetric):
    def __init__(self):
        name = 'RCNNIoULoss' if config.RCNN_IOU_LOSS else 'RCNNL1Loss'
        super(RCNNRegLossMetric, self).__init__(name)
        self.e2e = config.TRAIN.END2END
        self.rcnn_ohem = config.TRAIN.RCNN_OHEM
        self.pred, self.label = get_rcnn_names()

    def update(self, labels, preds):
        bbox_loss = preds[self.pred.index('rcnn_bbox_loss')].asnumpy()
        if self.e2e:
            label = preds[self.pred.index('rcnn_label')].asnumpy()
        else:
            label = labels[self.label.index('rcnn_label')].asnumpy()

        last_dim = bbox_loss.shape[-1]
        bbox_loss = bbox_loss.reshape((-1, last_dim))
        label = label.reshape(-1)

        # calculate num_inst
        if self.rcnn_ohem:
            bbox_mask = preds[self.pred.index('rcnn_bbox_mask')].asnumpy().reshape((-1, last_dim))
            bbox_loss *= bbox_mask
            keep_inds = np.where((bbox_mask.sum(axis=1) > 0) & (label != 0))[0]
        else:
            keep_inds = np.where(label != 0)[0]
        num_inst = len(keep_inds)

        self.sum_metric += np.sum(bbox_loss)
        self.num_inst += num_inst
