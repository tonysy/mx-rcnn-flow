import mxnet as mx
import numpy as np


def check_equal(lst, errstr='check_equal'):
    assert len(set(lst)) <= 1, '%s:%s' % (errstr, lst)


class RCNNIoULossOperator(mx.operator.CustomOp):
    def __init__(self, num_classes, grad_scale):
        super(RCNNIoULossOperator, self).__init__()
        self._num_classes = num_classes
        self._grad_scale = grad_scale

        self._data = None
        self._info = None

    def forward(self, is_train, req, in_data, out_data, aux):
        # data (batch_rois, num_classes * 4): pred_boxes - rois
        data = in_data[0].asnumpy()
        # bbox_target (batch_rois, num_classes * 4): gt_boxes
        label = in_data[1].asnumpy()
        # bbox_weight (batch_rois, num_classes * 4): mark to learn regression
        bbox_weight = in_data[2].asnumpy()
        # rois (batch_rois, 5)
        rois = in_data[3].asnumpy()
        rois = rois[:, 1:]

        IoU = np.zeros(data.shape)
        IoULoss = np.zeros(data.shape)
        info = np.zeros(data.shape)

        for i in range(self._num_classes):
            data[:, 4 * i:4 * (i + 1)] += rois

            # predicted boxes
            w = np.maximum(0, data[:, 4 * i + 2] - data[:, 4 * i + 0])
            h = np.maximum(0, data[:, 4 * i + 3] - data[:, 4 * i + 1])
            # print max(np.maximum(w, h))
            # np.clip(w, -10000, 10000, out=w)
            # np.clip(h, -10000, 10000, out=h)
            X = w * h

            # gt boxes
            w_gt = label[:, 4 * i + 2] - data[:, 4 * i + 0]
            h_gt = label[:, 4 * i + 3] - data[:, 4 * i + 1]
            X_gt = w_gt * h_gt

            # calculate iou between pred_boxes and gt_boxes
            I_w = np.maximum(0, np.minimum(data[:, 4 * i + 2], label[:, 4 * i + 2]) - np.maximum(data[:, 4 * i + 0], label[:, 4 * i + 0]))
            I_h = np.maximum(0, np.minimum(data[:, 4 * i + 3], label[:, 4 * i + 3]) - np.maximum(data[:, 4 * i + 1], label[:, 4 * i + 1]))
            I = I_w * I_h
            U = X + X_gt - I
            IoU[:, 4 * i] = I / (U + 1e-14)

            # store info
            info[:, 4 * i + 0] = I
            info[:, 4 * i + 1] = U
            info[:, 4 * i + 2] = I_w
            info[:, 4 * i + 3] = I_h

        # IoU loss is -ln(IoU)
        keep_inds = np.where((bbox_weight > 0) & (IoU > 0))
        IoULoss[keep_inds] = -np.log(IoU[keep_inds])

        self._data = data
        self._info = info
        self.assign(out_data[0], req[0], IoULoss)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        data = self._data
        info = self._info

        label = in_data[1].asnumpy()
        bbox_weight = in_data[2].asnumpy()
        g_data = np.zeros(in_data[0].shape)
        o_grad = out_grad[0].asnumpy()

        for i in range(data.shape[0]):
            for j in range(self._num_classes):
                if bbox_weight[i, 4 * j] > 0:
                    I = info[i, 4 * j + 0]
                    U = info[i, 4 * j + 1]
                    I_w = info[i, 4 * j + 2]
                    I_h = info[i, 4 * j + 3]

                    box = data[i, 4 * j:4 * (j + 1)]
                    gt_box = label[i, 4 * j:4 * (j + 1)]
                    dXdx = np.zeros(4)
                    dIdx = np.zeros(4)

                    if I > 0:
                        # X = (box[2] - box[1]) * (box[3] - box[1])
                        dXdx[0] = box[1] - box[3]
                        dXdx[1] = box[0] - box[2]
                        dXdx[2] = box[3] - box[1]
                        dXdx[3] = box[2] - box[0]
                        # I = (xx2 - xx1) * (yy2 - yy1)
                        # x1 > x1_gt:
                        if box[0] > gt_box[0]:
                            dIdx[0] = -I_h
                        # y1 > y1_gt
                        if box[1] > gt_box[1]:
                            dIdx[1] = -I_w
                        # x2 < x2_gt
                        if box[2] < gt_box[2]:
                            dIdx[2] = I_h
                        # y2 < y2_gt
                        if box[3] < gt_box[3]:
                            dIdx[3] = I_w
                        # grad = (1 / U) * dXdx - (U + I) / (U * I) * dIdx
                        grad = (1 / U) * dXdx - (U + I) / (U * I) * dIdx
                        g_data[i, 4 * j:4 * (j + 1)] = grad
                    else:
                        for k in range(0, 4):
                            g_data[i, 4 * j + k] = 0.03 * np.sign(box[k] - gt_box[k])
        # pos = np.where(g_data > 0)
        # print np.sum(np.abs(g_data[pos])) / np.prod(g_data[pos].shape)
        g_data *= o_grad
        g_data *= self._grad_scale
        self.assign(in_grad[0], req[0], g_data)


@mx.operator.register("rcnn_iou_loss")
class RCNNIoULossProp(mx.operator.CustomOpProp):
    def __init__(self, num_classes='21', grad_scale='1.0'):
        super(RCNNIoULossProp, self).__init__(need_top_grad=True)
        self._num_classes = int(num_classes)
        self._grad_scale = float(grad_scale)

    def list_arguments(self):
        return ['data', 'bbox_target', 'bbox_weight', 'rois']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        bbox_target_shape = in_shape[1]
        bbox_weight_shape = in_shape[2]
        rois_shape = in_shape[3]

        # share batch size
        batch_sizes = [data_shape[0], bbox_target_shape[0],
                       bbox_weight_shape[0], rois_shape[0]]
        check_equal(batch_sizes, 'inconsistent batch size')

        out_shape = [data_shape]
        return in_shape, out_shape

    def create_operator(self, ctx, shapes, dtypes):
        return RCNNIoULossOperator(self._num_classes, self._grad_scale)
