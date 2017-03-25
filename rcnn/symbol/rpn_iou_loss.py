import mxnet as mx
import numpy as np
from ..processing.generate_anchor import generate_anchors


def check_equal(lst, errstr='check_equal'):
    assert len(set(lst)) <= 1, '%s:%s' % (errstr, lst)


class RPNIoULossOperator(mx.operator.CustomOp):
    def __init__(self, feature_stride, scales, ratios, grad_scale):
        super(RPNIoULossOperator, self).__init__()
        self._feat_stride = feature_stride
        self._anchors = generate_anchors(base_size=feature_stride, scales=scales, ratios=ratios)
        self._grad_scale = grad_scale

        self._data = None
        self._label = None
        self._info = None

    def forward(self, is_train, req, in_data, out_data, aux):
        # data (batch_image, num_anchor * 4, height, width): pred_boxes - rois
        data = in_data[0].asnumpy()
        # bbox_target (batch_image, num_anchor * 4, height, width): gt_boxes
        label = in_data[1].asnumpy()
        # bbox_weight (batch_image, num_anchor * 4, height, width): mark to learn regression
        bbox_weight = in_data[2].asnumpy()

        # calculate anchors
        batch_size, num_anchor, height, width = data.shape
        num_anchor /= 4
        check_equal([num_anchor, len(self._anchors)], 'inconsistent anchor shape')

        # get anchors
        # enumerate all shifts
        shift_x = np.arange(0, width) * self._feat_stride
        shift_y = np.arange(0, height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()
        check_equal([shifts.shape[0], height * width], 'inconsistent shift')

        # add A anchors (1, A, 4) to K shifts (K, 1, 4) to (K*A, 4) anchors
        shifts = shifts.reshape((1, shifts.shape[0], 4)).transpose((1, 0, 2))
        anchors = self._anchors.reshape((1, num_anchor, 4)) + shifts
        anchors = anchors.reshape((height * width * num_anchor, 4))
        # reshape anchors to (b*h*w*a, 4)
        if batch_size > 1:
            anchors = np.tile(anchors, (batch_size, 1))
        anchors = anchors.reshape((-1, 4))

        # reshape data, gt_boxes from (b, a*4, h, w) into (b*h*w*a, 4)
        data = data.transpose((0, 2, 3, 1)).reshape((-1, 4)) + anchors
        label = label.transpose((0, 2, 3, 1)).reshape((-1, 4))
        bbox_weight = bbox_weight.transpose((0, 2, 3, 1)).reshape((-1, 4)).sum(axis=1)

        # calculate iou for all inside_inds
        inside_inds = np.where(bbox_weight > 0)[0]
        data = data[inside_inds, :]
        label = label[inside_inds, :]
        # print np.sum(np.abs(data - label)) / np.prod(data.shape)
        # print np.max(data)

        # predicted boxes
        w = np.maximum(0, data[:, 2] - data[:, 0])
        h = np.maximum(0, data[:, 3] - data[:, 1])
        X = w * h

        # gt_boxes
        w_gt = label[:, 2] - label[:, 0]
        h_gt = label[:, 3] - label[:, 1]
        X_gt = w_gt * h_gt

        # calculate iou between pred_boxes and gt_boxes
        I_w = np.maximum(0, np.minimum(data[:, 2], label[:, 2]) - np.maximum(data[:, 0], label[:, 0]))
        I_h = np.maximum(0, np.minimum(data[:, 3], label[:, 3]) - np.maximum(data[:, 1], label[:, 1]))
        I = I_w * I_h
        U = X + X_gt - I
        IoU = I / (U + 1e-14)

        # fill to the original format
        IoULoss = np.zeros(in_data[0].shape)
        b, h, w, a = np.unravel_index(inside_inds, (batch_size, height, width, num_anchor))
        IoULoss[b, a * 4, h, w] = IoU

        # IoU loss is -ln(IoU)
        keep_inds = np.where(IoULoss > 0)
        IoULoss[keep_inds] = -np.log(IoULoss[keep_inds])

        self._data = data
        self._label = label
        self._info = (inside_inds, I, U, I_w, I_h)
        self.assign(out_data[0], req[0], IoULoss)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        # parse shape
        batch_size, num_anchor, height, width = in_data[0].shape
        num_anchor /= 4

        data = self._data
        label = self._label
        inside_inds, inside_I, inside_U, inside_I_w, inside_I_h = self._info

        inside_grad = np.zeros(data.shape)
        for k in range(data.shape[0]):
            I = inside_I[k]
            U = inside_U[k]
            I_w = inside_I_w[k]
            I_h = inside_I_h[k]

            box = data[k, :]
            gt_box = label[k, :]
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
                inside_grad[k, :] = grad
            else:
                # print 'missing iou'
                for l in range(0, 4):
                    inside_grad[k, l] = 0.03 * np.sign(box[l] - gt_box[l])

        # print inside_grad.shape[0]
        # print np.sum(np.abs(inside_grad)) / np.prod(inside_grad.shape)
        # fill to the original format
        g_data = np.zeros(in_data[0].shape)
        for i, index in enumerate(inside_inds):
            b, h, w, a = np.unravel_index(index, (batch_size, height, width, num_anchor))
            g_data[b, a * 4:(a + 1) * 4, h, w] = inside_grad[i, :]

        o_grad = out_grad[0].asnumpy()
        g_data *= o_grad
        g_data *= self._grad_scale
        # print np.sum(np.abs(g_data))
        self.assign(in_grad[0], req[0], g_data)


@mx.operator.register("rpn_iou_loss")
class RPNIoULossProp(mx.operator.CustomOpProp):
    def __init__(self, feature_stride='16', scales='(8, 16, 32)', ratios='(0.5, 1, 2)', grad_scale='1.0'):
        super(RPNIoULossProp, self).__init__(need_top_grad=True)
        self._feature_stride = int(feature_stride)
        self._scales = np.fromstring(scales[1:-1], dtype=float, sep=',')
        self._ratios = np.fromstring(ratios[1:-1], dtype=float, sep=',')
        self._grad_scale = float(grad_scale)

    def list_arguments(self):
        return ['data', 'bbox_target', 'bbox_weight']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        bbox_target_shape = in_shape[1]
        bbox_weight_shape = in_shape[2]

        # share batch size
        batch_sizes = [data_shape[0], bbox_target_shape[0], bbox_weight_shape[0]]
        check_equal(batch_sizes, 'inconsistent batch size')

        out_shape = [data_shape]
        return in_shape, out_shape

    def create_operator(self, ctx, shapes, dtypes):
        return RPNIoULossOperator(self._feature_stride, self._scales, self._ratios, self._grad_scale)
