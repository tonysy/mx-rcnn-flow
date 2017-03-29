import mxnet as mx
import numpy as np
from distutils.util import strtobool

from ..processing.generate_anchor import generate_anchors
from ..processing.bbox_transform import iou_pred, nonlinear_pred
from ..processing.nms import py_nms_wrapper, cpu_nms_wrapper, gpu_nms_wrapper


def check_equal(lst, errstr='check_equal'):
    assert len(set(lst)) <= 1, '%s:%s' % (errstr, lst)


class SampleAnchorsOperator(mx.operator.CustomOp):
    """
    RPN OHEM:
    input: ['cls_prob', 'bbox_loss', 'label', 'bbox_pred']
    hyperparams: basic anchors
    output: ['cls_prob', 'bbox_pred', 'cls_mask', 'bbox_mask']
    in_shape:
        cls_prob: (batch_images, 2, a*h*w)
        bbox_loss: (batch_images, a*4, h, w)
        label: (batch_images, a*h*w)
        bbox_pred: (batch_images, a*4, h, w)
    out_shape:
        cls_prob_out: (batch_images, 2, a*h*w)
        bbox_loss_out: (batch_images, a*4, h, w)
        cls_mask: (batch_images, 2, a*h*w)
        bbox_mask: (batch_images, a*4, h, w)
    procedure:
        1. calculate log loss of softmax_prob
        2. predict boxes (applying bbox_pred)
        3. nms
        4. sort [logloss + bbox_loss] and backprop grad
    """
    def __init__(self, feature_stride, scales, ratios,
                 rpn_pre_nms_top_n, rpn_batch_size, nms_threshold,
                 iou_loss, ignore, transform, np_ratio):
        super(SampleAnchorsOperator, self).__init__()
        self._feat_stride = feature_stride
        self._anchors = generate_anchors(base_size=feature_stride, scales=scales, ratios=ratios)
        self._rpn_pre_nms_top_n = rpn_pre_nms_top_n
        self._rpn_batch_size = rpn_batch_size
        self._nms_threshold = nms_threshold
        self._bbox_pred = iou_pred if iou_loss else nonlinear_pred
        self._ignore = ignore
        self._transform = transform
        self._np_ratio = np_ratio

    def forward(self, is_train, req, in_data, out_data, aux):
        nms = gpu_nms_wrapper(self._nms_threshold, in_data[0].context.device_id)

        cls_prob = in_data[0].asnumpy()
        bbox_loss = in_data[1].asnumpy()
        label = in_data[2].asnumpy()
        bbox_deltas = in_data[3].asnumpy()

        # get shape and check consistency
        batch_size, num_anchor, height, width = bbox_deltas.shape
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
        # reshape anchors to (b, h*w*a, 4)
        if batch_size > 1:
            anchors = np.tile(anchors, (batch_size, 1))
        anchors = anchors.reshape((batch_size, -1, 4))

        # calculate log loss
        # cls_prob is now (b, 2, a*h, w)
        cls_prob_flat_shape = (cls_prob.shape[0], cls_prob.shape[1], cls_prob.shape[2] * cls_prob.shape[3])
        cls_prob = cls_prob.reshape(cls_prob_flat_shape)
        # turn cls_prob from (b, 2, a*h*w) to (b*a*h*w, 2)
        # label from (b, a*h*w) to (b*a*h*w)
        label = label.astype(np.int32).reshape((-1))
        cls_prob = cls_prob.transpose((0, 2, 1)).reshape((label.shape[0], -1))
        cls_prob = cls_prob[np.arange(label.shape[0]), label]
        cls_loss = -1 * np.log(cls_prob + 1e-14)
        # softmax has ignore_label; set cls_loss to be 0 (seanlx)
        # so that they will not be treated as negative example
        cls_loss[label == -1] = 0

        # turn label from (b*a*h*w) to (b, h, w, a) to (b, h*w*a)
        label = label.reshape((batch_size, num_anchor, height, width))
        label = label.transpose((0, 2, 3, 1)).reshape((batch_size, -1))
        # turn log loss from (b*a*h*w) to (b, h, w, a) to (b, h*w*a)
        cls_loss = cls_loss.reshape((batch_size, num_anchor, height, width))
        cls_loss = cls_loss.transpose((0, 2, 3, 1)).reshape((batch_size, -1))
        # turn bbox from (b, a*4, h, w) to (b,h,w,a*4) to (b, h*w*a, 4)
        # and then reduce the last dimension
        bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1)).reshape((batch_size, -1, 4))
        bbox_loss = bbox_loss.transpose((0, 2, 3, 1)).reshape((batch_size, -1, 4))
        bbox_loss_red = bbox_loss.sum(axis=2)

        # # sanity check for label and bbox_loss
        # check_equal([len(np.where(label == 1)[1]), len(np.where(bbox_loss_red > 0)[1])])
        # # while label == 1, bbox_loss must be > 0
        # check_equal([len(np.where(label == 1)[1]), len(np.where(bbox_loss_red[label == 1])[0])])

        # convert anchors to proposals
        all_loss = cls_loss + bbox_loss_red
        # initialize output
        cls_mask = np.zeros(cls_prob_flat_shape)
        bbox_mask = np.zeros(in_data[1].shape)
        for batch_index, label_i in enumerate(label):
            anchor = anchors[batch_index]
            bbox_delta = bbox_deltas[batch_index]
            loss = all_loss[batch_index]

            pos_ind = np.where(label_i > 0)[0]
            loss[pos_ind] = 0
            num_pos = len(pos_ind)
            num_neg = int(num_pos * self._np_ratio)

            if self._transform:
                # proposal only predicts foreground class (rcnn will have background)
                proposal = self._bbox_pred(anchor, bbox_delta)
            else:
                proposal = anchor

            # ohem
            pre_inds = np.argsort(loss)[::-1][:self._rpn_pre_nms_top_n]
            det = np.hstack((proposal[pre_inds, :], loss[pre_inds, np.newaxis])).astype(np.float32)
            keep_ind = nms(det)  # no need to pad

            # select neg examples by bootstrap
            start = int(self._ignore * len(keep_ind))
            keep_ind = keep_ind[start:start + num_neg]
            keep_ind = pre_inds[keep_ind]

            # select pos examples
            # pos_ind = np.where(label[batch_index] > 0)[0]
            keep_ind = np.append(keep_ind, pos_ind)

            # check
            # print 'selected pos label', len(np.where(label[0, keep_ind] == 1)[0])
            # print 'selected pos bbox', len(np.where(bbox_loss_red[0, keep_ind] > 0)[0])

            # convert back to spatial index in h*w*a
            ind_h, ind_w, ind_a = np.unravel_index(keep_ind, (height, width, num_anchor))
            keep_ind = np.ravel_multi_index((ind_a, ind_h, ind_w), (num_anchor, height, width))
            # output cls_mask
            cls_mask[batch_index, :, keep_ind] = 1
            # in_a should be step by 4
            bbox_mask_red = np.zeros((1, num_anchor, height, width))
            bbox_mask_red[0, ind_a, ind_h, ind_w] = 1
            bbox_mask[batch_index] = np.repeat(bbox_mask_red, 4, axis=1)
            # bbox_mask[batch_index, 4 * ind_a, ind_h, ind_w] = 1
            # bbox_mask[batch_index, 4 * ind_a + 1, ind_h, ind_w] = 1
            # bbox_mask[batch_index, 4 * ind_a + 2, ind_h, ind_w] = 1
            # bbox_mask[batch_index, 4 * ind_a + 3, ind_h, ind_w] = 1

            # pos label check
            # label = in_data[2].asnumpy()
            # print 'original pos label', np.sum(label[batch_index, keep_ind] == 1)
            # output bbox_mask
            # pos bbox check
            # bbox_loss_o = in_data[1].asnumpy()
            # bbox_loss_o_red = bbox_loss_o[:, ::4, :, :] + bbox_loss_o[:, 1::4, :, :] + \
            #                   bbox_loss_o[:, 2::4, :, :] + bbox_loss_o[:, 3::4, :, :]
            # print 'original pos bbox', \
            #     np.sum(bbox_loss_o_red[batch_index, ind_a, ind_h, ind_w] > 0)

        # output ['cls_prob_out', 'bbox_loss_out', 'cls_mask', 'bbox_mask']
        self.assign(out_data[0], req[0], in_data[0])
        self.assign(out_data[1], req[1], in_data[1])
        self.assign(out_data[2], req[2], cls_mask.reshape(in_data[0].shape))
        self.assign(out_data[3], req[3], bbox_mask)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        # in_grad: ['cls_prob', 'bbox_loss', 'label', 'bbox_pred']
        # put cls_mask and bbox_mask to in_grad of cls_prob and bbox_loss
        # normalize bbox_loss by total selected box
        ncount = mx.nd.sum(out_data[3]) / 4
        self.assign(in_grad[0], req[0], out_data[2])
        self.assign(in_grad[1], req[1], out_data[3] / ncount)
        self.assign(in_grad[2], req[2], 0)
        self.assign(in_grad[3], req[3], 0)


@mx.operator.register('sample_anchors')
class SampleAnchorsProp(mx.operator.CustomOpProp):
    def __init__(self, feature_stride='16', scales='(8, 16, 32)', ratios='(0.5, 1, 2)',
                 rpn_pre_nms_top_n='12000', rpn_batch_size='256', nms_threshold='0.7',
                 iou_loss='False', ignore='0', transform='False', np_ratio='2'):
        # iou_loss: different bbox_pred
        # ignore: give up top hard examples, e.g. 0.05
        # transform: nms on the transformed rois
        super(SampleAnchorsProp, self).__init__(need_top_grad=False)
        self._feature_stride = int(feature_stride)
        self._scales = np.fromstring(scales[1:-1], dtype=float, sep=',')
        self._ratios = np.fromstring(ratios[1:-1], dtype=float, sep=',')
        self._rpn_pre_nms_top_n = int(rpn_pre_nms_top_n)
        self._rpn_batch_size = int(rpn_batch_size)
        self._nms_threshold = float(nms_threshold)
        self._iou_loss = strtobool(iou_loss)
        self._ignore = float(ignore)
        self._transform = strtobool(transform)
        self._np_ratio = float(np_ratio)

    def list_arguments(self):
        return ['cls_prob', 'bbox_loss', 'label', 'bbox_pred']

    def list_outputs(self):
        return ['cls_prob_out', 'bbox_loss_out', 'cls_mask', 'bbox_mask']

    def infer_shape(self, in_shape):
        cls_prob_shape = in_shape[0]
        bbox_loss_shape = in_shape[1]
        label_shape = in_shape[2]
        bbox_pred_shape = in_shape[3]

        # share batch size
        batch_sizes = [cls_prob_shape[0], label_shape[0],
                       bbox_pred_shape[0], bbox_loss_shape[0]]
        check_equal(batch_sizes, 'inconsistent batch size')
        # share spatial dimension
        spatial_dims = [cls_prob_shape[2] * cls_prob_shape[3], label_shape[1],
                        bbox_pred_shape[1] / 4 * bbox_pred_shape[2] * bbox_pred_shape[3],
                        bbox_loss_shape[1] / 4 * bbox_loss_shape[2] * bbox_loss_shape[3]]
        check_equal(spatial_dims, 'inconsistent spatial dimension')

        out_shape = [cls_prob_shape, bbox_loss_shape, cls_prob_shape, bbox_loss_shape]
        return in_shape, out_shape

    def create_operator(self, ctx, shapes, dtypes):
        return SampleAnchorsOperator(self._feature_stride, self._scales, self._ratios,
                                     self._rpn_pre_nms_top_n, self._rpn_batch_size, self._nms_threshold,
                                     self._iou_loss, self._ignore, self._transform, self._np_ratio)
