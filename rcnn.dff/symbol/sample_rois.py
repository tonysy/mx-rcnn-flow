import mxnet as mx
import numpy as np
from distutils.util import strtobool

from ..processing.bbox_transform import iou_pred, nonlinear_pred
from ..processing.nms import py_nms_wrapper, cpu_nms_wrapper, gpu_nms_wrapper


def check_equal(lst, errstr='check_equal'):
    assert len(set(lst)) <= 1, '%s:%s' % (errstr, lst)


class SampleRoisOperator(mx.operator.CustomOp):
    """
    RCNN OHEM
    input: ['cls_prob', 'bbox_loss', 'label', 'rois', 'bbox_pred']
    output: ['cls_prob_out', 'bbox_loss_out', 'cls_mask', 'bbox_mask']
    in_shape:
        cls_prob: (batch_size, num_classes)
        bbox_loss: (batch_size, num_classes * 4)
        label: (batch_size,)
        rois: (batch_size, 5)
        bbox_pred: (batch_size, num_classes * 4)
    out_shape:
        cls_prob_out: (batch_size, num_classes)
        bbox_loss_out: (batch_size, num_classes * 4)
        cls_mask: (batch_size, num_classes)
        bbox_mask: (batch_size, num_classes * 4)
    """
    def __init__(self, batch_images, batch_size, nms_threshold,
                 iou_loss, ignore, transform):
        super(SampleRoisOperator, self).__init__()
        self._batch_images = batch_images
        self._batch_size = batch_size
        self._nms_threshold = nms_threshold
        self._bbox_pred = iou_pred if iou_loss else nonlinear_pred
        self._ignore = ignore
        self._transform = transform

    def forward(self, is_train, req, in_data, out_data, aux):
        nms = cpu_nms_wrapper(self._nms_threshold)

        cls_prob = in_data[0].asnumpy()
        bbox_loss = in_data[1].asnumpy()
        label = in_data[2].asnumpy()
        rois = in_data[3].asnumpy()
        bbox_deltas = in_data[4].asnumpy()

        # calculate log loss
        label = label.astype(np.int32)
        cls_prob = cls_prob[np.arange(label.shape[0]), label]
        cls_loss = -1 * np.log(cls_prob + 1e-14)

        # calculate loss
        bbox_loss = np.sum(bbox_loss, axis=1)
        loss = cls_loss + bbox_loss

        # dets is [rois, loss] drop batch_index
        if bool(self._transform):
            proposals = self._bbox_pred(rois[:, 1:], bbox_deltas)
            lineidx = np.arange(label.shape[0])
            proposals = np.hstack((proposals[lineidx, 4 * label + 0].reshape(-1, 1),
                                   proposals[lineidx, 4 * label + 1].reshape(-1, 1),
                                   proposals[lineidx, 4 * label + 2].reshape(-1, 1),
                                   proposals[lineidx, 4 * label + 3].reshape(-1, 1)))
        else:
            proposals = rois[:, 1:]

        # assemble dets
        dets = np.hstack((proposals, loss[:, np.newaxis])).astype(np.float32)

        rois_per_image = self._batch_size / self._batch_images
        keep_inds = []
        for im_i in range(self._batch_images):
            index_i = np.where(rois[:, 0] == im_i)[0]

            # ohem
            dets_i = dets[index_i, :]
            keep_ind = nms(dets_i)

            # select top loss examples
            start = int(self._ignore * len(keep_ind))
            keep_ind = keep_ind[start:start + rois_per_image]
            keep_inds_i = index_i[keep_ind]

            # select positive examples
            # pos_index_i = np.where(label[index_i] > 0)[0]
            # keep_inds_i = np.append(keep_inds_i, pos_index_i)

            # add to keep_inds
            keep_inds.append(keep_inds_i)
        keep_inds = np.concatenate(keep_inds).astype(np.int32)

        cls_mask = np.zeros(in_data[0].shape)
        cls_mask[keep_inds, :] = 1

        bbox_mask = np.zeros(in_data[1].shape)
        bbox_mask[keep_inds, :] = 1

        self.assign(out_data[0], req[0], in_data[0])
        self.assign(out_data[1], req[1], in_data[1])
        self.assign(out_data[2], req[2], cls_mask)
        self.assign(out_data[3], req[3], bbox_mask)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], out_data[2])
        self.assign(in_grad[1], req[1], out_data[3] / self._batch_size)
        self.assign(in_grad[2], req[2], 0)
        self.assign(in_grad[3], req[3], 0)
        self.assign(in_grad[4], req[4], 0)


@mx.operator.register('sample_rois')
class SampleRoisProp(mx.operator.CustomOpProp):
    def __init__(self, batch_images='2', batch_size='128', nms_threshold='0.7',
                 iou_loss='False', ignore='0', transform='False'):
        # iou_loss: different bbox_pred
        # ignore: give up top hard examples, e.g. 0.05
        # transform: nms on the transformed rois
        super(SampleRoisProp, self).__init__(need_top_grad=False)
        self.batch_images = int(batch_images)
        self.batch_size = int(batch_size)
        self.nms_threshold = float(nms_threshold)
        self._iou_loss = strtobool(iou_loss)
        self._ignore = float(ignore)
        self._transform = strtobool(transform)

    def list_arguments(self):
        return ['cls_prob', 'bbox_loss', 'label', 'rois', 'bbox_pred']

    def list_outputs(self):
        return ['cls_prob_out', 'bbox_loss_out', 'cls_mask', 'bbox_mask']

    def infer_shape(self, in_shape):
        cls_prob_shape = in_shape[0]
        bbox_loss_shape = in_shape[1]
        label_shape = in_shape[2]
        rois_shape = in_shape[3]
        bbox_pred_shape = in_shape[4]

        # share batch size
        batch_sizes = [cls_prob_shape[0], label_shape[0],
                       bbox_loss_shape[0], rois_shape[0], bbox_pred_shape[0]]
        check_equal(batch_sizes, 'inconsistent batch size')

        out_shape = [cls_prob_shape, bbox_loss_shape, cls_prob_shape, bbox_loss_shape]
        return in_shape, out_shape

    def create_operator(self, ctx, shapes, dtypes):
        return SampleRoisOperator(self.batch_images, self.batch_size, self.nms_threshold,
                                  self._iou_loss, self._ignore, self._transform)
