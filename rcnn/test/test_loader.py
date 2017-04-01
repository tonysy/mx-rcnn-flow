import argparse
import random
import mxnet as mx
import numpy as np

from ..config import config, default, generate_config
from ..symbol import *
from ..dataset import *
from ..io import image
from ..core.loader import AnchorLoader
from ..processing.generate_anchor import generate_anchors
from ..processing.bbox_transform import iou_pred, nonlinear_pred


def process_proposals(feat_shape, label, bbox_deltas, feat_stride=16, scales=(8, 16, 32), ratios=(0.5, 1, 2)):
    bbox_pred = iou_pred if config.RPN_IOU_LOSS else nonlinear_pred
    base_anchors = generate_anchors(base_size=feat_stride, scales=np.array(scales), ratios=np.array(ratios))
    num_anchors = base_anchors.shape[0]
    height, width = feat_shape[-2:]

    shift_x = np.arange(0, width) * feat_stride
    shift_y = np.arange(0, height) * feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()

    A = num_anchors
    K = shifts.shape[0]
    anchors = base_anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
    anchors = anchors.reshape((K * A, 4))
    bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, 4))
    labels = label.reshape((1, A, height, width))
    labels = labels.transpose((0, 2, 3, 1)).reshape((-1, 1))
    assert anchors.shape[0] == bbox_deltas.shape[0]
    assert bbox_deltas.shape[0] == labels.shape[0]

    assigned_inds = np.where(labels == 1)[0]
    proposals = bbox_pred(anchors[assigned_inds, :], bbox_deltas[assigned_inds, :])
    return anchors[assigned_inds, :], proposals


def vis_all_detection(im_array, detections, class_names, scale):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(16, 9), dpi=90)
    im = image.transform_inverse(im_array, config.PIXEL_MEANS)
    plt.imshow(im)
    for j, name in enumerate(class_names):
        if name == '__background__':
            continue
        color = colors[j]
        dets = detections[j]
        for det in dets:
            bbox = det[:4] * scale
            rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                                 fill=False, edgecolor=color, linewidth=2)
            plt.gca().add_patch(rect)
        plt.plot([0], [0], color=color, label=name)
    plt.gca().legend()
    plt.show()


def test_loader(args):
    # load symbol
    sym = eval('get_' + args.network + '_rpn')(num_anchors=config.NUM_ANCHORS)
    feat_sym = sym.get_internals()['rpn_cls_score_output']

    # load dataset and prepare imdb for training
    imdb = eval(args.dataset)(args.image_set, args.root_path, args.dataset_path)
    roidb = imdb.gt_roidb()

    # load training data
    train_data = AnchorLoader(feat_sym, roidb, batch_size=1, shuffle=not args.no_shuffle,
                              feat_stride=config.RPN_FEAT_STRIDE, anchor_scales=config.ANCHOR_SCALES,
                              anchor_ratios=config.ANCHOR_RATIOS)
    train_data.data_name += ['im_info', 'gt_boxes']

    # start visualize
    while True:
        data_batch = train_data.next()
        data_dict = dict(zip(train_data.data_name, data_batch.data) + zip(train_data.label_name, data_batch.label))
        data = data_dict['data'].asnumpy()
        label = data_dict['label'].asnumpy()
        bbox_deltas = data_dict['bbox_target'].asnumpy()
        gt_boxes = data_dict['gt_boxes'].asnumpy()[0]
        feat_shape = bbox_deltas.shape
        anchors, proposals = \
            process_proposals(feat_shape, label, bbox_deltas,
                              feat_stride=config.RPN_FEAT_STRIDE, scales=config.ANCHOR_SCALES, ratios=config.ANCHOR_RATIOS)
        dets = [anchors, proposals, gt_boxes]
        cls_names = ['anchor', 'proposal', 'gt_boxes']
        vis_all_detection(data, dets, cls_names, scale=1.0)


def parse_args():
    parser = argparse.ArgumentParser(description='Anchor loader test')
    # general
    parser.add_argument('--network', help='network name', default=default.network, type=str)
    parser.add_argument('--dataset', help='dataset name', default=default.dataset, type=str)
    args, rest = parser.parse_known_args()
    generate_config(args.network, args.dataset)
    parser.add_argument('--image_set', help='image_set name', default=default.image_set, type=str)
    parser.add_argument('--root_path', help='output data folder', default=default.root_path, type=str)
    parser.add_argument('--dataset_path', help='dataset path', default=default.dataset_path, type=str)
    # other
    parser.add_argument('--no_shuffle', help='disable random shuffle', action='store_true')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print 'Called with argument:', args
    test_loader(args)

if __name__ == '__main__':
    colors = [(random.random(), random.random(), random.random()) for _ in xrange(config.NUM_CLASSES)]
    main()
