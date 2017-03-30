import argparse
import cv2
import random
import mxnet as mx
import numpy as np

from ..config import config, default, generate_config
from ..symbol import *
from ..dataset import *
from ..io import image


def vis_all_detection(im_array, classes, boxes, class_names, scale):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(16, 9), dpi=90)
    im = image.transform_inverse(im_array, config.PIXEL_MEANS)
    plt.imshow(im)
    for cls, det in zip(classes, boxes):
        color = colors[cls]
        bbox = det * scale
        rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                             fill=False, edgecolor=color, linewidth=2)
        plt.gca().add_patch(rect)
    for color, name in zip(colors, class_names):
        plt.plot([0], [0], color=color, label=name)
    plt.gca().legend()
    plt.show()


def test_imdb(args):
    # load dataset and prepare imdb for training
    imdb = eval(args.dataset)(args.image_set, args.root_path, args.dataset_path)
    roidb = imdb.gt_roidb()

    # start vis
    if not args.no_shuffle:
        np.random.shuffle(roidb)
    for roi_rec in roidb:
        im = cv2.imread(roi_rec['image'])
        im_array = image.transform(im, config.PIXEL_MEANS)
        gt_classes = roi_rec['gt_classes']
        gt_boxes = roi_rec['boxes']
        vis_all_detection(im_array, gt_classes, gt_boxes, imdb.classes, 1.0)


def parse_args():
    parser = argparse.ArgumentParser(description='imdb test')
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
    test_imdb(args)


if __name__ == '__main__':
    colors = [(random.random(), random.random(), random.random()) for _ in xrange(config.NUM_CLASSES)]
    main()
