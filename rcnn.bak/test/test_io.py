import argparse
import random
import time
import mxnet as mx
import numpy as np

from ..config import config, default, generate_config
from ..symbol import *
from ..dataset import *
from ..io import image
from ..core.loader import AnchorLoader


def test_io(args):
    # set config
    config.TRAIN.END2END = True

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
    # train_data = mx.io.PrefetchingIter(train_data)

    t = time.time()
    for data_batch in train_data:
        print time.time() - t
        t = time.time()


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
    test_io(args)

if __name__ == '__main__':
    colors = [(random.random(), random.random(), random.random()) for _ in xrange(config.NUM_CLASSES)]
    main()
