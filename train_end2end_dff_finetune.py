# import sys
# sys.path.insert(0, '/home/syzhang/mxnet_0.7.0/python/')
import argparse
import logging
import pprint
import datetime
import os

import mxnet as mx
import numpy as np

from rcnn_dff.config import config, default, generate_config
from rcnn_dff.symbol import *
from rcnn_dff.core import callback, metric
from rcnn_dff.core.loader import AnchorLoader
from rcnn_dff.core.module import MutableModule
from rcnn_dff.utils.load_data import load_gt_roidb, merge_roidb, filter_roidb
from rcnn_dff.utils.load_model import load_param


def train_net(args, ctx, pretrained, epoch, prefix, begin_epoch, end_epoch,
              lr=0.001, lr_step='5'):
    # set up logger
    # logging.basicConfig()
    # logger = logging.getLogger()
    # logger.setLevel(logging.INFO)

    # new logger that also save log file on the dist
    d = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") # create log file name
    log_path = os.path.join('log', os.path.basename(prefix)) #get prefix basename: os.path.basename('zhang/song')='song'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=os.path.join(log_path, d + '_epoch'+ str(end_epoch)+'_finetune.log'),
                        filemode='w')

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    logging.info('start with arguments %s', args)

    # d = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") # create log file name
    # log_path = os.path.join('log', os.path.basename(prefix)) #get prefix basename: os.path.basename('zhang/song')='song'
    # if not os.path.exists(log_path):
    #     os.makedirs(log_path)
    # logger = logging.basicConfig(level=logging.DEBUG,
    #                     format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
    #                     datefmt='%m-%d %H:%M',
    #                     filename=os.path.join(log_path, d + '_epoch'+ str(end_epoch)+'.log'),
    #                     filemode='w')
    # logger = logging.getLogger()
    # logger.setLevel(logging.INFO)

    # setup config
    config.TRAIN.BATCH_IMAGES = 1
    config.TRAIN.BATCH_ROIS = 128
    config.TRAIN.END2END = True
    config.TRAIN.BBOX_NORMALIZATION_PRECOMPUTED = True
    config.TRAIN.BATCH_SIZE = 1

    # load symbol
    # use vgg_dff symbol
    sym = eval('get_' + args.network + '_train_dff')(num_classes=config.NUM_CLASSES, num_anchors=config.NUM_ANCHORS)
    feat_sym = sym.get_internals()['rpn_cls_score_output']
    print "========symbol load"
    # "data":(2,3,384,1280),"data2":(2,3,384,1280)
    # net = mx.viz.plot_network(feat_sym, shape={'data':(2,3,376,1242),'data2':(2,3,384,1280)}, node_attrs={"shape": "rect", "fixedsize":"false"})
    # net.render('feature_flow', view = True)

    # setup multi-gpu
    batch_size = len(ctx)
    input_batch_size = config.TRAIN.BATCH_IMAGES * batch_size

    # print config
    # pprint.pprint(config)

    # load dataset and prepare imdb for training
    image_sets = [iset for iset in args.image_set.split('+')]
    print "========image set"
    roidbs = [load_gt_roidb(args.dataset, image_set, args.root_path, args.dataset_path,
                            flip=not args.no_flip)
              for image_set in image_sets]
    # print roidbs
    roidb = merge_roidb(roidbs)
    roidb = filter_roidb(roidb)

    print "========roidb load"

    # load training data
    train_data = AnchorLoader(feat_sym, roidb, batch_size=input_batch_size, shuffle=not args.no_shuffle,
                              ctx=ctx, work_load_list=args.work_load_list,
                              feat_stride=config.RPN_FEAT_STRIDE, anchor_scales=config.ANCHOR_SCALES,
                              anchor_ratios=config.ANCHOR_RATIOS, aspect_grouping=config.TRAIN.ASPECT_GROUPING)


    print "========load data complete"
    # =====================
    # max_data_shape = [('data', (config.TRAIN.BATCH_IMAGES, 3, max([v[0] + 8 for v in config.SCALES]), max([v[1] + 38 for v in config.SCALES]))), ('data2', (config.TRAIN.BATCH_IMAGES, 3, max([v[0] + 8 for v in config.SCALES]), max([v[1] + 38 for v in config.SCALES])))]
    # max_data_shape, max_label_shape = train_data.infer_shape(max_data_shape)
    # max_data_shape.append(('gt_boxes', (config.TRAIN.BATCH_SIZE, 100, 5)))
    # print 'providing maximum shape', max_data_shape, max_label_shape

    # =====================
    # infer max shape
    if config.TRAIN.CROP == 'origin':
        max_data_shape = [('data', (input_batch_size, 3, max([v[0] for v in config.SCALES]), max([v[1] for v in config.SCALES]))),('data2', (input_batch_size, 3, max([v[0] for v in config.SCALES]), max([v[1] for v in config.SCALES])))]
        # max_data_shape = [('data', (input_batch_size, 3, \
        #                             max([v[0] + 8 for v in config.SCALES]), \
        #                             max([v[1] + 38 for v in config.SCALES]))), \
        #                   ('data2', (input_batch_size, 3, \
        #                             max([v[0] + 8 for v in config.SCALES]), \
        #                             max([v[1] + 38 for v in config.SCALES])))]
        max_data_shape, max_label_shape = train_data.infer_shape(max_data_shape)
        max_data_shape.append(('gt_boxes', (input_batch_size, 100, 5)))
        print 'providing maximum shape', max_data_shape, max_label_shape
    else:
        max_data_shape = [('gt_boxes', (input_batch_size, 100, 5))]
        max_label_shape = None

    print max_data_shape
    # infer shape
    data_shape_dict = dict(train_data.provide_data + train_data.provide_label)
    arg_shape, out_shape, aux_shape = sym.infer_shape(**data_shape_dict)
    arg_shape_dict = dict(zip(sym.list_arguments(), arg_shape))
    out_shape_dict = zip(sym.list_outputs(), out_shape)
    aux_shape_dict = dict(zip(sym.list_auxiliary_states(), aux_shape))
    print 'output shape'
    pprint.pprint(out_shape_dict)

    # load and initialize params
    if args.resume:
        arg_params, aux_params = load_param(prefix, begin_epoch, convert=True)
    else:
        arg_params, aux_params = load_param(pretrained, epoch, convert=True)
    # initialize params

        # arg_params['conv5_1_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['conv5_1_weight'])
        # arg_params['conv5_1_bias'] = mx.nd.zeros(shape=arg_shape_dict['conv5_1_bias'])
        # arg_params['conv5_2_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['conv5_2_weight'])
        # arg_params['conv5_2_bias'] = mx.nd.zeros(shape=arg_shape_dict['conv5_2_bias'])
        # arg_params['conv5_3_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['conv5_3_weight'])
        # arg_params['conv5_3_bias'] = mx.nd.zeros(shape=arg_shape_dict['conv5_3_bias'])
        # arg_params['stereo_scale_weight'] = mx.nd.zeros(shape=arg_shape_dict['stereo_scale_weight'])
        # arg_params['stereo_scale_bias'] = mx.nd.ones(shape=arg_shape_dict['stereo_scale_bias'])
        # arg_params['rpn_conv_3x3_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['rpn_conv_3x3_weight'])
        # arg_params['rpn_conv_3x3_bias'] = mx.nd.zeros(shape=arg_shape_dict['rpn_conv_3x3_bias'])
        # arg_params['rpn_cls_score_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['rpn_cls_score_weight'])
        # arg_params['rpn_cls_score_bias'] = mx.nd.zeros(shape=arg_shape_dict['rpn_cls_score_bias'])
        # arg_params['rpn_bbox_pred_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['rpn_bbox_pred_weight'])
        # arg_params['rpn_bbox_pred_bias'] = mx.nd.zeros(shape=arg_shape_dict['rpn_bbox_pred_bias'])
        # arg_params['cls_score_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['cls_score_weight'])
        # arg_params['cls_score_bias'] = mx.nd.zeros(shape=arg_shape_dict['cls_score_bias'])
        # arg_params['bbox_pred_weight'] = mx.random.normal(0, 0.001, shape=arg_shape_dict['bbox_pred_weight'])
        # arg_params['bbox_pred_bias'] = mx.nd.zeros(shape=arg_shape_dict['bbox_pred_bias'])
        #

        # init = mx.init.Xavier(factor_type="in", rnd_type='gaussian', magnitude=2)
        # for k in sym.list_arguments():
        #     if k in data_shape_dict:
        #         continue
        #     if k not in arg_params:
        #         print 'init', k
        #         arg_params[k] = mx.nd.zeros(shape=arg_shape_dict[k])
        #         init(k, arg_params[k])
        #         arg_params[k][:] /= 10
        #         if 'ctx_red_weight' in k:
        #             ctx_shape = np.array(arg_shape_dict[k])
        #             ctx_shape[1] /= 2
        #             arg_params[k][:] = np.concatenate((np.eye(ctx_shape[1]).reshape(ctx_shape), np.zeros(ctx_shape)), axis=1)
        #
        # for k in sym.list_auxiliary_states():
        #     if k not in aux_params:
        #         print 'init', k
        #         aux_params[k] = mx.nd.zeros(shape=aux_shape_dict[k])
        #         init(k, aux_params[k])

    # load flownet parameter and merge with rcnn's
    flow_model_path = '/data/syzhang/model/flow/flow'
    _, flow_arg_params, flow_aux_params = mx.model.load_checkpoint(flow_model_path, 0)
    arg_params.update(flow_arg_params) # dict.update(dict2) add dict2 into dict
    aux_params.update(flow_aux_params)

    # check parameter shapes
    for k in sym.list_arguments():
        if k in data_shape_dict:
            continue
        assert k in arg_params, k + ' not initialized'
        assert arg_params[k].shape == arg_shape_dict[k], \
            'shape inconsistent for ' + k + ' inferred ' + str(arg_shape_dict[k]) + ' provided ' + str(arg_params[k].shape)
    for k in sym.list_auxiliary_states():
        assert k in aux_params, k + ' not initialized'
        assert aux_params[k].shape == aux_shape_dict[k], \
            'shape inconsistent for ' + k + ' inferred ' + str(aux_shape_dict[k]) + ' provided ' + str(aux_params[k].shape)

    # create solver
    fixed_param_prefix = config.FIXED_PARAMS
    fixed_param_prefix += flow_arg_params.keys() + flow_aux_params.keys()

    data_names = [k[0] for k in train_data.provide_data]
    label_names = [k[0] for k in train_data.provide_label]
    mod = MutableModule(sym, data_names=data_names, label_names=label_names, context=ctx, work_load_list=args.work_load_list,
                        max_data_shapes=max_data_shape, max_label_shapes=max_label_shape,
                        fixed_param_prefix=fixed_param_prefix)

    # decide training params
    # metric
    rpn_eval_metric = metric.RPNAccMetric()
    rpn_cls_metric = metric.RPNLogLossMetric()
    rpn_bbox_metric = metric.RPNRegLossMetric()
    eval_metric = metric.RCNNAccMetric()
    cls_metric = metric.RCNNLogLossMetric()
    bbox_metric = metric.RCNNRegLossMetric()
    eval_metrics = mx.metric.CompositeEvalMetric()
    for child_metric in [rpn_eval_metric, rpn_cls_metric, rpn_bbox_metric, eval_metric, cls_metric, bbox_metric]:
        eval_metrics.add(child_metric)
    # callback
    batch_end_callback = callback.Speedometer(train_data.batch_size, frequent=args.frequent)
    means = np.tile(np.array(config.TRAIN.BBOX_MEANS), config.NUM_CLASSES)
    stds = np.tile(np.array(config.TRAIN.BBOX_STDS), config.NUM_CLASSES)
    epoch_end_callback = callback.do_checkpoint(prefix, means, stds)
    # decide learning rate
    base_lr = lr
    lr_factor = 0.1
    lr_epoch = [int(epoch) for epoch in lr_step.split(',')]
    lr_epoch_diff = [epoch - begin_epoch for epoch in lr_epoch if epoch > begin_epoch]
    lr = base_lr * (lr_factor ** (len(lr_epoch) - len(lr_epoch_diff)))
    lr_iters = [int(epoch * len(roidb) / batch_size) for epoch in lr_epoch_diff]
    print 'lr', lr, 'lr_epoch_diff', lr_epoch_diff, 'lr_iters', lr_iters
    lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(lr_iters, lr_factor)
    # optimizer
    if not args.finetune:
        opt = 'sgd'
        optimizer_params = {'momentum': 0.9,
                            'wd': 0.0005,
                            'learning_rate': lr,
                            'lr_scheduler': lr_scheduler,
                            'rescale_grad': (1.0 / batch_size),
                            'clip_gradient': 5}
    else:
        opt = 'Adam'
        optimizer_params = {'learning_rate' : lr,
                            'lr_scheduler' : lr_scheduler,
                            'rescale_grad' : (1.0 / batch_size)}


    # train
    mod.fit(train_data, eval_metric=eval_metrics, epoch_end_callback=epoch_end_callback,
            batch_end_callback=batch_end_callback, kvstore=args.kvstore,
            optimizer=opt, optimizer_params=optimizer_params,
            arg_params=arg_params, aux_params=aux_params, begin_epoch=begin_epoch, num_epoch=end_epoch)


def parse_args():
    parser = argparse.ArgumentParser(description='Train Faster R-CNN network')
    # general
    parser.add_argument('--network', help='network name', default=default.network, type=str)
    parser.add_argument('--dataset', help='dataset name', default=default.dataset, type=str)
    args, rest = parser.parse_known_args()
    generate_config(args.network, args.dataset)
    parser.add_argument('--image_set', help='image_set name', default=default.image_set, type=str)
    parser.add_argument('--root_path', help='output data folder', default=default.root_path, type=str)
    parser.add_argument('--dataset_path', help='dataset path', default=default.dataset_path, type=str)
    # training
    parser.add_argument('--frequent', help='frequency of logging', default=default.frequent, type=int)
    parser.add_argument('--kvstore', help='the kv-store type', default=default.kvstore, type=str)
    parser.add_argument('--work_load_list', help='work load for different devices', default=None, type=list)
    parser.add_argument('--no_flip', help='disable flip images', action='store_true')
    parser.add_argument('--no_shuffle', help='disable random shuffle', action='store_true')
    parser.add_argument('--resume', help='continue training', action='store_true')
    parser.add_argument('--finetune', help='continue training', action='store_true')
    # e2e
    parser.add_argument('--gpus', help='GPU device to train with', default='0', type=str)
    parser.add_argument('--pretrained', help='pretrained model prefix', default=default.pretrained, type=str)
    parser.add_argument('--pretrained_epoch', help='pretrained model epoch', default=default.pretrained_epoch, type=int)
    parser.add_argument('--prefix', help='new model prefix', default=default.e2e_prefix, type=str)
    parser.add_argument('--begin_epoch', help='begin epoch of training, use with resume', default=0, type=int)
    parser.add_argument('--end_epoch', help='end epoch of training', default=default.e2e_epoch, type=int)
    parser.add_argument('--lr', help='base learning rate', default=default.e2e_lr, type=float)
    parser.add_argument('--lr_step', help='learning rate steps (in epoch)', default=default.e2e_lr_step, type=str)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print 'Called with argument:', args
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',')]
    train_net(args, ctx, args.pretrained, args.pretrained_epoch, args.prefix, args.begin_epoch, args.end_epoch,
              lr=args.lr, lr_step=args.lr_step)

if __name__ == '__main__':
    main()