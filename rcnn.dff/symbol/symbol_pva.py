import mxnet as mx
import proposal
import proposal_target
from rcnn.config import config

# inception settings
use_global_stats = True
eps = 1e-10 + 1e-5
dilate = (1, 1)


def ConvFactory(data, num_filter, kernel, stride=(1, 1), pad=(0, 0), dilate=(1, 1), name=None, prefix=''):
    conv = mx.symbol.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, dilate=dilate, no_bias=True,
                                 name='%s%s/conv' % (prefix, name))
    bn = mx.symbol.BatchNorm(data=conv, name='%s%s/bn' % (prefix, name), fix_gamma=False, eps=eps, use_global_stats=use_global_stats)
    act = mx.symbol.Activation(data=bn, act_type='relu', name='%s%s' % (prefix, name))
    return act


# no relu
def ConvFactory2(data, num_filter, kernel, stride=(1, 1), pad=(0, 0), dilate=(1, 1), name=None, prefix=''):
    conv = mx.symbol.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, dilate=dilate, no_bias=True,
                                 name='%s%s/conv' % (prefix, name))
    bn = mx.symbol.BatchNorm(data=conv, name='%s%s/bn' % (prefix, name), fix_gamma=False, eps=eps, use_global_stats=use_global_stats)
    return bn


# has bias
def ProjFactory(data, num_filter, kernel, stride=(1, 1), pad=(0, 0), dilate=(1, 1), name=None, prefix=''):
    conv = mx.symbol.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, dilate=dilate,
                                 name='%s%s/conv' % (prefix, name))
    bn = mx.symbol.BatchNorm(data=conv, name='%s%s/bn' % (prefix, name), fix_gamma=False, eps=eps, use_global_stats=use_global_stats)
    return bn


def CReLUFactory(data, num_filter, kernel, stride=(1, 1), pad=(0, 0), dilate=(1, 1), name=None, prefix=''):
    conv = mx.symbol.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, dilate=dilate, no_bias=True,
                                 name='%s%s/conv' % (prefix, name))
    neg = mx.symbol.negative(data=conv, name='%s%s/neg' % (prefix, name))
    concat = mx.symbol.Concat(*[conv, neg], name='%s%s/concat' % (prefix, name))
    bn = mx.symbol.BatchNorm(data=concat, name='%s%s/bn' % (prefix, name), fix_gamma=False, eps=eps, use_global_stats=use_global_stats)
    act = mx.symbol.Activation(data=bn, act_type='relu', name='%s%s' % (prefix, name))
    return act


# bottleneck
def KKCReLUFactory_x(data, num_1x1_a, num_3x3_b, num_1x1_c, name=None, prefix=''):
    branch2a = ConvFactory(data, num_filter=num_1x1_a, kernel=(1, 1), stride=(1, 1), pad=(0, 0), name='%s/1' % name, prefix=prefix)
    branch2b = CReLUFactory(branch2a, num_filter=num_3x3_b, kernel=(3, 3), stride=(1, 1), pad=(1, 1), name='%s/2' % name, prefix=prefix)
    branch2c = ConvFactory2(branch2b, num_filter=num_1x1_c, kernel=(1, 1), stride=(1, 1), pad=(0, 0), name='%s/3' % name, prefix=prefix)
    summ = mx.symbol.ElementWiseSum(*[data, branch2c], name=('%s/sum' % name))
    return summ


# with projection
def KKCReLUFactory_o(data, num_1x1_a, num_3x3_b, num_1x1_c, name=None, prefix=''):
    branch1 = ProjFactory(data, num_filter=num_1x1_c, kernel=(1, 1), stride=(1, 1), pad=(0, 0), name='%s/proj' % name, prefix=prefix)
    branch2a = ConvFactory(data, num_filter=num_1x1_a, kernel=(1, 1), stride=(1, 1), pad=(0, 0), name='%s/1' % name, prefix=prefix)
    branch2b = CReLUFactory(branch2a, num_filter=num_3x3_b, kernel=(3, 3), stride=(1, 1), pad=(1, 1), name='%s/2' % name, prefix=prefix)
    branch2c = ConvFactory2(branch2b, num_filter=num_1x1_c, kernel=(1, 1), stride=(1, 1), pad=(0, 0), name='%s/3' % name, prefix=prefix)
    summ = mx.symbol.ElementWiseSum(*[branch1, branch2c], name=('%s/sum' % name))
    return summ


# with projection and downsampling
def KKCReLUFactory_d(data, num_1x1_a, num_3x3_b, num_1x1_c, name=None, prefix=''):
    branch1 = ProjFactory(data, num_filter=num_1x1_c, kernel=(1, 1), stride=(2, 2), pad=(0, 0), name='%s/proj' % name, prefix=prefix)
    branch2a = ConvFactory(data, num_filter=num_1x1_a, kernel=(1, 1), stride=(2, 2), pad=(0, 0), name='%s/1' % name, prefix=prefix)
    branch2b = CReLUFactory(branch2a, num_filter=num_3x3_b, kernel=(3, 3), stride=(1, 1), pad=(1, 1), name='%s/2' % name, prefix=prefix)
    branch2c = ConvFactory2(branch2b, num_filter=num_1x1_c, kernel=(1, 1), stride=(1, 1), pad=(0, 0), name='%s/3' % name, prefix=prefix)
    summ = mx.symbol.ElementWiseSum(*[branch1, branch2c], name=('%s/sum' % name))
    return summ


# normal
def InceptionFactoryA(data, num_1x1, num_3x3red, num_3x3, num_d3x3red, num_d3x3, num_out, name, prefix='', dilate=(1, 1)):
    # 1x1
    c1x1 = ConvFactory(data=data, num_filter=num_1x1, kernel=(1, 1), name=('%s/incep/0' % name), prefix=prefix)
    # 3x3 reduce + 3x3
    c3x3r = ConvFactory(data=data, num_filter=num_3x3red, kernel=(1, 1), name=('%s/incep/1_reduce' % name), prefix=prefix)
    c3x3 = ConvFactory(data=c3x3r, num_filter=num_3x3, kernel=(3, 3), pad=dilate, dilate=dilate, name=('%s/incep/1_0' % name), prefix=prefix)
    # double 3x3 reduce + double 3x3
    cd3x3r = ConvFactory(data=data, num_filter=num_d3x3red, kernel=(1, 1), name=('%s/incep/2_reduce' % name), prefix=prefix)
    cd3x3 = ConvFactory(data=cd3x3r, num_filter=num_d3x3, kernel=(3, 3), pad=dilate, dilate=dilate, name=('%s/incep/2_0' % name), prefix=prefix)
    cd3x3 = ConvFactory(data=cd3x3, num_filter=num_d3x3, kernel=(3, 3), pad=dilate, dilate=dilate, name=('%s/incep/2_1' % name), prefix=prefix)
    # concat
    concat = mx.symbol.Concat(*[c1x1, c3x3, cd3x3], name='%s%s/incep' % (prefix, name))
    # out
    out = ConvFactory2(data=concat, num_filter=num_out, kernel=(1, 1), name=('%s/out' % name), prefix=prefix)
    # ssum
    summ = mx.symbol.ElementWiseSum(*[out, data], name=('%s/sum' % name))
    return summ


# downsample
def InceptionFactoryB(data, num_1x1, num_3x3red, num_3x3, num_d3x3red, num_d3x3, num_poolproj, num_out, name, prefix='', stride=(2, 2), dilate=(1, 1)):
    # 1x1
    c1x1 = ConvFactory(data=data, num_filter=num_1x1, kernel=(1, 1), stride=stride, name=('%s/incep/0' % name), prefix=prefix)
    # 3x3 reduce + 3x3
    c3x3r = ConvFactory(data=data, num_filter=num_3x3red, kernel=(1, 1), stride=stride, name=('%s/incep/1_reduce' % name), prefix=prefix)
    c3x3 = ConvFactory(data=c3x3r, num_filter=num_3x3, kernel=(3, 3), pad=dilate, dilate=dilate, name=('%s/incep/1_0' % name), prefix=prefix)
    # double 3x3 reduce + double 3x3
    cd3x3r = ConvFactory(data=data, num_filter=num_d3x3red, kernel=(1, 1), stride=stride, name=('%s/incep/2_reduce' % name), prefix=prefix)
    cd3x3 = ConvFactory(data=cd3x3r, num_filter=num_d3x3, kernel=(3, 3), pad=dilate, dilate=dilate, name=('%s/incep/2_0' % name), prefix=prefix)
    cd3x3 = ConvFactory(data=cd3x3, num_filter=num_d3x3, kernel=(3, 3), pad=dilate, dilate=dilate, name=('%s/incep/2_1' % name), prefix=prefix)
    # pool
    pooling = mx.symbol.Pooling(data=data, kernel=(3, 3), stride=stride, pad=(1, 1), pool_type="max", name=('%s%s/incep/pool' % (prefix, name)))
    poolproj = ConvFactory(data=pooling, num_filter=num_poolproj, kernel=(1, 1), name=('%s/incep/poolproj' % name), prefix=prefix)
    # concat
    concat = mx.symbol.Concat(*[c1x1, c3x3, cd3x3, poolproj], name='%s%s/incep' % (prefix, name))
    # out
    out = ConvFactory2(data=concat, num_filter=num_out, kernel=(1, 1), name=('%s/out' % name), prefix=prefix)
    # proj
    proj = ProjFactory(data=data, num_filter=num_out, kernel=(1, 1), stride=stride, name=('%s/proj' % name), prefix=prefix)
    # ssum
    summ = mx.symbol.ElementWiseSum(*[out, proj], name=('%s/sum' % name))
    return summ


def get_pva_conv(data):
    # conv1
    conv1_1 = CReLUFactory(data=data, num_filter=16, kernel=(7, 7), stride=(2, 2), pad=(3, 3), name='conv1_1')
    pool1_1 = mx.symbol.Pooling(data=conv1_1, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type="max", name='pool1')

    # conv2
    conv2_1 = KKCReLUFactory_o(pool1_1, 24, 24, 64, 'conv2_1')
    conv2_2 = KKCReLUFactory_x(conv2_1, 24, 24, 64, 'conv2_2')
    conv2_3 = KKCReLUFactory_x(conv2_2, 24, 24, 64, 'conv2_3')

    # conv3
    conv3_1 = KKCReLUFactory_d(conv2_3, 48, 48, 128, 'conv3_1')
    conv3_2 = KKCReLUFactory_x(conv3_1, 48, 48, 128, 'conv3_2')
    conv3_3 = KKCReLUFactory_x(conv3_2, 48, 48, 128, 'conv3_3')
    conv3_4 = KKCReLUFactory_x(conv3_3, 48, 48, 128, 'conv3_4')

    # conv4
    conv4_1 = InceptionFactoryB(conv3_4, 64, 48, 128, 24, 48, 128, 256, 'conv4_1')
    conv4_2 = InceptionFactoryA(conv4_1, 64, 64, 128, 24, 48, 256, 'conv4_2')
    conv4_3 = InceptionFactoryA(conv4_2, 64, 64, 128, 24, 48, 256, 'conv4_3')
    conv4_4 = InceptionFactoryA(conv4_3, 64, 64, 128, 24, 48, 256, 'conv4_4')

    # conv5
    conv5_1 = InceptionFactoryB(conv4_4, 64, 96, 192, 32, 64, 128, 384, 'conv5_1', stride=(1, 1))
    conv5_2 = InceptionFactoryA(conv5_1, 64, 96, 192, 32, 64, 384, 'conv5_2')
    conv5_3 = InceptionFactoryA(conv5_2, 64, 96, 192, 32, 64, 384, 'conv5_3')
    conv5_4 = InceptionFactoryA(conv5_3, 64, 96, 192, 32, 64, 384, 'conv5_4')

    return conv5_4


def get_pva_train(num_classes=config.NUM_CLASSES, num_anchors=config.NUM_ANCHORS):
    data = mx.symbol.Variable(name="data")
    im_info = mx.symbol.Variable(name="im_info")
    gt_boxes = mx.symbol.Variable(name="gt_boxes")
    rpn_label = mx.symbol.Variable(name='label')
    rpn_bbox_target = mx.symbol.Variable(name='bbox_target')
    rpn_bbox_weight = mx.symbol.Variable(name='bbox_weight')

    # shared convolutional layers
    conv_feat = get_pva_conv(data)

    # RPN layers
    rpn_conv = mx.symbol.Convolution(
        data=conv_feat, kernel=(3, 3), pad=(1, 1), num_filter=512, name="rpn_conv_3x3")
    rpn_relu = mx.symbol.Activation(data=rpn_conv, act_type="relu", name="rpn_relu")
    rpn_cls_score = mx.symbol.Convolution(
        data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=2 * num_anchors, name="rpn_cls_score")
    rpn_bbox_pred = mx.symbol.Convolution(
        data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=4 * num_anchors, name="rpn_bbox_pred")

    # prepare rpn data
    rpn_cls_score_reshape = mx.symbol.Reshape(
        data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape")

    # classification
    rpn_cls_prob = mx.symbol.SoftmaxOutput(data=rpn_cls_score_reshape, label=rpn_label, multi_output=True,
                                           normalization='valid', use_ignore=True, ignore_label=-1, name="rpn_cls_prob")
    # bounding box regression
    rpn_bbox_loss_ = rpn_bbox_weight * mx.symbol.smooth_l1(name='rpn_bbox_loss_', scalar=3.0, data=(rpn_bbox_pred - rpn_bbox_target))
    rpn_bbox_loss = mx.sym.MakeLoss(name='rpn_bbox_loss', data=rpn_bbox_loss_, grad_scale=1.0 / config.TRAIN.RPN_BATCH_SIZE)

    # ROI proposal
    rpn_cls_act = mx.symbol.SoftmaxActivation(
        data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_act")
    rpn_cls_act_reshape = mx.symbol.Reshape(
        data=rpn_cls_act, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_act_reshape')
    rois = mx.symbol.Custom(
        cls_prob=rpn_cls_act_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
        op_type='proposal', feat_stride=config.RPN_FEAT_STRIDE,
        scales=tuple(config.ANCHOR_SCALES), ratios=tuple(config.ANCHOR_RATIOS),
        rpn_pre_nms_top_n=config.TRAIN.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=config.TRAIN.RPN_POST_NMS_TOP_N,
        threshold=config.TRAIN.RPN_NMS_THRESH, rpn_min_size=config.TRAIN.RPN_MIN_SIZE)

    # ROI proposal target
    gt_boxes_reshape = mx.symbol.Reshape(data=gt_boxes, shape=(-1, 5), name='gt_boxes_reshape')
    group = mx.symbol.Custom(rois=rois, gt_boxes=gt_boxes_reshape, op_type='proposal_target',
                             num_classes=num_classes, batch_images=config.TRAIN.BATCH_IMAGES,
                             batch_rois=config.TRAIN.BATCH_ROIS, fg_fraction=config.TRAIN.FG_FRACTION)
    rois = group[0]
    label = group[1]
    bbox_target = group[2]
    bbox_weight = group[3]

    # Fast R-CNN
    pool5 = mx.symbol.ROIPooling(
        name='roi_pool5', data=conv_feat, rois=rois, pooled_size=(7, 7), spatial_scale=1.0 / config.RCNN_FEAT_SRTIDE)

    # fc6
    flatten = mx.symbol.Flatten(data=pool5, name="flatten")
    fc6 = mx.symbol.FullyConnected(data=flatten, num_hidden=4096, name="fc6")
    bn6 = mx.symbol.BatchNorm(data=fc6, name="fc6/bn", fix_gamma=False, eps=eps, use_global_stats=use_global_stats)
    drop6 = mx.symbol.Dropout(data=bn6, p=0.5, name="drop6")
    relu6 = mx.symbol.Activation(data=drop6, act_type="relu", name="relu6")

    # fc7
    fc7 = mx.symbol.FullyConnected(data=relu6, num_hidden=4096, name="fc7")
    bn7 = mx.symbol.BatchNorm(data=fc7, name="fc7/bn", fix_gamma=False, eps=eps, use_global_stats=use_global_stats)
    drop7 = mx.symbol.Dropout(data=bn7, p=0.5, name="drop7")
    relu7 = mx.symbol.Activation(data=drop7, act_type="relu", name="relu7")

    # classification
    cls_score = mx.symbol.FullyConnected(name='cls_score', data=relu7, num_hidden=num_classes)
    cls_prob = mx.symbol.SoftmaxOutput(name='cls_prob', data=cls_score, label=label, normalization='batch')
    # bounding box regression
    bbox_pred = mx.symbol.FullyConnected(name='bbox_pred', data=relu7, num_hidden=num_classes * 4)
    bbox_loss_ = bbox_weight * mx.symbol.smooth_l1(name='bbox_loss_', scalar=1.0, data=(bbox_pred - bbox_target))
    bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_, grad_scale=1.0 / config.TRAIN.BATCH_ROIS)

    # reshape output
    label = mx.symbol.Reshape(data=label, shape=(config.TRAIN.BATCH_IMAGES, -1), name='label_reshape')
    cls_prob = mx.symbol.Reshape(data=cls_prob, shape=(config.TRAIN.BATCH_IMAGES, -1, num_classes), name='cls_prob_reshape')
    bbox_loss = mx.symbol.Reshape(data=bbox_loss, shape=(config.TRAIN.BATCH_IMAGES, -1, 4 * num_classes), name='bbox_loss_reshape')

    group = mx.symbol.Group([rpn_cls_prob, rpn_bbox_loss, cls_prob, bbox_loss, mx.symbol.BlockGrad(label)])
    return group


def get_pva_test(num_classes=config.NUM_CLASSES, num_anchors=config.NUM_ANCHORS):
    data = mx.symbol.Variable(name="data")
    im_info = mx.symbol.Variable(name="im_info")

    # shared convolutional layers
    conv_feat = get_pva_conv(data)

    # RPN
    rpn_conv = mx.symbol.Convolution(
        data=conv_feat, kernel=(3, 3), pad=(1, 1), num_filter=512, name="rpn_conv_3x3")
    rpn_relu = mx.symbol.Activation(data=rpn_conv, act_type="relu", name="rpn_relu")
    rpn_cls_score = mx.symbol.Convolution(
        data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=2 * num_anchors, name="rpn_cls_score")
    rpn_bbox_pred = mx.symbol.Convolution(
        data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=4 * num_anchors, name="rpn_bbox_pred")

    # ROI Proposal
    rpn_cls_score_reshape = mx.symbol.Reshape(
        data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape")
    rpn_cls_prob = mx.symbol.SoftmaxActivation(
        data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_prob")
    rpn_cls_prob_reshape = mx.symbol.Reshape(
        data=rpn_cls_prob, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_prob_reshape')
    rois = mx.symbol.Custom(
        cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
        op_type='proposal', feat_stride=config.RPN_FEAT_STRIDE,
        scales=tuple(config.ANCHOR_SCALES), ratios=tuple(config.ANCHOR_RATIOS),
        rpn_pre_nms_top_n=config.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=config.TEST.RPN_POST_NMS_TOP_N,
        threshold=config.TEST.RPN_NMS_THRESH, rpn_min_size=config.TEST.RPN_MIN_SIZE)

    # Fast R-CNN
    pool5 = mx.symbol.ROIPooling(
        name='roi_pool5', data=conv_feat, rois=rois, pooled_size=(7, 7), spatial_scale=1.0 / config.RCNN_FEAT_SRTIDE)

    # fc6
    flatten = mx.symbol.Flatten(data=pool5, name="flatten")
    fc6 = mx.symbol.FullyConnected(data=flatten, num_hidden=4096, name="fc6")
    bn6 = mx.symbol.BatchNorm(data=fc6, name="fc6/bn", fix_gamma=False, eps=eps, use_global_stats=use_global_stats)
    drop6 = mx.symbol.Dropout(data=bn6, p=0.5, name="drop6")
    relu6 = mx.symbol.Activation(data=drop6, act_type="relu", name="relu6")

    # fc7
    fc7 = mx.symbol.FullyConnected(data=relu6, num_hidden=4096, name="fc7")
    bn7 = mx.symbol.BatchNorm(data=fc7, name="fc7/bn", fix_gamma=False, eps=eps, use_global_stats=use_global_stats)
    drop7 = mx.symbol.Dropout(data=bn7, p=0.5, name="drop7")
    relu7 = mx.symbol.Activation(data=drop7, act_type="relu", name="relu7")

    # classification
    cls_score = mx.symbol.FullyConnected(name='cls_score', data=relu7, num_hidden=num_classes)
    cls_prob = mx.symbol.SoftmaxOutput(name='cls_prob', data=cls_score)
    # bounding box regression
    bbox_pred = mx.symbol.FullyConnected(name='bbox_pred', data=relu7, num_hidden=num_classes * 4)

    # reshape output
    cls_prob = mx.symbol.Reshape(data=cls_prob, shape=(config.TEST.BATCH_IMAGES, -1, num_classes), name='cls_prob_reshape')
    bbox_pred = mx.symbol.Reshape(data=bbox_pred, shape=(config.TEST.BATCH_IMAGES, -1, 4 * num_classes), name='bbox_pred_reshape')

    # group output
    group = mx.symbol.Group([rois, cls_prob, bbox_pred])
    return group


def get_pva_rpn_test(num_anchors=config.NUM_ANCHORS):
    data = mx.symbol.Variable(name="data")
    im_info = mx.symbol.Variable(name="im_info")

    # shared convolutional layers
    conv_feat = get_pva_conv(data)

    # RPN
    rpn_conv = mx.symbol.Convolution(
        data=conv_feat, kernel=(3, 3), pad=(1, 1), num_filter=512, name="rpn_conv_3x3")
    rpn_relu = mx.symbol.Activation(data=rpn_conv, act_type="relu", name="rpn_relu")
    rpn_cls_score = mx.symbol.Convolution(
        data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=2 * num_anchors, name="rpn_cls_score")
    rpn_bbox_pred = mx.symbol.Convolution(
        data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=4 * num_anchors, name="rpn_bbox_pred")

    # ROI Proposal
    rpn_cls_score_reshape = mx.symbol.Reshape(
        data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape")
    rpn_cls_prob = mx.symbol.SoftmaxActivation(
        data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_prob")
    rpn_cls_prob_reshape = mx.symbol.Reshape(
        data=rpn_cls_prob, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_prob_reshape')
    if config.TEST.CXX_PROPOSAL:
        group = mx.symbol.Proposal(
            cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
            feature_stride=config.RPN_FEAT_STRIDE, scales=tuple(config.ANCHOR_SCALES), ratios=tuple(config.ANCHOR_RATIOS),
            rpn_pre_nms_top_n=config.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=config.TEST.RPN_POST_NMS_TOP_N,
            threshold=config.TEST.RPN_NMS_THRESH, rpn_min_size=config.TEST.RPN_MIN_SIZE, output_score=True)
    else:
        group = mx.symbol.Custom(
            cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois', output_score=True,
            op_type='proposal', feat_stride=config.RPN_FEAT_STRIDE,
            scales=tuple(config.ANCHOR_SCALES), ratios=tuple(config.ANCHOR_RATIOS),
            rpn_pre_nms_top_n=config.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=config.TEST.RPN_POST_NMS_TOP_N,
            threshold=config.TEST.RPN_NMS_THRESH, rpn_min_size=config.TEST.RPN_MIN_SIZE)
    # rois = group[0]
    # score = group[1]

    return group
