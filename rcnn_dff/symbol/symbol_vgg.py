import mxnet as mx
from collections import OrderedDict
import proposal
import proposal_target
import rcnn_iou_loss
import rpn_iou_loss
import sample_anchors
import sample_rois
from ..config import config
from symbol_flow import conv_unit, stereo_scale_net, feature_warp, feature_propagate, feature_propagate_share

def get_vgg_conv(data):
    """
    shared convolutional layers
    :param data: Symbol
    :return: Symbol
    """
    # group 1
    conv1_1 = mx.symbol.Convolution(
        data=data, kernel=(3, 3), pad=(1, 1), num_filter=64, workspace=2048, name="conv1_1")
    relu1_1 = mx.symbol.Activation(data=conv1_1, act_type="relu", name="relu1_1")
    conv1_2 = mx.symbol.Convolution(
        data=relu1_1, kernel=(3, 3), pad=(1, 1), num_filter=64, workspace=2048, name="conv1_2")
    relu1_2 = mx.symbol.Activation(data=conv1_2, act_type="relu", name="relu1_2")
    pool1 = mx.symbol.Pooling(
        data=relu1_2, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool1")
    # group 2
    conv2_1 = mx.symbol.Convolution(
        data=pool1, kernel=(3, 3), pad=(1, 1), num_filter=128, workspace=2048, name="conv2_1")
    relu2_1 = mx.symbol.Activation(data=conv2_1, act_type="relu", name="relu2_1")
    conv2_2 = mx.symbol.Convolution(
        data=relu2_1, kernel=(3, 3), pad=(1, 1), num_filter=128, workspace=2048, name="conv2_2")
    relu2_2 = mx.symbol.Activation(data=conv2_2, act_type="relu", name="relu2_2")
    pool2 = mx.symbol.Pooling(
        data=relu2_2, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool2")
    # group 3
    conv3_1 = mx.symbol.Convolution(
        data=pool2, kernel=(3, 3), pad=(1, 1), num_filter=256, workspace=2048, name="conv3_1")
    relu3_1 = mx.symbol.Activation(data=conv3_1, act_type="relu", name="relu3_1")
    conv3_2 = mx.symbol.Convolution(
        data=relu3_1, kernel=(3, 3), pad=(1, 1), num_filter=256, workspace=2048, name="conv3_2")
    relu3_2 = mx.symbol.Activation(data=conv3_2, act_type="relu", name="relu3_2")
    conv3_3 = mx.symbol.Convolution(
        data=relu3_2, kernel=(3, 3), pad=(1, 1), num_filter=256, workspace=2048, name="conv3_3")
    relu3_3 = mx.symbol.Activation(data=conv3_3, act_type="relu", name="relu3_3")
    pool3 = mx.symbol.Pooling(
        data=relu3_3, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool3")
    # group 4
    conv4_1 = mx.symbol.Convolution(
        data=pool3, kernel=(3, 3), pad=(1, 1), num_filter=512, workspace=2048, name="conv4_1")
    relu4_1 = mx.symbol.Activation(data=conv4_1, act_type="relu", name="relu4_1")
    conv4_2 = mx.symbol.Convolution(
        data=relu4_1, kernel=(3, 3), pad=(1, 1), num_filter=512, workspace=2048, name="conv4_2")
    relu4_2 = mx.symbol.Activation(data=conv4_2, act_type="relu", name="relu4_2")
    conv4_3 = mx.symbol.Convolution(
        data=relu4_2, kernel=(3, 3), pad=(1, 1), num_filter=512, workspace=2048, name="conv4_3")
    relu4_3 = mx.symbol.Activation(data=conv4_3, act_type="relu", name="relu4_3")
    pool4 = mx.symbol.Pooling(
        data=relu4_3, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool4")
    # group 5
    conv5_1 = mx.symbol.Convolution(
        data=pool4, kernel=(3, 3), pad=(1, 1), num_filter=512, workspace=2048, name="conv5_1")
    relu5_1 = mx.symbol.Activation(data=conv5_1, act_type="relu", name="relu5_1")
    conv5_2 = mx.symbol.Convolution(
        data=relu5_1, kernel=(3, 3), pad=(1, 1), num_filter=512, workspace=2048, name="conv5_2")
    relu5_2 = mx.symbol.Activation(data=conv5_2, act_type="relu", name="relu5_2")
    conv5_3 = mx.symbol.Convolution(
        data=relu5_2, kernel=(3, 3), pad=(1, 1), num_filter=512, workspace=2048, name="conv5_3")
    relu5_3 = mx.symbol.Activation(data=conv5_3, act_type="relu", name="relu5_3")

    return relu5_3

def get_vgg_rcnn(num_classes=config.NUM_CLASSES):
    """
    Fast R-CNN with VGG 16 conv layers
    :param num_classes: used to determine output size
    :return: Symbol
    """
    data = mx.symbol.Variable(name="data")
    rois = mx.symbol.Variable(name='rois')
    label = mx.symbol.Variable(name='label')
    bbox_target = mx.symbol.Variable(name='bbox_target')
    bbox_weight = mx.symbol.Variable(name='bbox_weight')

    # reshape input
    rois = mx.symbol.Reshape(data=rois, shape=(-1, 5), name='rois_reshape')
    label = mx.symbol.Reshape(data=label, shape=(-1, ), name='label_reshape')
    bbox_target = mx.symbol.Reshape(data=bbox_target, shape=(-1, 4 * num_classes), name='bbox_target_reshape')
    bbox_weight = mx.symbol.Reshape(data=bbox_weight, shape=(-1, 4 * num_classes), name='bbox_weight_reshape')

    # shared convolutional layers
    relu5_3 = get_vgg_conv(data)

    # Fast R-CNN
    roi_pool = mx.symbol.ROIPooling(
        name='roi_pool', data=relu5_3, rois=rois, pooled_size=(7, 7), spatial_scale=1.0 / config.RCNN_FEAT_STRIDE)
    if config.RCNN_CTX_WINDOW:
        roi_pool_ctx = mx.symbol.ROIPooling(
            name='roi_pool_ctx', data=relu5_3, rois=rois, pooled_size=(7, 7),
            spatial_scale=1.0 / config.RCNN_FEAT_STRIDE, pad=0.25)
        roi_pool_concat = mx.symbol.Concat(roi_pool, roi_pool_ctx, name='roi_pool_concat')
        roi_pool_red = mx.symbol.Convolution(
            data=roi_pool_concat, num_filter=512, kernel=(1, 1), stride=(1, 1), name='roi_pool_ctx_red')
        roi_pool = mx.symbol.Activation(data=roi_pool_red, act_type='relu', name='roi_pool_relu')

    # group 6
    flatten = mx.symbol.Flatten(data=roi_pool, name="flatten")
    fc6 = mx.symbol.FullyConnected(data=flatten, num_hidden=4096, name="fc6")
    relu6 = mx.symbol.Activation(data=fc6, act_type="relu", name="relu6")
    drop6 = mx.symbol.Dropout(data=relu6, p=0.5, name="drop6")
    # group 7
    fc7 = mx.symbol.FullyConnected(data=drop6, num_hidden=4096, name="fc7")
    relu7 = mx.symbol.Activation(data=fc7, act_type="relu", name="relu7")
    drop7 = mx.symbol.Dropout(data=relu7, p=0.5, name="drop7")

    # classification
    cls_score = mx.symbol.FullyConnected(name='cls_score', data=drop7, num_hidden=num_classes)
    cls_prob = mx.symbol.SoftmaxOutput(name='cls_prob', data=cls_score, label=label, normalization='batch',
                                       is_hidden_layer=config.TRAIN.RCNN_OHEM)

    # bounding box regression
    bbox_pred = mx.symbol.FullyConnected(name='bbox_pred', data=drop7, num_hidden=num_classes * 4)
    if config.RCNN_IOU_LOSS:
        bbox_loss_ = mx.symbol.Custom(data=bbox_pred, bbox_target=bbox_target, bbox_weight=bbox_weight, rois=rois,
                                      op_type='rcnn_iou_loss', name='bbox_loss_', num_classes=num_classes)
    else:
        bbox_loss_ = bbox_weight * mx.symbol.smooth_l1(name='bbox_loss_', scalar=1.0, data=(bbox_pred - bbox_target))

    if config.TRAIN.RCNN_OHEM:
        group = mx.symbol.Custom(
            cls_prob=cls_prob, bbox_loss=bbox_loss_, label=label, rois=rois, bbox_pred=bbox_pred,
            name='rcnn_ohem', op_type='sample_rois',
            batch_images=config.TRAIN.BATCH_IMAGES, batch_size=config.TRAIN.BATCH_ROIS,
            nms_threshold=config.TRAIN.RCNN_OHEM_NMS, iou_loss=config.RCNN_IOU_LOSS,
            transform=config.TRAIN.RCNN_OHEM_TRANSFORM, ignore=config.TRAIN.RCNN_OHEM_IGNORE)
        rcnn_group = [group[0], group[1], group[2], group[3]]
        for ind, name, last_shape in zip(range(len(rcnn_group)),
                                         ['cls_prob', 'bbox_loss', 'cls_mask', 'bbox_mask'],
                                         [num_classes, num_classes * 4, num_classes, num_classes * 4]):
            rcnn_group[ind] = mx.symbol.Reshape(data=rcnn_group[ind], shape=(config.TRAIN.BATCH_IMAGES, -1, last_shape),
                                                name=name + '_reshape')
    else:
        bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_, grad_scale=1.0 / config.TRAIN.BATCH_ROIS)
        rcnn_group = [cls_prob, bbox_loss]
        for ind, name, last_shape in zip(range(len(rcnn_group)),
                                         ['cls_prob', 'bbox_loss'],
                                         [num_classes, num_classes * 4]):
            rcnn_group[ind] = mx.symbol.Reshape(data=rcnn_group[ind], shape=(config.TRAIN.BATCH_IMAGES, -1, last_shape),
                                                name=name + '_reshape')

    # group output
    group = mx.symbol.Group(rcnn_group)
    return group


def get_vgg_rcnn_test(num_classes=config.NUM_CLASSES):
    """
    Fast R-CNN Network with VGG
    :param num_classes: used to determine output size
    :return: Symbol
    """
    data = mx.symbol.Variable(name="data")
    rois = mx.symbol.Variable(name='rois')

    # reshape rois
    rois = mx.symbol.Reshape(data=rois, shape=(-1, 5), name='rois_reshape')

    # shared convolutional layer
    relu5_3 = get_vgg_conv(data)

    # Fast R-CNN
    roi_pool = mx.symbol.ROIPooling(
        name='roi_pool', data=relu5_3, rois=rois, pooled_size=(7, 7), spatial_scale=1.0 / config.RCNN_FEAT_STRIDE)
    if config.RCNN_CTX_WINDOW:
        roi_pool_ctx = mx.symbol.ROIPooling(
            name='roi_pool_ctx', data=relu5_3, rois=rois, pooled_size=(7, 7),
            spatial_scale=1.0 / config.RCNN_FEAT_STRIDE, pad=0.25)
        roi_pool_concat = mx.symbol.Concat(roi_pool, roi_pool_ctx, name='roi_pool_concat')
        roi_pool_red = mx.symbol.Convolution(
            data=roi_pool_concat, num_filter=512, kernel=(1, 1), stride=(1, 1), name='roi_pool_ctx_red')
        roi_pool = mx.symbol.Activation(data=roi_pool_red, act_type='relu', name='roi_pool_relu')

    # group 6
    flatten = mx.symbol.Flatten(data=roi_pool, name="flatten")
    fc6 = mx.symbol.FullyConnected(data=flatten, num_hidden=4096, name="fc6")
    relu6 = mx.symbol.Activation(data=fc6, act_type="relu", name="relu6")
    drop6 = mx.symbol.Dropout(data=relu6, p=0.5, name="drop6")
    # group 7
    fc7 = mx.symbol.FullyConnected(data=drop6, num_hidden=4096, name="fc7")
    relu7 = mx.symbol.Activation(data=fc7, act_type="relu", name="relu7")
    drop7 = mx.symbol.Dropout(data=relu7, p=0.5, name="drop7")
    # classification
    cls_score = mx.symbol.FullyConnected(name='cls_score', data=drop7, num_hidden=num_classes)
    cls_prob = mx.symbol.SoftmaxOutput(name='cls_prob', data=cls_score)
    # bounding box regression
    bbox_pred = mx.symbol.FullyConnected(name='bbox_pred', data=drop7, num_hidden=num_classes * 4)

    # reshape output
    cls_prob = mx.symbol.Reshape(data=cls_prob, shape=(config.TEST.BATCH_IMAGES, -1, num_classes), name='cls_prob_reshape')
    bbox_pred = mx.symbol.Reshape(data=bbox_pred, shape=(config.TEST.BATCH_IMAGES, -1, 4 * num_classes), name='bbox_pred_reshape')

    # group output
    group = mx.symbol.Group([cls_prob, bbox_pred])
    return group


def get_vgg_rpn(num_anchors=config.NUM_ANCHORS):
    """
    Region Proposal Network with VGG
    :param num_anchors: used to determine output size
    :return: Symbol
    """
    data = mx.symbol.Variable(name="data")
    label = mx.symbol.Variable(name='label')
    bbox_target = mx.symbol.Variable(name='bbox_target')
    bbox_weight = mx.symbol.Variable(name='bbox_weight')

    # shared convolutional layers
    relu5_3 = get_vgg_conv(data)

    # RPN
    rpn_conv = mx.symbol.Convolution(
        data=relu5_3, kernel=(3, 3), pad=(1, 1), num_filter=512, name="rpn_conv_3x3")
    rpn_relu = mx.symbol.Activation(data=rpn_conv, act_type="relu", name="rpn_relu")
    rpn_cls_score = mx.symbol.Convolution(
        data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=2 * num_anchors, name="rpn_cls_score")
    rpn_bbox_pred = mx.symbol.Convolution(
        data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=4 * num_anchors, name="rpn_bbox_pred")

    # prepare rpn data
    rpn_cls_score_reshape = mx.symbol.Reshape(
        data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape")

    # classification
    cls_prob = mx.symbol.SoftmaxOutput(data=rpn_cls_score_reshape, label=label, multi_output=True,
                                       normalization='valid', use_ignore=True, ignore_label=-1,
                                       is_hidden_layer=config.TRAIN.RPN_OHEM, name="cls_prob")
    # bounding box regression
    if config.RPN_IOU_LOSS:
        bbox_loss_ = mx.symbol.Custom(data=rpn_bbox_pred, bbox_target=bbox_target, bbox_weight=bbox_weight,
                                      op_type='rpn_iou_loss', name='bbox_loss_',
                                      feature_stride=config.RPN_FEAT_STRIDE, scales=tuple(config.ANCHOR_SCALES),
                                      ratios=tuple(config.ANCHOR_RATIOS))
    else:
        bbox_loss_ = bbox_weight * mx.symbol.smooth_l1(name='bbox_loss_', scalar=3.0, data=(rpn_bbox_pred - bbox_target))

    if config.TRAIN.RPN_OHEM:
        group = mx.symbol.Custom(
            cls_prob=cls_prob, bbox_loss=bbox_loss_, label=label, bbox_pred=rpn_bbox_pred,
            name='rpn_ohem', op_type='sample_anchors',
            feature_stride=config.RPN_FEAT_STRIDE, scales=tuple(config.ANCHOR_SCALES), ratios=tuple(config.ANCHOR_RATIOS),
            rpn_pre_nms_top_n=config.TRAIN.RPN_OHEM_ANCHORS, rpn_batch_size=config.TRAIN.RPN_BATCH_SIZE,
            nms_threshold=config.TRAIN.RPN_OHEM_NMS, iou_loss=config.RPN_IOU_LOSS,
            transform=config.TRAIN.RPN_OHEM_TRANSFORM, ignore=config.TRAIN.RPN_OHEM_IGNORE, np_ratio=config.TRAIN.RPN_OHEM_NP_RATIO)
        rpn_group = [group[0], group[1], group[2], group[3]]
    else:
        bbox_loss = mx.sym.MakeLoss(name='rpn_bbox_loss', data=bbox_loss_, grad_scale=1.0 / config.TRAIN.RPN_BATCH_SIZE)
        rpn_group = [cls_prob, bbox_loss]

    # group output
    group = mx.symbol.Group(rpn_group)
    return group


def get_vgg_rpn_test(num_anchors=config.NUM_ANCHORS):
    """
    Region Proposal Network with VGG
    :param num_anchors: used to determine output size
    :return: Symbol
    """
    data = mx.symbol.Variable(name="data")
    im_info = mx.symbol.Variable(name="im_info")

    # shared convolutional layers
    relu5_3 = get_vgg_conv(data)

    # RPN
    rpn_conv = mx.symbol.Convolution(
        data=relu5_3, kernel=(3, 3), pad=(1, 1), num_filter=512, name="rpn_conv_3x3")
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
            cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois', output_score=True,
            feature_stride=config.RPN_FEAT_STRIDE, scales=tuple(config.ANCHOR_SCALES), ratios=tuple(config.ANCHOR_RATIOS),
            rpn_pre_nms_top_n=config.TEST.PROPOSAL_PRE_NMS_TOP_N, rpn_post_nms_top_n=config.TEST.PROPOSAL_POST_NMS_TOP_N,
            threshold=config.TEST.PROPOSAL_NMS_THRESH, rpn_min_size=config.TEST.PROPOSAL_MIN_SIZE, iou_loss=config.RPN_IOU_LOSS)
    else:
        group = mx.symbol.Custom(
            cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois', output_score=True,
            op_type='proposal', feat_stride=config.RPN_FEAT_STRIDE,
            scales=tuple(config.ANCHOR_SCALES), ratios=tuple(config.ANCHOR_RATIOS),
            rpn_pre_nms_top_n=config.TEST.PROPOSAL_PRE_NMS_TOP_N, rpn_post_nms_top_n=config.TEST.PROPOSAL_POST_NMS_TOP_N,
            threshold=config.TEST.PROPOSAL_NMS_THRESH, rpn_min_size=config.TEST.PROPOSAL_MIN_SIZE, iou_loss=config.RPN_IOU_LOSS)
    # rois = group[0]
    # score = group[1]

    return group


def get_vgg_test(num_classes=config.NUM_CLASSES, num_anchors=config.NUM_ANCHORS):
    """
    Faster R-CNN test with VGG 16 conv layers
    :param num_classes: used to determine output size
    :param num_anchors: used to determine output size
    :return: Symbol
    """
    data = mx.symbol.Variable(name="data")
    im_info = mx.symbol.Variable(name="im_info")

    # shared convolutional layers
    relu5_3 = get_vgg_conv(data)

    # RPN
    rpn_conv = mx.symbol.Convolution(
        data=relu5_3, kernel=(3, 3), pad=(1, 1), num_filter=512, name="rpn_conv_3x3")
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
        rois = mx.symbol.Proposal(
            cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
            feature_stride=config.RPN_FEAT_STRIDE, scales=tuple(config.ANCHOR_SCALES), ratios=tuple(config.ANCHOR_RATIOS),
            rpn_pre_nms_top_n=config.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=config.TEST.RPN_POST_NMS_TOP_N,
            threshold=config.TEST.RPN_NMS_THRESH, rpn_min_size=config.TEST.RPN_MIN_SIZE, iou_loss=config.RPN_IOU_LOSS)
    else:
        rois = mx.symbol.Custom(
            cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
            op_type='proposal', feat_stride=config.RPN_FEAT_STRIDE,
            scales=tuple(config.ANCHOR_SCALES), ratios=tuple(config.ANCHOR_RATIOS),
            rpn_pre_nms_top_n=config.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=config.TEST.RPN_POST_NMS_TOP_N,
            threshold=config.TEST.RPN_NMS_THRESH, rpn_min_size=config.TEST.RPN_MIN_SIZE, iou_loss=config.RPN_IOU_LOSS)

    # Fast R-CNN
    roi_pool = mx.symbol.ROIPooling(
        name='roi_pool', data=relu5_3, rois=rois, pooled_size=(7, 7), spatial_scale=1.0 / config.RCNN_FEAT_STRIDE)
    if config.RCNN_CTX_WINDOW:
        roi_pool_ctx = mx.symbol.ROIPooling(
            name='roi_pool_ctx', data=relu5_3, rois=rois, pooled_size=(7, 7),
            spatial_scale=1.0 / config.RCNN_FEAT_STRIDE, pad=0.25)
        roi_pool_concat = mx.symbol.Concat(roi_pool, roi_pool_ctx, name='roi_pool_concat')
        roi_pool_red = mx.symbol.Convolution(
            data=roi_pool_concat, num_filter=512, kernel=(1, 1), stride=(1, 1), name='roi_pool_ctx_red')
        roi_pool = mx.symbol.Activation(data=roi_pool_red, act_type='relu', name='roi_pool_relu')

    # group 6
    flatten = mx.symbol.Flatten(data=roi_pool, name="flatten")
    fc6 = mx.symbol.FullyConnected(data=flatten, num_hidden=4096, name="fc6")
    relu6 = mx.symbol.Activation(data=fc6, act_type="relu", name="relu6")
    drop6 = mx.symbol.Dropout(data=relu6, p=0.5, name="drop6")
    # group 7
    fc7 = mx.symbol.FullyConnected(data=drop6, num_hidden=4096, name="fc7")
    relu7 = mx.symbol.Activation(data=fc7, act_type="relu", name="relu7")
    drop7 = mx.symbol.Dropout(data=relu7, p=0.5, name="drop7")
    # classification
    cls_score = mx.symbol.FullyConnected(name='cls_score', data=drop7, num_hidden=num_classes)
    cls_prob = mx.symbol.SoftmaxOutput(name='cls_prob', data=cls_score)
    # bounding box regression
    bbox_pred = mx.symbol.FullyConnected(name='bbox_pred', data=drop7, num_hidden=num_classes * 4)

    # reshape output
    cls_prob = mx.symbol.Reshape(data=cls_prob, shape=(config.TEST.BATCH_IMAGES, -1, num_classes), name='cls_prob_reshape')
    bbox_pred = mx.symbol.Reshape(data=bbox_pred, shape=(config.TEST.BATCH_IMAGES, -1, 4 * num_classes), name='bbox_pred_reshape')

    # group output
    group = mx.symbol.Group([rois, cls_prob, bbox_pred])
    return group


def get_vgg_train(num_classes=config.NUM_CLASSES, num_anchors=config.NUM_ANCHORS):
    """
    Faster R-CNN end-to-end with VGG 16 conv layers
    :param num_classes: used to determine output size
    :param num_anchors: used to determine output size
    :return: Symbol
    """
    data = mx.symbol.Variable(name="data")
    im_info = mx.symbol.Variable(name="im_info")
    gt_boxes = mx.symbol.Variable(name="gt_boxes")
    rpn_label = mx.symbol.Variable(name='label')
    rpn_bbox_target = mx.symbol.Variable(name='bbox_target')
    rpn_bbox_weight = mx.symbol.Variable(name='bbox_weight')

    # shared convolutional layers
    relu5_3 = get_vgg_conv(data)

    # RPN layers
    rpn_conv = mx.symbol.Convolution(
        data=relu5_3, kernel=(3, 3), pad=(1, 1), num_filter=512, name="rpn_conv_3x3")
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
                                           normalization='valid', use_ignore=True, ignore_label=-1,
                                           is_hidden_layer=config.TRAIN.RPN_OHEM, name="rpn_cls_prob")
    # bounding box regression
    if config.RPN_IOU_LOSS:
        rpn_bbox_loss_ = mx.symbol.Custom(data=rpn_bbox_pred, bbox_target=rpn_bbox_target, bbox_weight=rpn_bbox_weight,
                                          op_type='rpn_iou_loss', name='rpn_bbox_loss_',
                                          feature_stride=config.RPN_FEAT_STRIDE, scales=tuple(config.ANCHOR_SCALES),
                                          ratios=tuple(config.ANCHOR_RATIOS))
    else:
        rpn_bbox_loss_ = rpn_bbox_weight * mx.symbol.smooth_l1(name='rpn_bbox_loss_', scalar=3.0, data=(rpn_bbox_pred - rpn_bbox_target))

    # rpn output
    if config.TRAIN.RPN_OHEM:
        group = mx.symbol.Custom(
            cls_prob=rpn_cls_prob, bbox_loss=rpn_bbox_loss_, label=rpn_label, bbox_pred=rpn_bbox_pred,
            name='rpn_ohem', op_type='sample_anchors',
            feature_stride=config.RPN_FEAT_STRIDE, scales=tuple(config.ANCHOR_SCALES), ratios=tuple(config.ANCHOR_RATIOS),
            rpn_pre_nms_top_n=config.TRAIN.RPN_OHEM_ANCHORS, rpn_batch_size=config.TRAIN.RPN_BATCH_SIZE,
            nms_threshold=config.TRAIN.RPN_OHEM_NMS, iou_loss=config.RPN_IOU_LOSS,
            transform=config.TRAIN.RPN_OHEM_TRANSFORM, ignore=config.TRAIN.RPN_OHEM_IGNORE, np_ratio=config.TRAIN.RPN_OHEM_NP_RATIO)
        rpn_group = [group[0], group[1], group[2], group[3]]
    else:
        rpn_bbox_loss = mx.sym.MakeLoss(name='rpn_bbox_loss', data=rpn_bbox_loss_, grad_scale=1.0 / config.TRAIN.RPN_BATCH_SIZE)
        rpn_group = [rpn_cls_prob, rpn_bbox_loss]

    # ROI proposal
    rpn_cls_act = mx.symbol.SoftmaxActivation(
        data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_act")
    rpn_cls_act_reshape = mx.symbol.Reshape(
        data=rpn_cls_act, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_act_reshape')
    if config.TRAIN.CXX_PROPOSAL:
        rois = mx.symbol.Proposal(
            cls_prob=rpn_cls_act_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
            feature_stride=config.RPN_FEAT_STRIDE, scales=tuple(config.ANCHOR_SCALES), ratios=tuple(config.ANCHOR_RATIOS),
            rpn_pre_nms_top_n=config.TRAIN.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=config.TRAIN.RPN_POST_NMS_TOP_N,
            threshold=config.TRAIN.RPN_NMS_THRESH, rpn_min_size=config.TRAIN.RPN_MIN_SIZE, iou_loss=config.RPN_IOU_LOSS)
    else:
        rois = mx.symbol.Custom(
            cls_prob=rpn_cls_act_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
            op_type='proposal', feat_stride=config.RPN_FEAT_STRIDE,
            scales=tuple(config.ANCHOR_SCALES), ratios=tuple(config.ANCHOR_RATIOS),
            rpn_pre_nms_top_n=config.TRAIN.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=config.TRAIN.RPN_POST_NMS_TOP_N,
            threshold=config.TRAIN.RPN_NMS_THRESH, rpn_min_size=config.TRAIN.RPN_MIN_SIZE, iou_loss=config.RPN_IOU_LOSS)

    # ROI proposal target
    gt_boxes_reshape = mx.symbol.Reshape(data=gt_boxes, shape=(-1, 5), name='gt_boxes_reshape')
    if config.TRAIN.RCNN_OHEM:
        group = mx.symbol.Custom(rois=rois, gt_boxes=gt_boxes_reshape, op_type='proposal_target',
                                 num_classes=num_classes, batch_images=config.TRAIN.BATCH_IMAGES,
                                 batch_rois=config.TRAIN.RCNN_OHEM_ROIS, ohem=config.TRAIN.RCNN_OHEM)
    else:
        group = mx.symbol.Custom(rois=rois, gt_boxes=gt_boxes_reshape, op_type='proposal_target',
                                 num_classes=num_classes, batch_images=config.TRAIN.BATCH_IMAGES,
                                 batch_rois=config.TRAIN.BATCH_ROIS, fg_fraction=config.TRAIN.FG_FRACTION)
    rois = group[0]
    label = group[1]
    bbox_target = group[2]
    bbox_weight = group[3]

    # Fast R-CNN
    roi_pool = mx.symbol.ROIPooling(
        name='roi_pool', data=relu5_3, rois=rois, pooled_size=(7, 7), spatial_scale=1.0 / config.RCNN_FEAT_STRIDE)
    if config.RCNN_CTX_WINDOW:
        roi_pool_ctx = mx.symbol.ROIPooling(
            name='roi_pool_ctx', data=relu5_3, rois=rois, pooled_size=(7, 7),
            spatial_scale=1.0 / config.RCNN_FEAT_STRIDE, pad=0.25)
        roi_pool_concat = mx.symbol.Concat(roi_pool, roi_pool_ctx, name='roi_pool_concat')
        roi_pool_red = mx.symbol.Convolution(
            data=roi_pool_concat, num_filter=512, kernel=(1, 1), stride=(1, 1), name='roi_pool_ctx_red')
        roi_pool = mx.symbol.Activation(data=roi_pool_red, act_type='relu', name='roi_pool_relu')

    # group 6
    flatten = mx.symbol.Flatten(data=roi_pool, name="flatten")
    fc6 = mx.symbol.FullyConnected(data=flatten, num_hidden=4096, name="fc6")
    relu6 = mx.symbol.Activation(data=fc6, act_type="relu", name="relu6")
    drop6 = mx.symbol.Dropout(data=relu6, p=0.5, name="drop6")
    # group 7
    fc7 = mx.symbol.FullyConnected(data=drop6, num_hidden=4096, name="fc7")
    relu7 = mx.symbol.Activation(data=fc7, act_type="relu", name="relu7")
    drop7 = mx.symbol.Dropout(data=relu7, p=0.5, name="drop7")

    # classification
    cls_score = mx.symbol.FullyConnected(name='cls_score', data=drop7, num_hidden=num_classes)
    cls_prob = mx.symbol.SoftmaxOutput(name='cls_prob', data=cls_score, label=label, normalization='batch',
                                       is_hidden_layer=config.TRAIN.RCNN_OHEM)

    # bounding box regression
    bbox_pred = mx.symbol.FullyConnected(name='bbox_pred', data=drop7, num_hidden=num_classes * 4)
    if config.RCNN_IOU_LOSS:
        bbox_loss_ = mx.symbol.Custom(data=bbox_pred, bbox_target=bbox_target, bbox_weight=bbox_weight, rois=rois,
                                      op_type='rcnn_iou_loss', name='bbox_loss_', num_classes=num_classes)
    else:
        bbox_loss_ = bbox_weight * mx.symbol.smooth_l1(name='bbox_loss_', scalar=1.0, data=(bbox_pred - bbox_target))

    if config.TRAIN.RCNN_OHEM:
        group = mx.symbol.Custom(
            cls_prob=cls_prob, bbox_loss=bbox_loss_, label=label, rois=rois, bbox_pred=bbox_pred,
            name='rcnn_ohem', op_type='sample_rois',
            batch_images=config.TRAIN.BATCH_IMAGES, batch_size=config.TRAIN.BATCH_ROIS,
            nms_threshold=config.TRAIN.RCNN_OHEM_NMS, iou_loss=config.RPN_IOU_LOSS,
            transform=config.TRAIN.RCNN_OHEM_TRANSFORM, ignore=config.TRAIN.RCNN_OHEM_IGNORE)
        rcnn_group = [group[0], group[1], group[2], group[3]]
        for ind, name, last_shape in zip(range(len(rcnn_group)),
                                         ['cls_prob', 'bbox_loss', 'cls_mask', 'bbox_mask'],
                                         [num_classes, num_classes * 4, num_classes, num_classes * 4]):
            rcnn_group[ind] = mx.symbol.Reshape(data=rcnn_group[ind], shape=(config.TRAIN.BATCH_IMAGES, -1, last_shape),
                                                name=name + '_reshape')
    else:
        bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_, grad_scale=1.0 / config.TRAIN.BATCH_ROIS)
        rcnn_group = [cls_prob, bbox_loss]
        for ind, name, last_shape in zip(range(len(rcnn_group)),
                                         ['cls_prob', 'bbox_loss'],
                                         [num_classes, num_classes * 4]):
            rcnn_group[ind] = mx.symbol.Reshape(data=rcnn_group[ind], shape=(config.TRAIN.BATCH_IMAGES, -1, last_shape),
                                                name=name + '_reshape')

    # append label
    label = mx.symbol.Reshape(data=label, shape=(config.TRAIN.BATCH_IMAGES, -1), name='label_reshape')
    rcnn_group += [mx.symbol.BlockGrad(label, name='label_blockgrad')]

    group = mx.symbol.Group(rpn_group + rcnn_group)
    return group


def get_vgg_acc_test(num_classes=config.NUM_CLASSES, num_anchors=config.NUM_ANCHORS):
    """
    Faster R-CNN test with VGG 16 conv layers
    :param num_classes: used to determine output size
    :param num_anchors: used to determine output size
    :return: Symbol
    """
    data = mx.symbol.Variable(name="data")
    im_info = mx.symbol.Variable(name="im_info")

    # group 1
    conv1_1 = mx.symbol.Convolution(
        data=data, kernel=(3, 3), pad=(1, 1), num_filter=32, workspace=2048, name="conv1_1")
    relu1_1 = mx.symbol.Activation(data=conv1_1, act_type="relu", name="relu1_1")
    conv1_2 = mx.symbol.Convolution(
        data=relu1_1, kernel=(3, 3), pad=(1, 1), num_filter=32, workspace=2048, name="conv1_2")
    relu1_2 = mx.symbol.Activation(data=conv1_2, act_type="relu", name="relu1_2")
    pool1 = mx.symbol.Pooling(
        data=relu1_2, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool1")
    # group 2
    conv2_1 = mx.symbol.Convolution(
        data=pool1, kernel=(3, 3), pad=(1, 1), num_filter=64, workspace=2048, name="conv2_1")
    relu2_1 = mx.symbol.Activation(data=conv2_1, act_type="relu", name="relu2_1")
    conv2_2 = mx.symbol.Convolution(
        data=relu2_1, kernel=(3, 3), pad=(1, 1), num_filter=64, workspace=2048, name="conv2_2")
    relu2_2 = mx.symbol.Activation(data=conv2_2, act_type="relu", name="relu2_2")
    pool2 = mx.symbol.Pooling(
        data=relu2_2, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool2")
    # group 3
    conv3_1 = mx.symbol.Convolution(
        data=pool2, kernel=(3, 3), pad=(1, 1), num_filter=64, workspace=2048, name="conv3_1")
    relu3_1 = mx.symbol.Activation(data=conv3_1, act_type="relu", name="relu3_1")
    conv3_2 = mx.symbol.Convolution(
        data=relu3_1, kernel=(3, 3), pad=(1, 1), num_filter=64, workspace=2048, name="conv3_2")
    relu3_2 = mx.symbol.Activation(data=conv3_2, act_type="relu", name="relu3_2")
    conv3_3 = mx.symbol.Convolution(
        data=relu3_2, kernel=(3, 3), pad=(1, 1), num_filter=128, workspace=2048, name="conv3_3")
    relu3_3 = mx.symbol.Activation(data=conv3_3, act_type="relu", name="relu3_3")
    pool3 = mx.symbol.Pooling(
        data=relu3_3, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool3")
    # group 4
    conv4_1 = mx.symbol.Convolution(
        data=pool3, kernel=(3, 3), pad=(1, 1), num_filter=128, workspace=2048, name="conv4_1")
    relu4_1 = mx.symbol.Activation(data=conv4_1, act_type="relu", name="relu4_1")
    conv4_2 = mx.symbol.Convolution(
        data=relu4_1, kernel=(3, 3), pad=(1, 1), num_filter=128, workspace=2048, name="conv4_2")
    relu4_2 = mx.symbol.Activation(data=conv4_2, act_type="relu", name="relu4_2")
    conv4_3 = mx.symbol.Convolution(
        data=relu4_2, kernel=(3, 3), pad=(1, 1), num_filter=128, workspace=2048, name="conv4_3")
    relu4_3 = mx.symbol.Activation(data=conv4_3, act_type="relu", name="relu4_3")
    pool4 = mx.symbol.Pooling(
        data=relu4_3, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool4")
    # group 5
    conv5_1 = mx.symbol.Convolution(
        data=pool4, kernel=(3, 3), pad=(1, 1), num_filter=128, workspace=2048, name="conv5_1")
    relu5_1 = mx.symbol.Activation(data=conv5_1, act_type="relu", name="relu5_1")
    conv5_2 = mx.symbol.Convolution(
        data=relu5_1, kernel=(3, 3), pad=(1, 1), num_filter=128, workspace=2048, name="conv5_2")
    relu5_2 = mx.symbol.Activation(data=conv5_2, act_type="relu", name="relu5_2")
    conv5_3 = mx.symbol.Convolution(
        data=relu5_2, kernel=(3, 3), pad=(1, 1), num_filter=128, workspace=2048, name="conv5_3")
    relu5_3 = mx.symbol.Activation(data=conv5_3, act_type="relu", name="relu5_3")

    # shared convolutional layers
    conv_feat = relu5_3

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
        rois = mx.symbol.Proposal(
            cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
            feature_stride=config.RPN_FEAT_STRIDE, scales=tuple(config.ANCHOR_SCALES), ratios=tuple(config.ANCHOR_RATIOS),
            rpn_pre_nms_top_n=config.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=config.TEST.RPN_POST_NMS_TOP_N,
            threshold=config.TEST.RPN_NMS_THRESH, rpn_min_size=config.TEST.RPN_MIN_SIZE, iou_loss=config.RPN_IOU_LOSS)
    else:
        rois = mx.symbol.Custom(
            cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
            op_type='proposal', feat_stride=config.RPN_FEAT_STRIDE,
            scales=tuple(config.ANCHOR_SCALES), ratios=tuple(config.ANCHOR_RATIOS),
            rpn_pre_nms_top_n=config.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=config.TEST.RPN_POST_NMS_TOP_N,
            threshold=config.TEST.RPN_NMS_THRESH, rpn_min_size=config.TEST.RPN_MIN_SIZE, iou_loss=config.RPN_IOU_LOSS)

    # Fast R-CNN
    roi_pool = mx.symbol.ROIPooling(
        name='roi_pool', data=conv_feat, rois=rois, pooled_size=(7, 7), spatial_scale=1.0 / config.RCNN_FEAT_STRIDE)
    if config.RCNN_CTX_WINDOW:
        roi_pool_ctx = mx.symbol.ROIPooling(
            name='roi_pool_ctx', data=conv_feat, rois=rois, pooled_size=(7, 7),
            spatial_scale=1.0 / config.RCNN_FEAT_STRIDE, pad=0.25)
        roi_pool_concat = mx.symbol.Concat(roi_pool, roi_pool_ctx, name='roi_pool_concat')
        roi_pool_red = mx.symbol.Convolution(
            data=roi_pool_concat, num_filter=512, kernel=(1, 1), stride=(1, 1), name='roi_pool_ctx_red')
        roi_pool = mx.symbol.Activation(data=roi_pool_red, act_type='relu', name='roi_pool_relu')

    # group 6
    flatten = mx.symbol.Flatten(data=roi_pool, name="flatten")
    fc6 = mx.symbol.FullyConnected(data=flatten, num_hidden=512, name="fc6")
    relu6 = mx.symbol.Activation(data=fc6, act_type="relu", name="relu6")
    drop6 = mx.symbol.Dropout(data=relu6, p=0.5, name="drop6")
    # group 7
    fc7 = mx.symbol.FullyConnected(data=drop6, num_hidden=512, name="fc7")
    relu7 = mx.symbol.Activation(data=fc7, act_type="relu", name="relu7")
    drop7 = mx.symbol.Dropout(data=relu7, p=0.5, name="drop7")
    # classification
    cls_score = mx.symbol.FullyConnected(name='cls_score', data=drop7, num_hidden=num_classes)
    cls_prob = mx.symbol.SoftmaxOutput(name='cls_prob', data=cls_score)
    # bounding box regression
    bbox_pred = mx.symbol.FullyConnected(name='bbox_pred', data=drop7, num_hidden=num_classes * 4)

    # reshape output
    cls_prob = mx.symbol.Reshape(data=cls_prob, shape=(config.TEST.BATCH_IMAGES, -1, num_classes), name='cls_prob_reshape')
    bbox_pred = mx.symbol.Reshape(data=bbox_pred, shape=(config.TEST.BATCH_IMAGES, -1, 4 * num_classes), name='bbox_pred_reshape')

    # group output
    group = mx.symbol.Group([rois, cls_prob, bbox_pred])
    return group

def get_vgg_dilate_conv(data):
    """
    vgg-16
    shared convolutional layers,use dilated convolution in group 5
    :param data: Symbol
    :return: Symbol
    """

    # ====group 1

    conv1_1 = mx.symbol.Convolution(data=data, kernel=(3,3), pad=(1,1), \
                                    num_filter=64, name='conv1_1')
    relu1_1 = mx.symbol.Activation(data=conv1_1, act_type='relu', \
                                   name='relu1_1')

    conv1_2 = mx.symbol.Convolution(data=relu1_1,kernel=(3,3), pad=(1,1), \
                                    num_filter=64, name='conv1_2')
    relu1_2 = mx.symbol.Activation(data=conv1_2, act_type='relu', \
                                   name='relu1_2')

    pool1 = mx.symbol.Pooling(data=relu1_2, pool_type="max", kernel=(2,2), \
                              stride=(2,2), name="pool1")

    # ======group 2

    conv2_1 = mx.symbol.Convolution(data=pool1, kernel=(3,3), pad=(1,1), \
                                    num_filter=128, name='conv2_1')
    relu2_1 = mx.symbol.Activation(data=conv2_1, act_type="relu", \
                                   name="relu2_1")


    conv2_2 = mx.symbol.Convolution(data=relu2_1, kernel=(3,3), pad=(1,1), \
                                    num_filter=128, name="conv2_2")
    relu2_2 = mx.symbol.Activation(data=conv2_2, act_type="relu", \
                                   name="relu2_2")

    pool2 = mx.symbol.Pooling(data=relu2_2, pool_type="max",kernel=(2,2), \
                              stride=(2,2), name="pool2")

    # ======group 3
    conv3_1 = mx.symbol.Convolution(data=pool2, kernel=(3,3), pad=(1,1), \
                                    num_filter=256, name="conv3_1")
    relu3_1 = mx.symbol.Activation(data=conv3_1, act_type='relu', \
                                    name='relu3_1')

    conv3_2 = mx.symbol.Convolution(data=relu3_1, kernel=(3,3), pad=(1,1), \
                                    num_filter=256, name='conv3_2')
    relu3_2 = mx.symbol.Activation(data=conv3_2, act_type='relu', \
                                   name='relu3_2')

    conv3_3 = mx.symbol.Convolution(data=relu3_2, kernel=(3,3), pad=(1,1), \
                                    num_filter=256, name="conv3_3")
    relu3_3 = mx.symbol.Activation(data=conv3_3, act_type='relu', \
                                   name='relu3_3')

    pool3 = mx.symbol.Pooling(data=relu3_3, pool_type='max', kernel=(2,2), \
                              stride=(2,2), name="pool3")


    # ======group 4
    conv4_1 = mx.symbol.Convolution(data=pool3, kernel=(3,3), pad=(1, 1), \
                                    num_filter=512, name='conv4_1')
    relu4_1 = mx.symbol.Activation(data=conv4_1, act_type='relu', \
                                   name='relu4_1')

    conv4_2 = mx.symbol.Convolution(data=relu4_1,kernel=(3,3), pad=(1,1), \
                                    num_filter=512, name='conv4_2')
    relu4_2 = mx.symbol.Activation(data=conv4_1,act_type='relu',\
                                    name='relu4_2')

    conv4_3 = mx.symbol.Convolution(data=relu4_2,kernel=(3,3), pad=(1,1), \
                                    num_filter=512, name='conv4_3')
    relu4_3 = mx.symbol.Activation(data=conv4_3, act_type='relu', \
                                   name='relu4_3')

    pool4 = mx.symbol.Pooling(data=relu4_3,pool_type='max', kernel=(2,2), \
                              stride=(2,2), name='pool4')

    # ======group5
    # ======use dilation conv
    conv5_1 = mx.symbol.Convolution(data=pool4, kernel=(3,3), pad=(2,2), \
                                    dilate=(2,2), num_filter=512, \
                                    name='conv5_1')
    relu5_1 = mx.symbol.Activation(data=conv5_1, act_type='relu', \
                                   name='relu5_1')

    conv5_2 = mx.symbol.Convolution(data=relu5_1, kernel=(3,3), pad=(2,2), \
                                   dilate=(2,2), num_filter=512, \
                                   name='conv5_2')
    relu5_2 = mx.symbol.Activation(data=conv5_2, act_type='relu', \
                                    name='relu5_2')

    conv5_3 = mx.symbol.Convolution(data=relu5_2, kernel=(3,3), pad=(2,2), \
                                    dilate=(2,2), num_filter=512, \
                                    name='conv5_3')
    relu5_3 = mx.symbol.Activation(data=conv5_3, act_type='relu', \
                                   name='relu5_3')

    return relu5_3


def get_vgg_train_dff(num_classes=config.NUM_CLASSES, num_anchors=config.NUM_ANCHORS):
    """
    Faster R-CNN end-to-end with VGG 16 conv layers
    Edited for deep feature flow, use flownet warp feature
    :param num_classes: used to determine output size
    :param num_anchors: used to determine output size
    :return: Symbol
    """
    data = mx.symbol.Variable(name="data")
    data2 = mx.symbol.Variable(name="data2")

    im_info = mx.symbol.Variable(name="im_info")
    gt_boxes = mx.symbol.Variable(name="gt_boxes")
    rpn_label = mx.symbol.Variable(name='label')
    rpn_bbox_target = mx.symbol.Variable(name='bbox_target')
    rpn_bbox_weight = mx.symbol.Variable(name='bbox_weight')

    # shared convolutional layers
    # relu5_3 = get_vgg_conv(data)

    relu5_3 = get_vgg_dilate_conv(data2)
    relu5_3, _, _ = feature_propagate(relu5_3, data, data2)

    # flownet = stereo_scale_net(data*config.FLOW_SCALE_FACTOR, \
    #                            data2*config.FLOW_SCALE_FACTOR,\
    #                            net_type='flow')
    # flow = flownet[0]
    # scale = flownet[1]
    # scale_avg = mx.sym.Pooling(data=scale*0.125, pool_type='avg',\
    #                            kernel=(8,8),stride=(8,8),name="scale_avg")
    # flow_avg = mx.sym.Pooling(data=flow*0.125, pool_type='avg',\
    #                            kernel=(8,8),stride=(8,8),name="flow_avg")
    #
    # flow_grid = mx.symbol.GridGenerator(data=flow_avg,transform_type='warp',\
    #                                     name='flow_grid')
    # warp_res = mx.symbol.BilinearSampler(data=relu5_3,grid=flow_grid,\
    #                                      name='warp_res')
    #
    # relu5_3 = warp_res * scale_avg

    # RPN layers
    rpn_conv = mx.symbol.Convolution(
        data=relu5_3, kernel=(3, 3), pad=(1, 1), num_filter=512, name="rpn_conv_3x3")
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
                                           normalization='valid', use_ignore=True, ignore_label=-1,
                                           is_hidden_layer=config.TRAIN.RPN_OHEM, name="rpn_cls_prob")
    # bounding box regression
    if config.RPN_IOU_LOSS:
        rpn_bbox_loss_ = mx.symbol.Custom(data=rpn_bbox_pred, bbox_target=rpn_bbox_target, bbox_weight=rpn_bbox_weight,
                                          op_type='rpn_iou_loss', name='rpn_bbox_loss_',
                                          feature_stride=config.RPN_FEAT_STRIDE, scales=tuple(config.ANCHOR_SCALES),
                                          ratios=tuple(config.ANCHOR_RATIOS))
    else:
        rpn_bbox_loss_ = rpn_bbox_weight * mx.symbol.smooth_l1(name='rpn_bbox_loss_', scalar=3.0, data=(rpn_bbox_pred - rpn_bbox_target))

    # rpn output
    if config.TRAIN.RPN_OHEM:
        group = mx.symbol.Custom(
            cls_prob=rpn_cls_prob, bbox_loss=rpn_bbox_loss_, label=rpn_label, bbox_pred=rpn_bbox_pred,
            name='rpn_ohem', op_type='sample_anchors',
            feature_stride=config.RPN_FEAT_STRIDE, scales=tuple(config.ANCHOR_SCALES), ratios=tuple(config.ANCHOR_RATIOS),
            rpn_pre_nms_top_n=config.TRAIN.RPN_OHEM_ANCHORS, rpn_batch_size=config.TRAIN.RPN_BATCH_SIZE,
            nms_threshold=config.TRAIN.RPN_OHEM_NMS, iou_loss=config.RPN_IOU_LOSS,
            transform=config.TRAIN.RPN_OHEM_TRANSFORM, ignore=config.TRAIN.RPN_OHEM_IGNORE, np_ratio=config.TRAIN.RPN_OHEM_NP_RATIO)
        rpn_group = [group[0], group[1], group[2], group[3]]
    else:
        rpn_bbox_loss = mx.sym.MakeLoss(name='rpn_bbox_loss', data=rpn_bbox_loss_, grad_scale=1.0 / config.TRAIN.RPN_BATCH_SIZE)
        rpn_group = [rpn_cls_prob, rpn_bbox_loss]

    # ROI proposal
    rpn_cls_act = mx.symbol.SoftmaxActivation(
        data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_act")
    rpn_cls_act_reshape = mx.symbol.Reshape(
        data=rpn_cls_act, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_act_reshape')
    if config.TRAIN.CXX_PROPOSAL:
        rois = mx.symbol.Proposal(
            cls_prob=rpn_cls_act_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
            feature_stride=config.RPN_FEAT_STRIDE, scales=tuple(config.ANCHOR_SCALES), ratios=tuple(config.ANCHOR_RATIOS),
            rpn_pre_nms_top_n=config.TRAIN.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=config.TRAIN.RPN_POST_NMS_TOP_N,
            threshold=config.TRAIN.RPN_NMS_THRESH, rpn_min_size=config.TRAIN.RPN_MIN_SIZE, iou_loss=config.RPN_IOU_LOSS)
    else:
        rois = mx.symbol.Custom(
            cls_prob=rpn_cls_act_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
            op_type='proposal', feat_stride=config.RPN_FEAT_STRIDE,
            scales=tuple(config.ANCHOR_SCALES), ratios=tuple(config.ANCHOR_RATIOS),
            rpn_pre_nms_top_n=config.TRAIN.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=config.TRAIN.RPN_POST_NMS_TOP_N,
            threshold=config.TRAIN.RPN_NMS_THRESH, rpn_min_size=config.TRAIN.RPN_MIN_SIZE, iou_loss=config.RPN_IOU_LOSS)

    # ROI proposal target
    gt_boxes_reshape = mx.symbol.Reshape(data=gt_boxes, shape=(-1, 5), name='gt_boxes_reshape')
    if config.TRAIN.RCNN_OHEM:
        group = mx.symbol.Custom(rois=rois, gt_boxes=gt_boxes_reshape, op_type='proposal_target',
                                 num_classes=num_classes, batch_images=config.TRAIN.BATCH_IMAGES,
                                 batch_rois=config.TRAIN.RCNN_OHEM_ROIS, ohem=config.TRAIN.RCNN_OHEM)
    else:
        group = mx.symbol.Custom(rois=rois, gt_boxes=gt_boxes_reshape, op_type='proposal_target',
                                 num_classes=num_classes, batch_images=config.TRAIN.BATCH_IMAGES,
                                 batch_rois=config.TRAIN.BATCH_ROIS, fg_fraction=config.TRAIN.FG_FRACTION)
    rois = group[0]
    label = group[1]
    bbox_target = group[2]
    bbox_weight = group[3]

    # Fast R-CNN
    roi_pool = mx.symbol.ROIPooling(
        name='roi_pool', data=relu5_3, rois=rois, pooled_size=(7, 7), spatial_scale=1.0 / config.RCNN_FEAT_STRIDE)
    if config.RCNN_CTX_WINDOW:
        roi_pool_ctx = mx.symbol.ROIPooling(
            name='roi_pool_ctx', data=relu5_3, rois=rois, pooled_size=(7, 7),
            spatial_scale=1.0 / config.RCNN_FEAT_STRIDE, pad=0.25)
        roi_pool_concat = mx.symbol.Concat(roi_pool, roi_pool_ctx, name='roi_pool_concat')
        roi_pool_red = mx.symbol.Convolution(
            data=roi_pool_concat, num_filter=512, kernel=(1, 1), stride=(1, 1), name='roi_pool_ctx_red')
        roi_pool = mx.symbol.Activation(data=roi_pool_red, act_type='relu', name='roi_pool_relu')

    # group 6
    flatten = mx.symbol.Flatten(data=roi_pool, name="flatten")
    fc6 = mx.symbol.FullyConnected(data=flatten, num_hidden=4096, name="fc6")
    relu6 = mx.symbol.Activation(data=fc6, act_type="relu", name="relu6")
    drop6 = mx.symbol.Dropout(data=relu6, p=0.5, name="drop6")
    # group 7
    fc7 = mx.symbol.FullyConnected(data=drop6, num_hidden=4096, name="fc7")
    relu7 = mx.symbol.Activation(data=fc7, act_type="relu", name="relu7")
    drop7 = mx.symbol.Dropout(data=relu7, p=0.5, name="drop7")

    # classification
    cls_score = mx.symbol.FullyConnected(name='cls_score', data=drop7, num_hidden=num_classes)
    cls_prob = mx.symbol.SoftmaxOutput(name='cls_prob', data=cls_score, label=label, normalization='batch',
                                       is_hidden_layer=config.TRAIN.RCNN_OHEM)

    # bounding box regression
    bbox_pred = mx.symbol.FullyConnected(name='bbox_pred', data=drop7, num_hidden=num_classes * 4)
    if config.RCNN_IOU_LOSS:
        bbox_loss_ = mx.symbol.Custom(data=bbox_pred, bbox_target=bbox_target, bbox_weight=bbox_weight, rois=rois,
                                      op_type='rcnn_iou_loss', name='bbox_loss_', num_classes=num_classes)
    else:
        bbox_loss_ = bbox_weight * mx.symbol.smooth_l1(name='bbox_loss_', scalar=1.0, data=(bbox_pred - bbox_target))

    if config.TRAIN.RCNN_OHEM:
        group = mx.symbol.Custom(
            cls_prob=cls_prob, bbox_loss=bbox_loss_, label=label, rois=rois, bbox_pred=bbox_pred,
            name='rcnn_ohem', op_type='sample_rois',
            batch_images=config.TRAIN.BATCH_IMAGES, batch_size=config.TRAIN.BATCH_ROIS,
            nms_threshold=config.TRAIN.RCNN_OHEM_NMS, iou_loss=config.RPN_IOU_LOSS,
            transform=config.TRAIN.RCNN_OHEM_TRANSFORM, ignore=config.TRAIN.RCNN_OHEM_IGNORE)
        rcnn_group = [group[0], group[1], group[2], group[3]]
        for ind, name, last_shape in zip(range(len(rcnn_group)),
                                         ['cls_prob', 'bbox_loss', 'cls_mask', 'bbox_mask'],
                                         [num_classes, num_classes * 4, num_classes, num_classes * 4]):
            rcnn_group[ind] = mx.symbol.Reshape(data=rcnn_group[ind], shape=(config.TRAIN.BATCH_IMAGES, -1, last_shape),
                                                name=name + '_reshape')
    else:
        bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_, grad_scale=1.0 / config.TRAIN.BATCH_ROIS)
        rcnn_group = [cls_prob, bbox_loss]
        for ind, name, last_shape in zip(range(len(rcnn_group)),
                                         ['cls_prob', 'bbox_loss'],
                                         [num_classes, num_classes * 4]):
            rcnn_group[ind] = mx.symbol.Reshape(data=rcnn_group[ind], shape=(config.TRAIN.BATCH_IMAGES, -1, last_shape),
                                                name=name + '_reshape')

    # append label
    label = mx.symbol.Reshape(data=label, shape=(config.TRAIN.BATCH_IMAGES, -1), name='label_reshape')
    rcnn_group += [mx.symbol.BlockGrad(label, name='label_blockgrad')]

    group = mx.symbol.Group(rpn_group + rcnn_group)
    return group

def get_vgg_test_dff(num_classes=config.NUM_CLASSES, num_anchors=config.NUM_ANCHORS):
    """
    Faster R-CNN test with VGG 16 conv layers
    :param num_classes: used to determine output size
    :param num_anchors: used to determine output size
    :return: Symbol
    """
    data = mx.symbol.Variable(name="data")
    data2 = mx.symbol.Variable(name="data2")
    im_info = mx.symbol.Variable(name="im_info")

    # shared convolutional layers
    # relu5_3 = get_vgg_conv(data)
    relu5_3 = get_vgg_dilate_conv(data2)
    relu5_3, flow, flow_avg = feature_propagate(relu5_3, data, data2)

    # flownet = stereo_scale_net(data*config.FLOW_SCALE_FACTOR, \
    #                            data2*config.FLOW_SCALE_FACTOR,\
    #                            net_type='flow')
    # flow = flownet[0]
    # scale = flownet[1]
    # scale_avg = mx.sym.Pooling(data=scale*0.125, pool_type='avg',\
    #                            kernel=(8,8),stride=(8,8),name="scale_avg")
    # flow_avg = mx.sym.Pooling(data=flow*0.125, pool_type='avg',\
    #                            kernel=(8,8),stride=(8,8),name="flow_avg")
    #
    # flow_grid = mx.symbol.GridGenerator(data=flow_avg,transform_type='warp',\
    #                                     name='flow_grid')
    # warp_res = mx.symbol.BilinearSampler(data=relu5_3,grid=flow_grid,\
    #                                      name='warp_res')
    #
    # relu5_3 = warp_res * scale_avg

    # RPN
    rpn_conv = mx.symbol.Convolution(
        data=relu5_3, kernel=(3, 3), pad=(1, 1), num_filter=512, name="rpn_conv_3x3")
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
        rois = mx.symbol.Proposal(
            cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
            feature_stride=config.RPN_FEAT_STRIDE, scales=tuple(config.ANCHOR_SCALES), ratios=tuple(config.ANCHOR_RATIOS),
            rpn_pre_nms_top_n=config.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=config.TEST.RPN_POST_NMS_TOP_N,
            threshold=config.TEST.RPN_NMS_THRESH, rpn_min_size=config.TEST.RPN_MIN_SIZE, iou_loss=config.RPN_IOU_LOSS)
    else:
        rois = mx.symbol.Custom(
            cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
            op_type='proposal', feat_stride=config.RPN_FEAT_STRIDE,
            scales=tuple(config.ANCHOR_SCALES), ratios=tuple(config.ANCHOR_RATIOS),
            rpn_pre_nms_top_n=config.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=config.TEST.RPN_POST_NMS_TOP_N,
            threshold=config.TEST.RPN_NMS_THRESH, rpn_min_size=config.TEST.RPN_MIN_SIZE, iou_loss=config.RPN_IOU_LOSS)

    # Fast R-CNN
    roi_pool = mx.symbol.ROIPooling(
        name='roi_pool', data=relu5_3, rois=rois, pooled_size=(7, 7), spatial_scale=1.0 / config.RCNN_FEAT_STRIDE)
    if config.RCNN_CTX_WINDOW:
        roi_pool_ctx = mx.symbol.ROIPooling(
            name='roi_pool_ctx', data=relu5_3, rois=rois, pooled_size=(7, 7),
            spatial_scale=1.0 / config.RCNN_FEAT_STRIDE, pad=0.25)
        roi_pool_concat = mx.symbol.Concat(roi_pool, roi_pool_ctx, name='roi_pool_concat')
        roi_pool_red = mx.symbol.Convolution(
            data=roi_pool_concat, num_filter=512, kernel=(1, 1), stride=(1, 1), name='roi_pool_ctx_red')
        roi_pool = mx.symbol.Activation(data=roi_pool_red, act_type='relu', name='roi_pool_relu')

    # group 6
    flatten = mx.symbol.Flatten(data=roi_pool, name="flatten")
    fc6 = mx.symbol.FullyConnected(data=flatten, num_hidden=4096, name="fc6")
    relu6 = mx.symbol.Activation(data=fc6, act_type="relu", name="relu6")
    drop6 = mx.symbol.Dropout(data=relu6, p=0.5, name="drop6")
    # group 7
    fc7 = mx.symbol.FullyConnected(data=drop6, num_hidden=4096, name="fc7")
    relu7 = mx.symbol.Activation(data=fc7, act_type="relu", name="relu7")
    drop7 = mx.symbol.Dropout(data=relu7, p=0.5, name="drop7")
    # classification
    cls_score = mx.symbol.FullyConnected(name='cls_score', data=drop7, num_hidden=num_classes)
    cls_prob = mx.symbol.SoftmaxOutput(name='cls_prob', data=cls_score)
    # bounding box regression
    bbox_pred = mx.symbol.FullyConnected(name='bbox_pred', data=drop7, num_hidden=num_classes * 4)

    # reshape output
    cls_prob = mx.symbol.Reshape(data=cls_prob, shape=(config.TEST.BATCH_IMAGES, -1, num_classes), name='cls_prob_reshape')
    bbox_pred = mx.symbol.Reshape(data=bbox_pred, shape=(config.TEST.BATCH_IMAGES, -1, 4 * num_classes), name='bbox_pred_reshape')

    # group output
    # group = mx.symbol.Group([rois, cls_prob, bbox_pred])
    group = mx.symbol.Group([rois, cls_prob, bbox_pred, flow, flow_avg])
    return group


def get_vgg_train_dff_cycle(num_classes=config.NUM_CLASSES, num_anchors=config.NUM_ANCHORS):
    """
    Faster R-CNN end-to-end with VGG 16 conv layers
    Edited for deep feature flow, use flownet warp feature
    :param num_classes: used to determine output size
    :param num_anchors: used to determine output size
    :return: Symbol
    """
    data = mx.symbol.Variable(name="data")
    data2 = mx.symbol.Variable(name="data2")

    im_info = mx.symbol.Variable(name="im_info")
    gt_boxes = mx.symbol.Variable(name="gt_boxes")
    rpn_label = mx.symbol.Variable(name='label')
    rpn_bbox_target = mx.symbol.Variable(name='bbox_target')
    rpn_bbox_weight = mx.symbol.Variable(name='bbox_weight')

    param_variable_list = [mx.sym.Variable(item) for item in config.SHARE_PARAMS_LIST]
    param_dic = dict(zip(config.SHARE_PARAMS_LIST, param_variable_list))
    print param_dic

    # shared convolutional layers
    # relu5_3 = get_vgg_conv(data)

    # relu5_3 = get_vgg_dilate_conv(data2)
    # relu5_3, _, _ = feature_propagate(relu5_3, data, data2)

    # Cycle rcnn_dff
    # Prev_Image   ------>   Curr_Image
    #                            |
    #                            |
    #                            v
    # Prev_Feature <-----  Curr_Feature
    #
    # Curr_Image   ------>   Prev_Image
    #                            |
    #                            |
    #                            v
    # Curr_Feature <-----  Prev_Feature

    src_feature = get_vgg_dilate_conv(data)
    temp_feature, _, _ = feature_propagate_share(param_dic, src_feature, data2, data)
    dst_feature, _, _ = feature_propagate_share(param_dic, temp_feature, data, data2)
    relu5_3 = dst_feature

    # RPN layers
    rpn_conv = mx.symbol.Convolution(
        data=relu5_3, kernel=(3, 3), pad=(1, 1), num_filter=512, name="rpn_conv_3x3")
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
                                           normalization='valid', use_ignore=True, ignore_label=-1,
                                           is_hidden_layer=config.TRAIN.RPN_OHEM, name="rpn_cls_prob")
    # bounding box regression
    if config.RPN_IOU_LOSS:
        rpn_bbox_loss_ = mx.symbol.Custom(data=rpn_bbox_pred, bbox_target=rpn_bbox_target, bbox_weight=rpn_bbox_weight,
                                          op_type='rpn_iou_loss', name='rpn_bbox_loss_',
                                          feature_stride=config.RPN_FEAT_STRIDE, scales=tuple(config.ANCHOR_SCALES),
                                          ratios=tuple(config.ANCHOR_RATIOS))
    else:
        rpn_bbox_loss_ = rpn_bbox_weight * mx.symbol.smooth_l1(name='rpn_bbox_loss_', scalar=3.0, data=(rpn_bbox_pred - rpn_bbox_target))

    # rpn output
    if config.TRAIN.RPN_OHEM:
        group = mx.symbol.Custom(
            cls_prob=rpn_cls_prob, bbox_loss=rpn_bbox_loss_, label=rpn_label, bbox_pred=rpn_bbox_pred,
            name='rpn_ohem', op_type='sample_anchors',
            feature_stride=config.RPN_FEAT_STRIDE, scales=tuple(config.ANCHOR_SCALES), ratios=tuple(config.ANCHOR_RATIOS),
            rpn_pre_nms_top_n=config.TRAIN.RPN_OHEM_ANCHORS, rpn_batch_size=config.TRAIN.RPN_BATCH_SIZE,
            nms_threshold=config.TRAIN.RPN_OHEM_NMS, iou_loss=config.RPN_IOU_LOSS,
            transform=config.TRAIN.RPN_OHEM_TRANSFORM, ignore=config.TRAIN.RPN_OHEM_IGNORE, np_ratio=config.TRAIN.RPN_OHEM_NP_RATIO)
        rpn_group = [group[0], group[1], group[2], group[3]]
    else:
        rpn_bbox_loss = mx.sym.MakeLoss(name='rpn_bbox_loss', data=rpn_bbox_loss_, grad_scale=1.0 / config.TRAIN.RPN_BATCH_SIZE)
        rpn_group = [rpn_cls_prob, rpn_bbox_loss]

    # ROI proposal
    rpn_cls_act = mx.symbol.SoftmaxActivation(
        data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_act")
    rpn_cls_act_reshape = mx.symbol.Reshape(
        data=rpn_cls_act, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_act_reshape')
    if config.TRAIN.CXX_PROPOSAL:
        rois = mx.symbol.Proposal(
            cls_prob=rpn_cls_act_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
            feature_stride=config.RPN_FEAT_STRIDE, scales=tuple(config.ANCHOR_SCALES), ratios=tuple(config.ANCHOR_RATIOS),
            rpn_pre_nms_top_n=config.TRAIN.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=config.TRAIN.RPN_POST_NMS_TOP_N,
            threshold=config.TRAIN.RPN_NMS_THRESH, rpn_min_size=config.TRAIN.RPN_MIN_SIZE, iou_loss=config.RPN_IOU_LOSS)
    else:
        rois = mx.symbol.Custom(
            cls_prob=rpn_cls_act_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
            op_type='proposal', feat_stride=config.RPN_FEAT_STRIDE,
            scales=tuple(config.ANCHOR_SCALES), ratios=tuple(config.ANCHOR_RATIOS),
            rpn_pre_nms_top_n=config.TRAIN.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=config.TRAIN.RPN_POST_NMS_TOP_N,
            threshold=config.TRAIN.RPN_NMS_THRESH, rpn_min_size=config.TRAIN.RPN_MIN_SIZE, iou_loss=config.RPN_IOU_LOSS)

    # ROI proposal target
    gt_boxes_reshape = mx.symbol.Reshape(data=gt_boxes, shape=(-1, 5), name='gt_boxes_reshape')
    if config.TRAIN.RCNN_OHEM:
        group = mx.symbol.Custom(rois=rois, gt_boxes=gt_boxes_reshape, op_type='proposal_target',
                                 num_classes=num_classes, batch_images=config.TRAIN.BATCH_IMAGES,
                                 batch_rois=config.TRAIN.RCNN_OHEM_ROIS, ohem=config.TRAIN.RCNN_OHEM)
    else:
        group = mx.symbol.Custom(rois=rois, gt_boxes=gt_boxes_reshape, op_type='proposal_target',
                                 num_classes=num_classes, batch_images=config.TRAIN.BATCH_IMAGES,
                                 batch_rois=config.TRAIN.BATCH_ROIS, fg_fraction=config.TRAIN.FG_FRACTION)
    rois = group[0]
    label = group[1]
    bbox_target = group[2]
    bbox_weight = group[3]

    # Fast R-CNN
    roi_pool = mx.symbol.ROIPooling(
        name='roi_pool', data=relu5_3, rois=rois, pooled_size=(7, 7), spatial_scale=1.0 / config.RCNN_FEAT_STRIDE)
    if config.RCNN_CTX_WINDOW:
        roi_pool_ctx = mx.symbol.ROIPooling(
            name='roi_pool_ctx', data=relu5_3, rois=rois, pooled_size=(7, 7),
            spatial_scale=1.0 / config.RCNN_FEAT_STRIDE, pad=0.25)
        roi_pool_concat = mx.symbol.Concat(roi_pool, roi_pool_ctx, name='roi_pool_concat')
        roi_pool_red = mx.symbol.Convolution(
            data=roi_pool_concat, num_filter=512, kernel=(1, 1), stride=(1, 1), name='roi_pool_ctx_red')
        roi_pool = mx.symbol.Activation(data=roi_pool_red, act_type='relu', name='roi_pool_relu')

    # group 6
    flatten = mx.symbol.Flatten(data=roi_pool, name="flatten")
    fc6 = mx.symbol.FullyConnected(data=flatten, num_hidden=4096, name="fc6")
    relu6 = mx.symbol.Activation(data=fc6, act_type="relu", name="relu6")
    drop6 = mx.symbol.Dropout(data=relu6, p=0.5, name="drop6")
    # group 7
    fc7 = mx.symbol.FullyConnected(data=drop6, num_hidden=4096, name="fc7")
    relu7 = mx.symbol.Activation(data=fc7, act_type="relu", name="relu7")
    drop7 = mx.symbol.Dropout(data=relu7, p=0.5, name="drop7")

    # classification
    cls_score = mx.symbol.FullyConnected(name='cls_score', data=drop7, num_hidden=num_classes)
    cls_prob = mx.symbol.SoftmaxOutput(name='cls_prob', data=cls_score, label=label, normalization='batch',
                                       is_hidden_layer=config.TRAIN.RCNN_OHEM)

    # bounding box regression
    bbox_pred = mx.symbol.FullyConnected(name='bbox_pred', data=drop7, num_hidden=num_classes * 4)
    if config.RCNN_IOU_LOSS:
        bbox_loss_ = mx.symbol.Custom(data=bbox_pred, bbox_target=bbox_target, bbox_weight=bbox_weight, rois=rois,
                                      op_type='rcnn_iou_loss', name='bbox_loss_', num_classes=num_classes)
    else:
        bbox_loss_ = bbox_weight * mx.symbol.smooth_l1(name='bbox_loss_', scalar=1.0, data=(bbox_pred - bbox_target))

    if config.TRAIN.RCNN_OHEM:
        group = mx.symbol.Custom(
            cls_prob=cls_prob, bbox_loss=bbox_loss_, label=label, rois=rois, bbox_pred=bbox_pred,
            name='rcnn_ohem', op_type='sample_rois',
            batch_images=config.TRAIN.BATCH_IMAGES, batch_size=config.TRAIN.BATCH_ROIS,
            nms_threshold=config.TRAIN.RCNN_OHEM_NMS, iou_loss=config.RPN_IOU_LOSS,
            transform=config.TRAIN.RCNN_OHEM_TRANSFORM, ignore=config.TRAIN.RCNN_OHEM_IGNORE)
        rcnn_group = [group[0], group[1], group[2], group[3]]
        for ind, name, last_shape in zip(range(len(rcnn_group)),
                                         ['cls_prob', 'bbox_loss', 'cls_mask', 'bbox_mask'],
                                         [num_classes, num_classes * 4, num_classes, num_classes * 4]):
            rcnn_group[ind] = mx.symbol.Reshape(data=rcnn_group[ind], shape=(config.TRAIN.BATCH_IMAGES, -1, last_shape),
                                                name=name + '_reshape')
    else:
        bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_, grad_scale=1.0 / config.TRAIN.BATCH_ROIS)
        rcnn_group = [cls_prob, bbox_loss]
        for ind, name, last_shape in zip(range(len(rcnn_group)),
                                         ['cls_prob', 'bbox_loss'],
                                         [num_classes, num_classes * 4]):
            rcnn_group[ind] = mx.symbol.Reshape(data=rcnn_group[ind], shape=(config.TRAIN.BATCH_IMAGES, -1, last_shape),
                                                name=name + '_reshape')

    # append label
    label = mx.symbol.Reshape(data=label, shape=(config.TRAIN.BATCH_IMAGES, -1), name='label_reshape')
    rcnn_group += [mx.symbol.BlockGrad(label, name='label_blockgrad')]

    group = mx.symbol.Group(rpn_group + rcnn_group)
    return group

def get_embedding_weight(feature_input,embedding_param_dic):
    embedding_conv1 = mx.symbol.Convolution(data=feature_input, kernel=(1, 1), \
                        pad=(0, 0), num_filter=512, name="embedding_conv1",\
                        weight=embedding_param_dic['embedding_conv1_weight'],\
                        bias=embedding_param_dic['embedding_conv1_bias'])
    embedding_conv2 = mx.symbol.Convolution(data=embedding_conv1, kernel=(1, 1),\
                        pad=(1, 1), num_filter=512, name="embedding_conv2",\
                        weight=embedding_param_dic['embedding_conv2_weight'],\
                        bias=embedding_param_dic['embedding_conv2_bias'])
    embedding_conv3 = mx.symbol.Convolution(data=embedding_conv2, kernel=(1, 1),\
                        pad=(0, 0), num_filter=512, name="embedding_conv3",\
                        weight=embedding_param_dic['embedding_conv3_weight'],\
                        bias=embedding_param_dic['embedding_conv3_bias'])

    embedding_flatten = mx.symbol.Flatten(data=embedding_conv3, name='embedding_flatten')
    return embedding_flatten

def get_feature_aggregation(relu5_3_dict):
    embedding_param_variable_list = [mx.sym.Variable(item) for item in config.EMBEDDING_PARAMS_LIST]
    embedding_param_dic = dict(zip(config.EMBEDDING_PARAMS_LIST, embedding_param_variable_list))

    embedding_dict = {}
    for item in relu5_3_dict.keys():
        embedding_dict[item] = get_embedding_weight(relu5_3_dict[item],embedding_param_dic)

    ref_feature = embedding_dict['data']
    arg_shape, output_shape, aux_shape = ref_feature.infer_shape(data=(1, 3, 384, 1280))
    print '####ref_feature:',arg_shape, output_shape, aux_shape
    # ref_l2norm_vector = mx.symbol.L2Normalization(mx.symbol.Reshape(ref_feature, shape=(-1,)), name='ref_l2norm')
    ref_length_sqr = mx.symbol.dot(ref_feature ,mx.symbol.Reshape(ref_feature, shape=(-1,)))
    ref_length = mx.symbol.sqrt(ref_length_sqr)

    arg_shape, output_shape, aux_shape = ref_length.infer_shape(data=(1, 3, 384, 1280))
    print '####ref_l2norm:',arg_shape, output_shape, aux_shape


    score_dict = {}
    dot_product_dict = {}
    l2_product_dict = {}
    for item in embedding_dict.keys():
        # print "-----------", item
        curr_feature = embedding_dict[item]
        dot_product = mx.symbol.dot(curr_feature, mx.symbol.Reshape(ref_feature,\
                                                                    shape=(-1,)), \
                                    name='dot_product_{}'.format(item))
        dot_product_dict[item] = dot_product

        curr_length_sqr = mx.symbol.dot(curr_feature ,mx.symbol.Reshape(curr_feature, shape=(-1,)))
        curr_length = mx.symbol.sqrt(curr_length_sqr)

        # l2_product = mx.symbol.Reshape(curr_lengh * ref_length, shape=(1,1))
        l2_product = curr_length * ref_length

        l2_product_dict[item] = l2_product

        score_dict[item] = mx.symbol.exp(mx.symbol.Reshape(dot_product, shape=(1,))  / mx.symbol.Reshape(l2_product, shape=(1,)), name='score_{}'.format(item))
    # print score_dict.keys()

    arg_shape, output_shape, aux_shape = dot_product_dict['data'].infer_shape(data=(1, 3, 384, 1280))
    print '$$$$$dot_product:',arg_shape, output_shape, aux_shape
    arg_shape, output_shape, aux_shape = l2_product_dict['data'].infer_shape(data=(1, 3, 384, 1280))
    print '$$$$$l2_product:',arg_shape, output_shape, aux_shape

    arg_shape, output_shape, aux_shape = score_dict['data'].infer_shape(data=(1, 3, 384, 1280))
    print '$$$$$score_dict:',arg_shape, output_shape, aux_shape

    weight_dict = {}

    weight_sum = score_dict['data']
    print weight_sum.list_outputs()
    print weight_sum.list_arguments()
    arg_shape, output_shape, aux_shape = weight_sum.infer_shape(data=(1, 3, 384, 1280))
    print '$$$$$weight_sum:',arg_shape, output_shape, aux_shape

    for item in score_dict.keys():
        if item != 'data':
            weight_sum += score_dict[item]
    print weight_sum.list_outputs()
    print weight_sum.list_arguments()
    # weight_temp = mx.symbol.Group(weight_sum)
    # weight_sum_ = mx.symbol.ElementWiseSum(weight_temp)

    # for item in score_dict.keys():
    #     if item != 'data':
    #         weight_sum = mx.symbol.broadcast_add(lhs = weight_sum, rhs = score_dict[item], name = 'weight_sum')

    # print weight_sum_
    arg_shape, output_shape, aux_shape = weight_sum.infer_shape(data=(1, 3, 384, 1280),prev_1=(1, 3, 384, 1280),prev_2=(1, 3, 384, 1280),next_1=(1, 3, 384, 1280),next_2=(1, 3, 384, 1280))
    print '$$$$$weight_sum new:',arg_shape, output_shape, aux_shape

    for item in score_dict.keys():
        weight_dict[item] = score_dict[item] / weight_sum

    arg_shape, output_shape, aux_shape = weight_dict['data'].infer_shape(data=(1, 3, 384, 1280),prev_1=(1, 3, 384, 1280),prev_2=(1, 3, 384, 1280),next_1=(1, 3, 384, 1280),next_2=(1, 3, 384, 1280))
    print '$$$$$weight_product:',arg_shape, output_shape, aux_shape



    # sorted_score_dict = OrderedDict(sorted(score_dict.items(), key=lambda item_temp:item_temp[0]))
    # print "dict",sorted_score_dict
    # softmax_input = sorted_score_dict.values()
    # softmax_output = mx.symbol.SoftmaxActivation(data=softmax_input, mode="channel")
    #
    # weight_dict = dict(zip(sorted_score_dict.keys(), softmax_output))

    # print "weight_dict", weight_dict
    relu5_3 = relu5_3_dict['data']
    arg_shape, output_shape, aux_shape = relu5_3.infer_shape(data=(1, 3, 384, 1280))
    print '$$$$$:',arg_shape, output_shape, aux_shape
    print 'relu5_3', relu5_3
    for item in weight_dict.keys():
        # print 'item', item
        relu5_3_temp = mx.symbol.broadcast_mul(lhs=relu5_3_dict[item], rhs=weight_dict[item])
        relu5_3 += relu5_3_temp
    relu5_3 -= relu5_3_dict['data']

    return relu5_3

def get_vgg_train_ffa(num_classes=config.NUM_CLASSES, num_anchors=config.NUM_ANCHORS):
    """
    Flow Feature Aggregation
    Faster R-CNN end-to-end with VGG 16 conv layers
    Edited for deep feature flow, use flownet warp feature
    :param num_classes: used to determine output size
    :param num_anchors: used to determine output size
    :return: Symbol
    """
    param_variable_list = [mx.sym.Variable(item) for item in config.SHARE_PARAMS_LIST]
    param_dic = dict(zip(config.SHARE_PARAMS_LIST, param_variable_list))

    data_dict = {}
    for i in range(config.FRAMES_FEATURE_AGGREGATION):
        data_dict['prev_{}'.format(i+1)] = mx.symbol.Variable(name='prev_{}'.format(i+1))
        data_dict['next_{}'.format(i+1)] = mx.symbol.Variable(name='next_{}'.format(i+1))
    data_dict['data'] = mx.symbol.Variable(name="data")
    # data = data_dict['data']

    im_info = mx.symbol.Variable(name="im_info")
    gt_boxes = mx.symbol.Variable(name="gt_boxes")
    rpn_label = mx.symbol.Variable(name='label')
    rpn_bbox_target = mx.symbol.Variable(name='bbox_target')
    rpn_bbox_weight = mx.symbol.Variable(name='bbox_weight')

    # shared convolutional layers
    # relu5_3 = get_vgg_conv(data)
    relu5_3_ref = get_vgg_dilate_conv(data_dict['data'])
    arg_shape, output_shape, aux_shape = relu5_3_ref.infer_shape(data=(1, 3, 384, 1280))
    print 'aaaaaa relu5_3_ref:',arg_shape, output_shape, aux_shape

    relu5_3_dict = {}

    for item in data_dict.keys():
        # print "#####", data_dict[item].list_arguments()
        # feature_propagate_share(param_dic, src_feature, data2, data)
        relu5_3_dict[item], _, _ = feature_propagate_share(item, param_dic, relu5_3_ref, data_dict[item], data_dict['data'])
        # print "~~~~~~", relu5_3_dict[item].list_arguments()
    relu5_3_dict['data'] = relu5_3_ref
    print "flow flow flow", relu5_3_dict['prev_1'].list_arguments()
    relu5_3 = get_feature_aggregation(relu5_3_dict)
    print "arguments", relu5_3.list_arguments()
    # relu5_3_list = relu5_3_dict.values()
    # relu5_3 = sum(relu5_3_list) + relu5_3_ref

    # RPN layers
    rpn_conv = mx.symbol.Convolution(
        data=relu5_3, kernel=(3, 3), pad=(1, 1), num_filter=512, name="rpn_conv_3x3")
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
                                           normalization='valid', use_ignore=True, ignore_label=-1,
                                           is_hidden_layer=config.TRAIN.RPN_OHEM, name="rpn_cls_prob")
    # bounding box regression
    if config.RPN_IOU_LOSS:
        rpn_bbox_loss_ = mx.symbol.Custom(data=rpn_bbox_pred, bbox_target=rpn_bbox_target, bbox_weight=rpn_bbox_weight,
                                          op_type='rpn_iou_loss', name='rpn_bbox_loss_',
                                          feature_stride=config.RPN_FEAT_STRIDE, scales=tuple(config.ANCHOR_SCALES),
                                          ratios=tuple(config.ANCHOR_RATIOS))
    else:
        rpn_bbox_loss_ = rpn_bbox_weight * mx.symbol.smooth_l1(name='rpn_bbox_loss_', scalar=3.0, data=(rpn_bbox_pred - rpn_bbox_target))

    # rpn output
    if config.TRAIN.RPN_OHEM:
        group = mx.symbol.Custom(
            cls_prob=rpn_cls_prob, bbox_loss=rpn_bbox_loss_, label=rpn_label, bbox_pred=rpn_bbox_pred,
            name='rpn_ohem', op_type='sample_anchors',
            feature_stride=config.RPN_FEAT_STRIDE, scales=tuple(config.ANCHOR_SCALES), ratios=tuple(config.ANCHOR_RATIOS),
            rpn_pre_nms_top_n=config.TRAIN.RPN_OHEM_ANCHORS, rpn_batch_size=config.TRAIN.RPN_BATCH_SIZE,
            nms_threshold=config.TRAIN.RPN_OHEM_NMS, iou_loss=config.RPN_IOU_LOSS,
            transform=config.TRAIN.RPN_OHEM_TRANSFORM, ignore=config.TRAIN.RPN_OHEM_IGNORE, np_ratio=config.TRAIN.RPN_OHEM_NP_RATIO)
        rpn_group = [group[0], group[1], group[2], group[3]]
    else:
        rpn_bbox_loss = mx.sym.MakeLoss(name='rpn_bbox_loss', data=rpn_bbox_loss_, grad_scale=1.0 / config.TRAIN.RPN_BATCH_SIZE)
        rpn_group = [rpn_cls_prob, rpn_bbox_loss]

    # ROI proposal
    rpn_cls_act = mx.symbol.SoftmaxActivation(
        data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_act")
    rpn_cls_act_reshape = mx.symbol.Reshape(
        data=rpn_cls_act, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_act_reshape')
    if config.TRAIN.CXX_PROPOSAL:
        rois = mx.symbol.Proposal(
            cls_prob=rpn_cls_act_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
            feature_stride=config.RPN_FEAT_STRIDE, scales=tuple(config.ANCHOR_SCALES), ratios=tuple(config.ANCHOR_RATIOS),
            rpn_pre_nms_top_n=config.TRAIN.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=config.TRAIN.RPN_POST_NMS_TOP_N,
            threshold=config.TRAIN.RPN_NMS_THRESH, rpn_min_size=config.TRAIN.RPN_MIN_SIZE, iou_loss=config.RPN_IOU_LOSS)
    else:
        rois = mx.symbol.Custom(
            cls_prob=rpn_cls_act_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
            op_type='proposal', feat_stride=config.RPN_FEAT_STRIDE,
            scales=tuple(config.ANCHOR_SCALES), ratios=tuple(config.ANCHOR_RATIOS),
            rpn_pre_nms_top_n=config.TRAIN.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=config.TRAIN.RPN_POST_NMS_TOP_N,
            threshold=config.TRAIN.RPN_NMS_THRESH, rpn_min_size=config.TRAIN.RPN_MIN_SIZE, iou_loss=config.RPN_IOU_LOSS)

    # ROI proposal target
    gt_boxes_reshape = mx.symbol.Reshape(data=gt_boxes, shape=(-1, 5), name='gt_boxes_reshape')
    if config.TRAIN.RCNN_OHEM:
        group = mx.symbol.Custom(rois=rois, gt_boxes=gt_boxes_reshape, op_type='proposal_target',
                                 num_classes=num_classes, batch_images=config.TRAIN.BATCH_IMAGES,
                                 batch_rois=config.TRAIN.RCNN_OHEM_ROIS, ohem=config.TRAIN.RCNN_OHEM)
    else:
        group = mx.symbol.Custom(rois=rois, gt_boxes=gt_boxes_reshape, op_type='proposal_target',
                                 num_classes=num_classes, batch_images=config.TRAIN.BATCH_IMAGES,
                                 batch_rois=config.TRAIN.BATCH_ROIS, fg_fraction=config.TRAIN.FG_FRACTION)
    rois = group[0]
    label = group[1]
    bbox_target = group[2]
    bbox_weight = group[3]

    # Fast R-CNN
    roi_pool = mx.symbol.ROIPooling(
        name='roi_pool', data=relu5_3, rois=rois, pooled_size=(7, 7), spatial_scale=1.0 / config.RCNN_FEAT_STRIDE)
    if config.RCNN_CTX_WINDOW:
        roi_pool_ctx = mx.symbol.ROIPooling(
            name='roi_pool_ctx', data=relu5_3, rois=rois, pooled_size=(7, 7),
            spatial_scale=1.0 / config.RCNN_FEAT_STRIDE, pad=0.25)
        roi_pool_concat = mx.symbol.Concat(roi_pool, roi_pool_ctx, name='roi_pool_concat')
        roi_pool_red = mx.symbol.Convolution(
            data=roi_pool_concat, num_filter=512, kernel=(1, 1), stride=(1, 1), name='roi_pool_ctx_red')
        roi_pool = mx.symbol.Activation(data=roi_pool_red, act_type='relu', name='roi_pool_relu')

    # group 6
    flatten = mx.symbol.Flatten(data=roi_pool, name="flatten")
    fc6 = mx.symbol.FullyConnected(data=flatten, num_hidden=4096, name="fc6")
    relu6 = mx.symbol.Activation(data=fc6, act_type="relu", name="relu6")
    drop6 = mx.symbol.Dropout(data=relu6, p=0.5, name="drop6")
    # group 7
    fc7 = mx.symbol.FullyConnected(data=drop6, num_hidden=4096, name="fc7")
    relu7 = mx.symbol.Activation(data=fc7, act_type="relu", name="relu7")
    drop7 = mx.symbol.Dropout(data=relu7, p=0.5, name="drop7")

    # classification
    cls_score = mx.symbol.FullyConnected(name='cls_score', data=drop7, num_hidden=num_classes)
    cls_prob = mx.symbol.SoftmaxOutput(name='cls_prob', data=cls_score, label=label, normalization='batch',
                                       is_hidden_layer=config.TRAIN.RCNN_OHEM)

    # bounding box regression
    bbox_pred = mx.symbol.FullyConnected(name='bbox_pred', data=drop7, num_hidden=num_classes * 4)
    if config.RCNN_IOU_LOSS:
        bbox_loss_ = mx.symbol.Custom(data=bbox_pred, bbox_target=bbox_target, bbox_weight=bbox_weight, rois=rois,
                                      op_type='rcnn_iou_loss', name='bbox_loss_', num_classes=num_classes)
    else:
        bbox_loss_ = bbox_weight * mx.symbol.smooth_l1(name='bbox_loss_', scalar=1.0, data=(bbox_pred - bbox_target))

    if config.TRAIN.RCNN_OHEM:
        group = mx.symbol.Custom(
            cls_prob=cls_prob, bbox_loss=bbox_loss_, label=label, rois=rois, bbox_pred=bbox_pred,
            name='rcnn_ohem', op_type='sample_rois',
            batch_images=config.TRAIN.BATCH_IMAGES, batch_size=config.TRAIN.BATCH_ROIS,
            nms_threshold=config.TRAIN.RCNN_OHEM_NMS, iou_loss=config.RPN_IOU_LOSS,
            transform=config.TRAIN.RCNN_OHEM_TRANSFORM, ignore=config.TRAIN.RCNN_OHEM_IGNORE)
        rcnn_group = [group[0], group[1], group[2], group[3]]
        for ind, name, last_shape in zip(range(len(rcnn_group)),
                                         ['cls_prob', 'bbox_loss', 'cls_mask', 'bbox_mask'],
                                         [num_classes, num_classes * 4, num_classes, num_classes * 4]):
            rcnn_group[ind] = mx.symbol.Reshape(data=rcnn_group[ind], shape=(config.TRAIN.BATCH_IMAGES, -1, last_shape),
                                                name=name + '_reshape')
    else:
        bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_, grad_scale=1.0 / config.TRAIN.BATCH_ROIS)
        rcnn_group = [cls_prob, bbox_loss]
        for ind, name, last_shape in zip(range(len(rcnn_group)),
                                         ['cls_prob', 'bbox_loss'],
                                         [num_classes, num_classes * 4]):
            rcnn_group[ind] = mx.symbol.Reshape(data=rcnn_group[ind], shape=(config.TRAIN.BATCH_IMAGES, -1, last_shape),
                                                name=name + '_reshape')

    # append label
    label = mx.symbol.Reshape(data=label, shape=(config.TRAIN.BATCH_IMAGES, -1), name='label_reshape')
    rcnn_group += [mx.symbol.BlockGrad(label, name='label_blockgrad')]

    group = mx.symbol.Group(rpn_group + rcnn_group)
    return group
