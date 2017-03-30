import mxnet as mx
import proposal
import proposal_target
from rcnn.config import config

# enet settings
eps = 1e-5 + 1e-10
use_global_stats = True


def get_conv(name, data, num_filter, kernel, stride, pad,
             with_relu, bn_momentum, dilate=(1, 1)):
    conv = mx.symbol.Convolution(
        name=name,
        data=data,
        num_filter=num_filter,
        kernel=kernel,
        stride=stride,
        pad=pad,
        dilate=dilate,
        no_bias=True
    )
    bn = mx.symbol.BatchNorm(
        name=name + '_bn',
        data=conv,
        fix_gamma=False,
        momentum=bn_momentum,
        use_global_stats=use_global_stats,
        # Same with https://github.com/soumith/cudnn.torch/blob/master/BatchNormalization.lua
        eps=eps
    )
    return (
        # It's better to remove ReLU here
        # https://github.com/gcr/torch-residual-networks
        mx.symbol.LeakyReLU(name=name + '_prelu', act_type='prelu', data=bn)
        if with_relu else bn
    )


def initila_block(data):
    conv = mx.symbol.Convolution(
        name="initial_conv",
        data=data,
        num_filter=13,
        kernel=(3, 3),
        stride=(2, 2),
        pad=(1, 1),
        no_bias=True
    )

    maxpool = mx.symbol.Pooling(data=data, pool_type="max", kernel=(2, 2), stride=(2, 2),
                                name="initial_maxpool")
    concat = mx.symbol.Concat(
        conv,
        maxpool,
        num_args=2,
        name="initial_concat"
    )
    return concat


def make_block(name, data, num_filter, bn_momentum,
               down_sample=False, up_sample=False,
               dilated=(1, 1), asymmetric=0):
    """maxpooling & padding"""
    if down_sample:
        # 1x1 conv ensures that channel equal to main branch
        maxpool = get_conv(name=name + '_proj_maxpool',
                           data=data,
                           num_filter=num_filter,
                           kernel=(2, 2),
                           pad=(0, 0),
                           with_relu=True,
                           bn_momentum=bn_momentum,
                           stride=(2, 2))

    elif up_sample:
        # maxunpooling.
        maxpool = mx.symbol.Deconvolution(name=name + '_unpooling',
                                   data=data,
                                   num_filter=num_filter,
                                   kernel=(4, 4),
                                   stride=(2, 2),
                                   pad=(1, 1))

        # Reference: https://github.com/e-lab/ENet-training/blob/master/train/models/decoder.lua
        # Padding is replaced by 1x1 convolution
        maxpool = get_conv(name=name + '_padding',
                           data=maxpool,
                           num_filter=num_filter,
                           kernel=(1, 1),
                           stride=(1, 1),
                           pad=(0, 0),
                           bn_momentum=bn_momentum,
                           with_relu=False)
    # main branch begin
    proj = get_conv(name=name + '_proj0',
                    data=data,
                    num_filter=num_filter,
                    kernel=(1, 1) if not down_sample else (2, 2),
                    stride=(1, 1) if not down_sample else (2, 2),
                    pad=(0, 0),
                    with_relu=True,
                    bn_momentum=bn_momentum)

    if up_sample:
        conv = mx.symbol.Deconvolution(name=name + '_deconv',
                                   data=proj,
                                   num_filter=num_filter,
                                   kernel=(4, 4),
                                   stride=(2, 2),
                                   pad=(1, 1))
    else:
        if asymmetric == 0:
            conv = get_conv(name=name + '_conv',
                            data=proj,
                            num_filter=num_filter,
                            kernel=(3, 3),
                            pad=dilated,
                            dilate=dilated,
                            stride=(1, 1),
                            with_relu=True,
                            bn_momentum=bn_momentum)
        else:
            conv = get_conv(name=name + '_conv1',
                            data=proj,
                            num_filter=num_filter,
                            kernel=(1, asymmetric),
                            pad=(0, asymmetric / 2),
                            stride=(1, 1),
                            dilate=dilated,
                            with_relu=True,
                            bn_momentum=bn_momentum)
            conv = get_conv(name=name + '_conv2',
                            data=conv,
                            num_filter=num_filter,
                            kernel=(asymmetric, 1),
                            pad=(asymmetric / 2, 0),
                            dilate=dilated,
                            stride=(1, 1),
                            with_relu=True,
                            bn_momentum=bn_momentum)

    regular = mx.symbol.Convolution(name=name + '_expansion',
                                        data=conv,
                                        num_filter=num_filter,
                                        kernel=(1, 1),
                                        pad=(0, 0),
                                        stride=(1, 1),
                                        no_bias=True)
    regular = mx.symbol.BatchNorm(
        name=name + '_expansion_bn',
        data=regular,
        fix_gamma=False,
        momentum=bn_momentum,
        use_global_stats=use_global_stats,
        eps=eps
    )
    # main branch end
    # TODO: spatial dropout

    if down_sample or up_sample:
        regular = mx.symbol.ElementWiseSum(maxpool, regular, name =  name + "_plus")
    else:
        regular = mx.symbol.ElementWiseSum(data, regular, name =  name + "_plus")
    regular = mx.symbol.LeakyReLU(name=name + '_expansion_prelu', act_type='prelu', data=regular)
    return regular


def get_enet_conv(data, bn_momentum=0.9):
    ##level 0
    data = initila_block(data)  # 16

    ##level 1
    num_filter = 64
    data = data0 = make_block(name="bottleneck1.0", data=data, num_filter=num_filter,
                              bn_momentum=bn_momentum, down_sample=True, up_sample=False)  # 64
    for block in range(4):
        data = make_block(name='bottleneck1.%d' % (block + 1),
                          data=data, num_filter=num_filter,  bn_momentum=bn_momentum,
                          down_sample=False, up_sample=False)
    data0 = make_block(name="projection1", data=data0, num_filter=num_filter, bn_momentum=bn_momentum)
    data = mx.symbol.ElementWiseSum(*[data, data0], name="level1")
    ##level 2
    num_filter = 128
    data = data0 = make_block(name="bottleneck2.0", data=data, num_filter=num_filter,
                              bn_momentum=bn_momentum, down_sample=True, up_sample=False)
    data = make_block(name="bottleneck2.1", data=data, num_filter=num_filter, bn_momentum=bn_momentum)
    data = make_block(name="bottleneck2.2", data=data, num_filter=num_filter, bn_momentum=bn_momentum,
                      dilated=(2, 2))
    data = make_block(name="bottleneck2.3", data=data, num_filter=num_filter, bn_momentum=bn_momentum,
                      asymmetric=5)
    data = make_block(name="bottleneck2.4", data=data, num_filter=num_filter, bn_momentum=bn_momentum,
                      dilated=(4, 4))
    data = make_block(name="bottleneck2.5", data=data, num_filter=num_filter, bn_momentum=bn_momentum)
    data = make_block(name="bottleneck2.6", data=data, num_filter=num_filter, bn_momentum=bn_momentum,
                      dilated=(8, 8))
    data = make_block(name="bottleneck2.7", data=data, num_filter=num_filter, bn_momentum=bn_momentum,
                      asymmetric=5)
    data = make_block(name="bottleneck2.8", data=data, num_filter=num_filter, bn_momentum=bn_momentum,
                      dilated=(16, 16))

    data = make_block(name="bottleneck2.9", data=data, num_filter=num_filter, bn_momentum=bn_momentum)

    data = make_block(name="bottleneck2.10", data=data, num_filter=num_filter, bn_momentum=bn_momentum,
                      dilated=(32, 32))
    data0 = make_block(name="projection2", data=data0, num_filter=num_filter, bn_momentum=bn_momentum)
    data = mx.symbol.ElementWiseSum(*[data, data0], name="level2")
    #level 3
    num_filter = 128
    data = data0 = make_block(name="bottleneck3.1", data=data, num_filter=num_filter,
                              bn_momentum=bn_momentum)
    data = make_block(name="bottleneck3.2", data=data, num_filter=num_filter, bn_momentum=bn_momentum,
                      dilated=(2, 2))
    data = make_block(name="bottleneck3.3", data=data, num_filter=num_filter, bn_momentum=bn_momentum,
                      asymmetric=5)
    data = make_block(name="bottleneck3.4", data=data, num_filter=num_filter, bn_momentum=bn_momentum,
                      dilated=(4, 4))
    data = make_block(name="bottleneck3.5", data=data, num_filter=num_filter, bn_momentum=bn_momentum)
    data = make_block(name="bottleneck3.6", data=data, num_filter=num_filter, bn_momentum=bn_momentum,
                      dilated=(8, 8))
    data = make_block(name="bottleneck3.7", data=data, num_filter=num_filter, bn_momentum=bn_momentum,
                      asymmetric=5)
    data = make_block(name="bottleneck3.8", data=data, num_filter=num_filter, bn_momentum=bn_momentum,
                      dilated=(16, 16))
    data = make_block(name="bottleneck3.9", data=data, num_filter=num_filter, bn_momentum=bn_momentum)

    data = make_block(name="bottleneck3.10", data=data, num_filter=num_filter, bn_momentum=bn_momentum,
                      dilated=(32, 32))
    data0 = make_block(name="projection3", data=data0, num_filter=num_filter, bn_momentum=bn_momentum)
    data = mx.symbol.ElementWiseSum(*[data, data0], name="level3")

    return data


def get_enet_train(num_classes=config.NUM_CLASSES, num_anchors=config.NUM_ANCHORS):
    data = mx.symbol.Variable(name="data")
    im_info = mx.symbol.Variable(name="im_info")
    gt_boxes = mx.symbol.Variable(name="gt_boxes")
    rpn_label = mx.symbol.Variable(name='label')
    rpn_bbox_target = mx.symbol.Variable(name='bbox_target')
    rpn_bbox_weight = mx.symbol.Variable(name='bbox_weight')

    # shared convolutional layers
    conv_feat = get_enet_conv(data)

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
        name='roi_pool', data=conv_feat, rois=rois, pooled_size=(7, 7), spatial_scale=1.0 / config.RCNN_FEAT_STRIDE)
    if config.RCNN_CTX_WINDOW:
        roi_pool_ctx = mx.symbol.ROIPooling(
            name='roi_pool_ctx', data=conv_feat, rois=rois, pooled_size=(7, 7),
            spatial_scale=1.0 / config.RCNN_FEAT_STRIDE, pad=0.25)
        roi_pool_concat = mx.symbol.Concat(roi_pool, roi_pool_ctx, name='roi_pool_concat')
        roi_pool_red = mx.symbol.Convolution(
            data=roi_pool_concat, num_filter=256, kernel=(1, 1), stride=(1, 1), name='roi_pool_ctx_red')
        roi_pool = mx.symbol.Activation(data=roi_pool_red, act_type='relu', name='roi_pool_relu')

    # fc6
    flatten = mx.symbol.Flatten(data=roi_pool, name="flatten")
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
    cls_prob = mx.symbol.SoftmaxOutput(name='cls_prob', data=cls_score, label=label, normalization='batch',
                                       is_hidden_layer=config.TRAIN.RCNN_OHEM)

    # bounding box regression
    bbox_pred = mx.symbol.FullyConnected(name='bbox_pred', data=relu7, num_hidden=num_classes * 4)
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


def get_enet_test(num_classes=config.NUM_CLASSES, num_anchors=config.NUM_ANCHORS):
    data = mx.symbol.Variable(name="data")
    im_info = mx.symbol.Variable(name="im_info")

    # shared convolutional layers
    conv_feat = get_enet_conv(data)

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
            data=roi_pool_concat, num_filter=256, kernel=(1, 1), stride=(1, 1), name='roi_pool_ctx_red')
        roi_pool = mx.symbol.Activation(data=roi_pool_red, act_type='relu', name='roi_pool_relu')

    # fc6
    flatten = mx.symbol.Flatten(data=roi_pool, name="flatten")
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
