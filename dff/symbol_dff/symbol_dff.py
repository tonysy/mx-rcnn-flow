import mxnet as mx
import proposal
import proposal_target
from dff.dff_config import dff_config
from dff.symbol_dff.symbol_flow import stereo_scale_net

def get_vgg_train_flow(num_classes=dff_config.NUM_CLASSES,\
                       num_anchors=dff_config.NUM_ANCHORS):
    data = mx.symbol.Variable(name="data")
    data2 = mx.symbol.Variable(name="data2")

    im_info = mx.symbol.Variable(name="im_info")
    gt_boxes = mx.symbol.Variable(name="im_info")
    rpn_label = mx.symbol.Variable(name="label")
    rpn_bbox_target = mx.symbol.Variable(name="bbox_target")
    rpn_bbox_inside_weight = mx.symbol.Variable(name="bbox_inside_weight")
    rpn_bbox_outside_weight = mx.symbol.Variable(name="bbox_outside_weight")

    # shared  convolutional layers
    relu5_3 = get_vgg_dilate_conv(data2)

    # ===== flownet
    flownet = stereo_scale_net(data * dff_config.TRAIN.FLOW_INPUT_FACTOR, \
                               data2 * dff_config.TRAIN.FLOW_INPUT_FACTOR, \
                               net_type='flow')

    flow = flownet[0]
    scale = flownet[1]
    scale_avg = mx.sym.Pooling(data=scale*0.125, pool_type='avg', kernel=(8,8),\
                               stride=(8,8), name='scale_avg')
    flow_avg = mx.sym.Pooling(data=flow*0.125, pool_type='avg', kernel=(8,8), \
                              stride=(8,8), name='flow_avg')

    # ====== warp feature with optical flow
    flow_grid = mx.symbol.GridGenerator(data=flow_avg, transform_type='warp',\
                                   name='flow_grid')
    warp_res = mx.symbol.Bilinearsampler(data=relu5_3, grid=flow_grid, \
                                         name='warp_res')

    # flow_transpose = mx.sym.transpose(data=flow_avg, axes=(0,2,3,1), \
    #                                   name='flow_transpose')
    # relu5_3_transpose = mx.sym.transpose(data=relu5_3, axes=(0,2,3,1), \
    #                                      name='relu5_3_transpose')

    # warp_res = mx.sym.Warp(data=relu5_3_transpose, grid=flow_transpose, \
    #                        name='warp')
    # warp_transpose = mx.sym.transpose(data=warp_res, axes=(0,3,1,2), \
    #                                   name="warp_transpose")
    # relu5_3 = warp_transpose * scale_avg

    relu5_3 = warp_res * scale_avg


    # RPN layers
    rpn_conv = mx.symbol.Convolution(data=relu5_3, kernel=(3,3), pad=(1,1), \
                                     num_filter=512, name='rpn_conv_3x3')
    rpn_relu = mx.symbol.Activation(data=rpn_conv, act_type="relu", \
                                    name="rpn_relu")
    rpn_cls_score = mx.symbol.Convolution(data=rpn_relu,kernel=(1,1),pad=(0,0),\
                                          num_filter=2*num_anchors,\
                                          name = 'rpn_cls_score')
    rpn_bbox_pred = mx.symbol.Convolution(data=rpn_relu,kernel=(1,1),pad=(0,0),\
                                          num_filter=4*num_anchors, \
                                          name = 'rpn_bbox_pred')

    # prepare rpn data
    rpn_cls_score_reshape = mx.symbol.Reshape(data=rpn_cls_score, shape=(0,2,-1,0),\
                                              name='rpn_cls_score_reshape')

    # classification
    rpn_cls_prob = mx.symbol.SoftmaxOutput(data=rpn_cls_score_reshape,\
                                           label=rpn_label, multi_output=True, \
                                           normalization='valid', use_ignore=True,\
                                           ignore_label=-1, name="rpn_cls_prob")
    # bounding box regression
    rpn_bbox_loss_ = rpn_bbox_outside_weight * \
                    mx.symbol.smooth_l1(name='rpn_bbox_loss_', scalar=3.0, \
                                        data=rpn_bbox_inside_weight * \
                                        (rpn_bbox_pred - rpn_bbox_target))
    rpn_bbox_loss = mx.sym.MakeLoss(name='rpn_bbox_loss', data=rpn_bbox_loss_)

    # ROI proposal
    rpn_cls_act = mx.symbol.SoftmaxActivation(
        data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_act")
    rpn_cls_act_reshape = mx.symbol.Reshape(
        data=rpn_cls_act, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_act_reshape')

    if dff_config.TRAIN.CXX_PROPOSAL:
        rois = mx.symbol.Proposal(
            cls_prob=rpn_cls_act_reshape, bbox_pred=rpn_bbox_pred,
            im_info=im_info, name='rois',
            feature_stride=dff_config.RPN_FEAT_STRIDE,
            scales=tuple(dff_config.ANCHOR_SCALES),
            ratios=tuple(dff_config.ANCHOR_RATIOS),
            rpn_pre_nms_top_n=dff_config.TRAIN.RPN_PRE_NMS_TOP_N,
            rpn_post_nms_top_n=dff_config.TRAIN.RPN_POST_NMS_TOP_N,
            threshold=dff_config.TRAIN.RPN_NMS_THRESH,
            rpn_min_size=dff_config.TRAIN.RPN_MIN_SIZE)
    else:
        rois=mx.symbol.Custom(
            cls_prob=rpn_cls_act_reshape, bbox_pred=rpn_bbox_pred, \
            im_info=im_info, name='rois', op_type='proposal', \
            feat_stride=dff_config.RPN_FEAT_STRIDE,
            scales=tuple(dff_config.ANCHOR_SCALES), \
            ratios=tuple(dff_config.ANCHOR_RATIOS),
            rpn_pre_nms_top_n=dff_config.TRAIN.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=dff_config.TRAIN.RPN_POST_NMS_TOP_N,
            threshold=dff_config.TRAIN.RPN_NMS_THRESH, \
            rpn_min_size=dff_config.TRAIN.RPN_MIN_SIZE)

    # ROI proposal target
    gt_boxes_reshape = mx.symbol.Reshape(data=gt_boxes, shape=(-1,5),
                                         name='gt_boxes_reshape')
    group = mx.symbol.Custom(rois=rois, gt_boxes_reshape=gt_boxes_reshape, \
                             op_type='proposal_target', num_classes=num_classes,
                             batch_images=dff_config.TRAIN.BATCH_IMAGES,
                             batch_rois=dff_config.TRAIN.BATCH_ROIS,
                             fg_fraction=dff_config.TRAIN.FG_FRACTION)

    rois = group[0]
    label = group[1]
    bbox_target = group[2]
    bbox_inside_weight = group[3]
    bbox_outside_weight = group[4]

    # Fast R-CNN
    pool5 = mx.symbol.ROIPooling(
        name='roi_pool5', data=relu5_3, rois=rois, pooled_size=(7, 7), spatial_scale=1.0 / dff_config.RCNN_FEAT_SRTIDE)
    # group 6
    flatten = mx.symbol.Flatten(data=pool5, name="flatten")
    fc6 = mx.symbol.FullyConnected(data=flatten, num_hidden=4096, name="fc6")
    relu6 = mx.symbol.Activation(data=fc6, act_type="relu", name="relu6")
    drop6 = mx.symbol.Dropout(data=relu6, p=0.5, name="drop6")
    # group 7
    fc7 = mx.symbol.FullyConnected(data=drop6, num_hidden=4096, name="fc7")
    relu7 = mx.symbol.Activation(data=fc7, act_type="relu", name="relu7")
    drop7 = mx.symbol.Dropout(data=relu7, p=0.5, name="drop7")
    # classification
    cls_score = mx.symbol.FullyConnected(name='cls_score', data=drop7, num_hidden=num_classes)
    cls_prob = mx.symbol.SoftmaxOutput(name='cls_prob', data=cls_score, label=label, normalization='batch')
    # bounding box regression
    bbox_pred = mx.symbol.FullyConnected(name='bbox_pred', data=drop7, num_hidden=num_classes * 4)
    bbox_loss_ = bbox_outside_weight * \
                 mx.symbol.smooth_l1(name='bbox_loss_', scalar=1.0,
                                     data=bbox_inside_weight * (bbox_pred - bbox_target))
    bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_, grad_scale=1.0 / dff_config.TRAIN.BATCH_ROIS)

    # reshape output
    label = mx.symbol.Reshape(data=label, shape=(dff_config.TRAIN.BATCH_IMAGES, -1), name='label_reshape')
    cls_prob = mx.symbol.Reshape(data=cls_prob, shape=(dff_config.TRAIN.BATCH_IMAGES, -1, num_classes), name='cls_prob_reshape')
    bbox_loss = mx.symbol.Reshape(data=bbox_loss, shape=(dff_config.TRAIN.BATCH_IMAGES, -1, 4 * num_classes), name='bbox_loss_reshape')

    group = mx.symbol.Group([rpn_cls_prob, rpn_bbox_loss, cls_prob, bbox_loss,
        mx.symbol.BlockGrad(label)])  #, mx.symbol.BlockGrad(flow_pr)
    return group

def get_vgg_dilate_conv(data):
    """
    vgg16
    use dilated convolution in group 5
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
    relu3_1 = mx.symbol.Convolution(data=conv3_1, act_type='relu', \
                                    name='relu3_1')

    conv3_2 = mx.symbol.Convolution(data=relu3_1, kernel=(3,3), pad=(1,1), \
                                    num_filter=256, name='conv3_2')
    relu3_2 = mx.symbol.Activation(data=conv3_2, act_type='relu', \
                                   name='relu3_2')

    conv3_3 = mx.symbol.Convolution(data=relu3_2, kernel=(3,3), pad=(1,1), \
                                    num_filter=256, name="conv3_3")
    relu3_3 = mx.symbol.Activation(data=relu3_3, act_type='relu', \
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
    relu4_2 = mx.symbol.Convolution(data=conv4_1,act_type='relu',\
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
