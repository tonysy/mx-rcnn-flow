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

    flow_transpose = mx.sym.transpose(data=flow_avg, axes=(0,2,3,1), \
                                      name='flow_transpose')
    relu5_3_transpose = mx.sym.transpose(data=relu5_3, axes=(0,2,3,1), \
                                         name='relu5_3_transpose')

    warp_res = mx.sym.Warp(data=relu5_3_transpose, grid=flow_transpose, \
                           name='warp')
    warp_transpose = mx.sym.transpose(data=warp_res, axes=(0,3,1,2), \
                                      name="warp_transpose")
    relu5_3 = warp_transpose * scale_avg

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
