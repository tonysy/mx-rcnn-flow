# from ..config import config
from symbol_flownets import stereo_scale_net_share, conv_unit_share
import mxnet as mx
def feature_propagate_share(return_name, param_dic, feature_input, data, data2):
    """
    Get feature propagated with Optical Flow
    :param param_dic: Symbol Dict(share Parameters)
    :param feature_input: Symbol(reference feature for propagate)
    :param data: Symbol(image_01)
    :param data2: Symbol(image_02)
    :return: Symbol(new feature propagated use flow)
    """
    flownet = stereo_scale_net_share(data*config.FLOW_SCALE_FACTOR, \
                               data2*config.FLOW_SCALE_FACTOR,\
                               param_dic=param_dic,
                               net_type='flow')
    flow = flownet[0]
    scale = flownet[1]
    scale_avg = mx.sym.Pooling(data=scale*0.125, pool_type='avg',\
                               kernel=(8,8),stride=(8,8),name="scale_avg")
    flow_avg = mx.sym.Pooling(data=flow*0.125, pool_type='avg',\
                               kernel=(8,8),stride=(8,8),name="flow_avg")

    flow_grid = mx.symbol.GridGenerator(data=flow_avg,transform_type='warp',\
                                        name='flow_grid')
    warp_res = mx.symbol.BilinearSampler(data=relu5_3,grid=flow_grid,\
                                         name='warp_res')

    relu5_3_ = warp_res * scale_avg
    # relu5_3_ = mx.symbol.broadcast_mul(lhs=warp_res, rhs=scale_avg)
    # print "feature propagate:", relu5_3_.list_arguments()
    return relu5_3_, flow, flow_avg


def get_loss(data, label, loss_scale, name, get_input=False, is_sparse = False, type='stereo'):

    if type == 'stereo':
        data = mx.sym.Activation(data=data, act_type='relu',name=name+'relu')
    # loss
    if  is_sparse:
        loss =mx.symbol.Custom(data=data, label=label, name=name, loss_scale= loss_scale, is_l1=True,
            op_type='SparseRegressionLoss')
    else:
        loss = mx.sym.MAERegressionOutput(data=data, label=label, name=name, grad_scale=loss_scale)

    return (loss,data) if get_input else loss
