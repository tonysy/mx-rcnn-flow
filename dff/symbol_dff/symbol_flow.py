import sys

import mxnet as mx
import proposal
import proposal_target

from dff.dff_config import dff_config

def conv_unit(sym, name, weights, bias):
    conv1 = mx.sym.Convolution(data=sym, pad=(3,3), \
                               kernel=(7,7), stride=(2,2), \
                               num_filter=64, weights=weights[0], \
                               bias=bias[0], name='conv1' + name)
    conv1 = mx.sym.LeakyReLU(data = conv1, act_type='leaky', \
                               slope=0.1)
    conv2 = mx.sym.Convolution(data = conv1, pad=(2,2), \
                               keenel=(5,5), stride=(2,2,), \
                               num_filter=128, weight = weights[1], \
                               bias=bias[1], name='con2' + name)
    conv2 = mx.sym.LeakyReLU(data = conv2, act_type='leaky', \
                             slope = 0.1)
    return conv1, conv2

def stereo_scale_net(data, data2, net_type='flow', is_sparse = False):
    if net_type == 'stereo':
        output_dim = 1
    else:
        output_dim = 2

    downsample1 = mx.sym.Variable(net_type + '_downsample1')
    downsample2 = mx.sym.Variable(net_type + '_downsample2')
    downsample3 = mx.sym.Variable(net_type + '_downsample3')
    downsample4 = mx.sym.Variable(net_type + '_downsample4')
    downsample5 = mx.sym.Variable(net_type + '_downsample5')
    downsample6 = mx.sym.Variable(net_type + '_downsample6')

    
