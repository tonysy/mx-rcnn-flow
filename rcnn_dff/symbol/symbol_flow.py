"""
Optical flow for deep feature flow
"""
import sys
import mxnet as mx
import proposal
import proposal_target

from ..config import config


def conv_unit(sym, name, weights, bias):
    conv1 = mx.sym.Convolution(data=sym,pad=(3, 3), kernel=(7, 7),stride=(2, 2),num_filter=64,
                               weight=weights[0], bias=bias[0], name='conv1' + name)
    conv1 = mx.sym.LeakyReLU(data = conv1,  act_type = 'leaky', slope  = 0.1 ) # WHY leakyrelu?
    conv2 = mx.sym.Convolution(data = conv1, pad  = (2,2),  kernel=(5,5),stride=(2,2),num_filter=128,
                                 weight = weights[1], bias = bias[1], name='conv2' + name)
    conv2 = mx.sym.LeakyReLU(data = conv2, act_type = 'leaky', slope = 0.1)
    return conv1,conv2,

def conv_unit_share(sym, name, param_dic):
    conv1 = mx.sym.Convolution(data=sym,pad=(3, 3), kernel=(7, 7),stride=(2, 2),num_filter=64,
                               weight=param_dic['share1_weight'], bias=param_dic['share1_bias'], name='conv1' + name)
    conv1 = mx.sym.LeakyReLU(data = conv1,  act_type = 'leaky', slope  = 0.1 ) # WHY leakyrelu?
    conv2 = mx.sym.Convolution(data = conv1, pad  = (2,2),  kernel=(5,5),stride=(2,2),num_filter=128,
                                 weight = param_dic['share2_weight'], bias = param_dic['share2_bias'], name='conv2' + name)
    conv2 = mx.sym.LeakyReLU(data = conv2, act_type = 'leaky', slope = 0.1)
    return conv1,conv2,

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

    weights = [mx.sym.Variable('share{}_weight'.format(i)) for i in range(1,4)]
    bias    = [mx.sym.Variable('share{}_bias'.format(i)) for i in range(1,4)]

    conv1_img1, conv2_img1 = conv_unit(data, 'img1', weights, bias)
    conv1_img2, conv2_img2 = conv_unit(data2, 'img2', weights, bias)

    if net_type =='stereo':
        corr = mx.sym.Correlation1D(data1=conv2_img1, data2=conv2_img2, \
                                    pad_size=40, kernel_size=1, \
                                    max_displacement=40, stride1=1, stride2=1)

        conv_redir = mx.sym.Convolution(data=conv2_img1, pad=(0, 0),  \
                                        kernel=(1, 1), stride=(1, 1), \
                                        num_filter=64, name='conv_redir')

        conv_redir = mx.sym.LeakyReLU(data=conv_redir, act_type='leaky', \
                                      slope=0.1)
        concat = mx.sym.Concat(corr, conv_redir)
    else:
        conv3_img1 = mx.sym.Convolution(data=conv2_img1, pad=(2, 2), \
                                        kernel=(5, 5), stride=(2, 2), \
                                        num_filter=256, weight=weights[2], \
                                        bias=bias[2], name='conv3_img1')
        conv3_img1 = mx.sym.LeakyReLU(data=conv3_img1, act_type='leaky', \
                                      slope=0.1)

        conv3_img2 = mx.sym.Convolution(data=conv2_img2, pad=(2, 2), \
                                        kernel=(5, 5), stride=(2, 2), \
                                        num_filter=256, weight=weights[2], \
                                        bias=bias[2], name='conv3_img2')
        conv3_img2 = mx.sym.LeakyReLU(data=conv3_img2, act_type='leaky', \
                                      slope=0.1)

        corr = mx.sym.Correlation(data1=conv3_img1, data2=conv3_img2, \
                                  pad_size=20, kernel_size=1, \
                                  max_displacement=20, stride1=1, stride2=2)

        conv_redir = mx.sym.Convolution(data=conv3_img1, pad=(0, 0), \
                                        kernel=(1, 1), stride=(1, 1), \
                                        num_filter=64, name='conv_redir')
        conv_redir = mx.sym.LeakyReLU(data=conv_redir, act_type='leaky', \
                                      slope=0.1)
        concat = mx.sym.Concat(corr, conv_redir)

    if net_type =='stereo':
        stride = (2,2)
    else:
        stride = (1,1)
    conv3a = mx.sym.Convolution(concat, pad=(2, 2), kernel=(5, 5), \
                                stride=stride, num_filter=256, name='conv3a')
    conv3a = mx.sym.LeakyReLU(data=conv3a, act_type='leaky', slope=0.1)

    conv3b = mx.sym.Convolution(conv3a, pad=(1, 1), kernel=(3, 3), \
                                stride=(1, 1), num_filter=256, name='conv3b')
    conv3b = mx.sym.LeakyReLU(data=conv3b, act_type='leaky', slope=0.1)

    conv4a = mx.sym.Convolution(conv3b, pad=(1, 1), kernel=(3, 3), \
                                stride=(2, 2), num_filter=512, name='conv4a')
    conv4a = mx.sym.LeakyReLU(data=conv4a, act_type='leaky', slope=0.1)

    conv4b = mx.sym.Convolution(conv4a, pad=(1, 1), kernel=(3, 3), \
                                stride=(1, 1), num_filter=512, name='conv4b')
    conv4b = mx.sym.LeakyReLU(data=conv4b, act_type='leaky', slope=0.1)

    conv5a = mx.sym.Convolution(conv4b, pad=(1, 1), kernel=(3, 3), \
                                stride=(2, 2), num_filter=512, name='conv5a')
    conv5a = mx.sym.LeakyReLU(data=conv5a, act_type='leaky', slope=0.1)

    conv5b = mx.sym.Convolution(conv5a, pad=(1, 1), kernel=(3, 3), \
                                stride=(1, 1), num_filter=512, name='conv5b')
    conv5b = mx.sym.LeakyReLU(data=conv5b, act_type='leaky', slope=0.1)

    conv6a = mx.sym.Convolution(conv5b, pad=(1, 1), kernel=(3, 3), \
                                stride=(2, 2), num_filter=1024, name='conv6a')
    conv6a = mx.sym.LeakyReLU(data=conv6a, act_type='leaky', slope=0.1)

    conv6b = mx.sym.Convolution(conv6a, pad=(1, 1), kernel=(3, 3), \
                                stride=(1, 1), num_filter=1024, name='conv6b')
    conv6b = mx.sym.LeakyReLU(data=conv6b, act_type='leaky', slope=0.1, )

    pr6 = mx.sym.Convolution(conv6b,pad= (1,1),kernel=(3,3),stride=(1,1),\
                             num_filter=output_dim,name='pr6')

    upsample_pr6to5 = mx.sym.Deconvolution(pr6, pad=(1,1), kernel=(4,4), \
                                           stride=(2,2), num_filter=1, \
                                           name='upsample_pr6to5',no_bias=True)
    upconv5 = mx.sym.Deconvolution(conv6b,pad=(1,1),kernel=(4,4), stride=(2,2),\
                                   num_filter=512,name='upconv5',no_bias=True)
    upconv5 = mx.sym.LeakyReLU(data = upconv5,act_type = 'leaky',slope  = 0.1)
    concat_tmp = mx.sym.Concat(conv5b,upconv5,upsample_pr6to5,dim=1)

    iconv5 = mx.sym.Convolution(concat_tmp,pad = (1,1),kernel=(3,3),\
                                stride=(1,1),num_filter = 512,name='iconv5')

    pr5  = mx.sym.Convolution(iconv5, pad = (1,1),kernel=(3,3),stride=(1,1),\
                              num_filter = output_dim,name='pr5')

    upconv4 = mx.sym.Deconvolution(iconv5,pad = (1,1),kernel= (4,4),\
                                   stride = (2,2),num_filter=256,\
                                   name='upconv4',no_bias=True)
    upconv4 = mx.sym.LeakyReLU(data = upconv4,act_type = 'leaky',slope  = 0.1 )

    upsample_pr5to4 = mx.sym.Deconvolution(pr5,pad = (1,1),kernel= (4,4), \
                                           stride=(2,2),num_filter=1,\
                                           name='upsample_pr5to4',no_bias=True)

    concat_tmp2 = mx.sym.Concat(conv4b,upconv4,upsample_pr5to4)
    iconv4  = mx.sym.Convolution(concat_tmp2,pad = (1,1),kernel = (3,3),\
                                 stride=(1,1),num_filter=256,name='iconv4')
    pr4 = mx.sym.Convolution(iconv4,pad=(1,1),kernel=(3,3),stride=(1,1),\
                             num_filter=output_dim,name='pr4')

    upconv3 = mx.sym.Deconvolution(iconv4,pad=(1,1),kernel=(4,4),stride=(2,2),\
                                   num_filter=128,name='upconv3',no_bias=True)
    upconv3 = mx.sym.LeakyReLU(data = upconv3,act_type = 'leaky',slope  = 0.1 )

    upsample_pr4to3 = mx.sym.Deconvolution(pr4,pad = (1,1),kernel= (4,4), \
                                           stride=(2,2), num_filter=1, \
                                           name='upsample_pr4to3',no_bias=True)
    concat_tmp3 = mx.sym.Concat(conv3b,upconv3,upsample_pr4to3)
    iconv3 = mx.sym.Convolution(concat_tmp3,pad=(1,1),kernel=(3,3), \
                                stride=(1,1),num_filter = 128,name='iconv3')
    pr3 = mx.sym.Convolution(iconv3,pad = (1,1), kernel = (3,3), \
                             stride = (1,1),num_filter = output_dim,name='pr3')


    upconv2 = mx.sym.Deconvolution(iconv3,pad=(1,1),kernel=(4,4),stride=(2,2),\
                                   num_filter=64,name='upconv2',no_bias=True)
    upconv2 = mx.sym.LeakyReLU(data = upconv2,act_type = 'leaky',slope  = 0.1)

    upsample_pr3to2 = mx.sym.Deconvolution(pr3,pad = (1,1),kernel= (4,4), \
                                           stride=(2,2),num_filter=1, \
                                           name='upsample_pr3to2',no_bias=True)

    concat_tmp4 = mx.sym.Concat(conv2_img1,upconv2,upsample_pr3to2)

    iconv2 = mx.sym.Convolution(concat_tmp4,pad = (1,1),kernel = (3,3),\
                                stride= (1,1),num_filter = 64,name='iconv2')
    pr2 = mx.sym.Convolution(iconv2,pad = (1,1),kernel=(3,3),stride = (1,1),\
                             num_filter = output_dim,name='pr2')

    upconv1 = mx.sym.Deconvolution(iconv2,pad=(1,1),kernel=(4,4),stride=(2,2),
                                   num_filter = 32,name='upconv1',no_bias=True)
    upconv1 = mx.sym.LeakyReLU(data = upconv1,act_type = 'leaky',slope  = 0.1 )

    upsample_pr2to1 = mx.sym.Deconvolution(pr2,pad = (1,1),kernel= (4,4),stride=(2,2),num_filter=1,name='upsample_pr2to1',no_bias=True)

    concat_tmp5 = mx.sym.Concat(conv1_img1,upconv1,upsample_pr2to1)
    iconv1 = mx.sym.Convolution(concat_tmp5,pad=(1,1),kernel = (3,3),stride=(1,1),num_filter=32,name='iconv1')
    pr1 = mx.sym.Convolution(iconv1,pad=(1,1),kernel=(3,3),stride=(1,1), \
                             num_filter=output_dim,name='pr1')
    stereo_scale = mx.sym.Convolution(iconv1,pad=(1,1),kernel=(3,3),stride=(1,1), \
                                      num_filter=512,name='stereo_scale')

    img1 = mx.sym.BlockGrad(data=data,name='img1_tmp')
    img2 = mx.sym.BlockGrad(data=data2,name='img2_tmp')
    corr = mx.sym.BlockGrad(data=corr,name='corr')
    conv3_img1 = mx.sym.BlockGrad(data=conv3_img1,name='conv2_img1_tmp')
    conv3_img2 = mx.sym.BlockGrad(data=conv3_img2,name='conv2_img2_tmp')
    net = mx.sym.Group([pr1, stereo_scale])

    return net

def stereo_scale_net_share(data, data2, param_dic, net_type='flow', is_sparse = False):

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

    # weights = [mx.sym.Variable('share{}_weight'.format(i)) for i in range(1,4)]
    # bias    = [mx.sym.Variable('share{}_bias'.format(i)) for i in range(1,4)]

    conv1_img1, conv2_img1 = conv_unit_share(data, 'img1', param_dic)
    conv1_img2, conv2_img2 = conv_unit_share(data2, 'img2', param_dic)

    if net_type =='stereo':
        corr = mx.sym.Correlation1D(data1=conv2_img1, data2=conv2_img2, \
                                    pad_size=40, kernel_size=1, \
                                    max_displacement=40, stride1=1, stride2=1)

        conv_redir = mx.sym.Convolution(data=conv2_img1, pad=(0, 0),  \
                                        kernel=(1, 1), stride=(1, 1), \
                                        num_filter=64, name='conv_redir')

        conv_redir = mx.sym.LeakyReLU(data=conv_redir, act_type='leaky', \
                                      slope=0.1)
        concat = mx.sym.Concat(corr, conv_redir)
    else:
        conv3_img1 = mx.sym.Convolution(data=conv2_img1, pad=(2, 2), \
                                        kernel=(5, 5), stride=(2, 2), \
                                        num_filter=256, weight=param_dic['share3_weight'], \
                                        bias=param_dic['share3_bias'], name='conv3_img1')
        conv3_img1 = mx.sym.LeakyReLU(data=conv3_img1, act_type='leaky', \
                                      slope=0.1)

        conv3_img2 = mx.sym.Convolution(data=conv2_img2, pad=(2, 2), \
                                        kernel=(5, 5), stride=(2, 2), \
                                        num_filter=256, weight=param_dic['share3_weight'], \
                                        bias=param_dic['share3_bias'], name='conv3_img2')
        conv3_img2 = mx.sym.LeakyReLU(data=conv3_img2, act_type='leaky', \
                                      slope=0.1)

        corr = mx.sym.Correlation(data1=conv3_img1, data2=conv3_img2, \
                                  pad_size=20, kernel_size=1, \
                                  max_displacement=20, stride1=1, stride2=2)
        conv_redir = mx.sym.Convolution(data=conv3_img1, pad=(0, 0), \
                                        kernel=(1, 1), stride=(1, 1), \
                                        num_filter=64, name='conv_redir',
                                        weight=param_dic['conv_redir_weight'], \
                                        bias=param_dic['conv_redir_bias'])
        conv_redir = mx.sym.LeakyReLU(data=conv_redir, act_type='leaky', \
                                      slope=0.1)
        concat = mx.sym.Concat(corr, conv_redir)

    if net_type =='stereo':
        stride = (2,2)
    else:
        stride = (1,1)
    print param_dic['conv3a_weight']

    conv3a = mx.sym.Convolution(data=concat, pad=(2, 2), kernel=(5, 5), \
                                stride=stride, num_filter=256, name='conv3a',
                                weight=param_dic['conv3a_weight'],
                                bias=param_dic['conv3a_bias'])
    conv3a = mx.sym.LeakyReLU(data=conv3a, act_type='leaky', slope=0.1)

    conv3b = mx.sym.Convolution(data=conv3a, pad=(1, 1), kernel=(3, 3), \
                                stride=(1, 1), num_filter=256,
                                weight=param_dic['conv3b_weight'],
                                bias=param_dic['conv3b_bias'], name='conv3b')
    conv3b = mx.sym.LeakyReLU(data=conv3b, act_type='leaky', slope=0.1)

    conv4a = mx.sym.Convolution(data=conv3b, pad=(1, 1), kernel=(3, 3), \
                                stride=(2, 2), num_filter=512, name='conv4a',
                                weight=param_dic['conv4a_weight'], \
                                bias=param_dic['conv4a_bias'])
    conv4a = mx.sym.LeakyReLU(data=conv4a, act_type='leaky', slope=0.1)

    conv4b = mx.sym.Convolution(data=conv4a, pad=(1, 1), kernel=(3, 3), \
                                stride=(1, 1), num_filter=512, name='conv4b',
                                weight=param_dic['conv4b_weight'], \
                                bias=param_dic['conv4b_bias'])
    conv4b = mx.sym.LeakyReLU(data=conv4b, act_type='leaky', slope=0.1)

    conv5a = mx.sym.Convolution(data=conv4b, pad=(1, 1), kernel=(3, 3), \
                                stride=(2, 2), num_filter=512, name='conv5a',
                                weight=param_dic['conv5a_weight'], \
                                bias=param_dic['conv5a_bias'])
    conv5a = mx.sym.LeakyReLU(data=conv5a, act_type='leaky', slope=0.1)

    conv5b = mx.sym.Convolution(data=conv5a, pad=(1, 1), kernel=(3, 3), \
                                stride=(1, 1), num_filter=512, name='conv5b',
                                weight=param_dic['conv5b_weight'], \
                                bias=param_dic['conv5b_bias'])
    conv5b = mx.sym.LeakyReLU(data=conv5b, act_type='leaky', slope=0.1)

    conv6a = mx.sym.Convolution(data=conv5b, pad=(1, 1), kernel=(3, 3), \
                                stride=(2, 2), num_filter=1024, name='conv6a',
                                weight=param_dic['conv6a_weight'], \
                                bias=param_dic['conv6a_bias'])
    conv6a = mx.sym.LeakyReLU(data=conv6a, act_type='leaky', slope=0.1)

    conv6b = mx.sym.Convolution(data=conv6a, pad=(1, 1), kernel=(3, 3), \
                                stride=(1, 1), num_filter=1024, name='conv6b',
                                weight=param_dic['conv6b_weight'], \
                                bias=param_dic['conv6b_bias'])
    conv6b = mx.sym.LeakyReLU(data=conv6b, act_type='leaky', slope=0.1, )

    pr6 = mx.sym.Convolution(data=conv6b,pad= (1,1),kernel=(3,3),stride=(1,1),\
                             num_filter=output_dim,name='pr6',
                             weight=param_dic['pr6_weight'], \
                             bias=param_dic['pr6_bias'])

    upsample_pr6to5 = mx.sym.Deconvolution(data=pr6, pad=(1,1), kernel=(4,4), \
                                           stride=(2,2), num_filter=1, \
                                           name='upsample_pr6to5',no_bias=True,
                                           weight=param_dic['upsample_pr6to5_weight'])
    upconv5 = mx.sym.Deconvolution(data=conv6b,pad=(1,1),kernel=(4,4), stride=(2,2),\
                                   num_filter=512,name='upconv5',no_bias=True,
                                   weight = param_dic['upconv5_weight'])
    upconv5 = mx.sym.LeakyReLU(data = upconv5,act_type = 'leaky',slope  = 0.1)
    concat_tmp = mx.sym.Concat(conv5b,upconv5,upsample_pr6to5,dim=1)

    iconv5 = mx.sym.Convolution(data=concat_tmp,pad = (1,1),kernel=(3,3),\
                                stride=(1,1),num_filter = 512,name='iconv5',
                                weight = param_dic['iconv5_weight'],
                                bias=param_dic['iconv5_bias'])

    pr5  = mx.sym.Convolution(data=iconv5, pad = (1,1),kernel=(3,3),stride=(1,1),\
                              num_filter = output_dim,name='pr5',
                              weight = param_dic['pr5_weight'],
                              bias=param_dic['pr5_bias'])

    upconv4 = mx.sym.Deconvolution(data=iconv5,pad = (1,1),kernel= (4,4),\
                                   stride = (2,2),num_filter=256,\
                                   name='upconv4',no_bias=True,
                                   weight = param_dic['upconv4_weight'])
    upconv4 = mx.sym.LeakyReLU(data = upconv4,act_type = 'leaky',slope  = 0.1 )

    upsample_pr5to4 = mx.sym.Deconvolution(data=pr5,pad = (1,1),kernel= (4,4), \
                                           stride=(2,2),num_filter=1,\
                                           name='upsample_pr5to4',no_bias=True,
                                           weight = param_dic['upsample_pr5to4_weight'])

    concat_tmp2 = mx.sym.Concat(conv4b,upconv4,upsample_pr5to4)
    iconv4  = mx.sym.Convolution(data=concat_tmp2,pad = (1,1),kernel = (3,3),\
                                 stride=(1,1),num_filter=256,name='iconv4',
                                 weight = param_dic['iconv4_weight'],
                                 bias=param_dic['iconv4_bias'])
    pr4 = mx.sym.Convolution(data=iconv4,pad=(1,1),kernel=(3,3),stride=(1,1),\
                             num_filter=output_dim,name='pr4',
                             weight = param_dic['pr4_weight'],
                             bias=param_dic['pr4_bias'])

    upconv3 = mx.sym.Deconvolution(data=iconv4,pad=(1,1),kernel=(4,4),stride=(2,2),\
                                   num_filter=128,name='upconv3',no_bias=True,
                                   weight = param_dic['upconv3_weight'])
    upconv3 = mx.sym.LeakyReLU(data = upconv3,act_type = 'leaky',slope  = 0.1 )

    upsample_pr4to3 = mx.sym.Deconvolution(data=pr4,pad = (1,1),kernel= (4,4), \
                                           stride=(2,2), num_filter=1, \
                                           name='upsample_pr4to3',no_bias=True,
                                           weight = param_dic['upsample_pr4to3_weight'])
    concat_tmp3 = mx.sym.Concat(conv3b,upconv3,upsample_pr4to3)
    iconv3 = mx.sym.Convolution(data=concat_tmp3,pad=(1,1),kernel=(3,3), \
                                stride=(1,1),num_filter = 128,name='iconv3',
                                weight = param_dic['iconv3_weight'],
                                bias=param_dic['iconv3_bias'])
    pr3 = mx.sym.Convolution(data=iconv3,pad = (1,1), kernel = (3,3), \
                             stride = (1,1),num_filter = output_dim,name='pr3',
                             weight = param_dic['pr3_weight'],
                             bias=param_dic['pr3_bias'])


    upconv2 = mx.sym.Deconvolution(data=iconv3,pad=(1,1),kernel=(4,4),stride=(2,2),\
                                   num_filter=64,name='upconv2',no_bias=True,
                                   weight = param_dic['upconv2_weight'])
    upconv2 = mx.sym.LeakyReLU(data = upconv2,act_type = 'leaky',slope  = 0.1)

    upsample_pr3to2 = mx.sym.Deconvolution(data=pr3,pad = (1,1),kernel= (4,4), \
                                           stride=(2,2),num_filter=1, \
                                           name='upsample_pr3to2',no_bias=True,
                                           weight = param_dic['upsample_pr3to2_weight'])

    concat_tmp4 = mx.sym.Concat(conv2_img1,upconv2,upsample_pr3to2)

    iconv2 = mx.sym.Convolution(data=concat_tmp4,pad = (1,1),kernel = (3,3),\
                                stride= (1,1),num_filter = 64,name='iconv2',
                                weight = param_dic['iconv2_weight'],
                                bias=param_dic['iconv2_bias'])
    pr2 = mx.sym.Convolution(data=iconv2,pad = (1,1),kernel=(3,3),stride = (1,1),\
                             num_filter = output_dim,name='pr2',
                             weight = param_dic['pr2_weight'],
                             bias=param_dic['pr2_bias'])

    upconv1 = mx.sym.Deconvolution(data=iconv2,pad=(1,1),kernel=(4,4),stride=(2,2),
                                   num_filter = 32,name='upconv1',no_bias=True,
                                   weight = param_dic['upconv1_weight'])
    upconv1 = mx.sym.LeakyReLU(data = upconv1,act_type = 'leaky',slope  = 0.1 )

    upsample_pr2to1 = mx.sym.Deconvolution(data=pr2,pad = (1,1),kernel= (4,4),stride=(2,2),num_filter=1,name='upsample_pr2to1',no_bias=True,
                        weight = param_dic['upsample_pr2to1_weight'])

    concat_tmp5 = mx.sym.Concat(conv1_img1,upconv1,upsample_pr2to1)
    iconv1 = mx.sym.Convolution(data=concat_tmp5,pad=(1,1),kernel = (3,3),stride=(1,1),num_filter=32,name='iconv1',
    weight = param_dic['iconv1_weight'],
    bias=param_dic['iconv1_bias'])
    pr1 = mx.sym.Convolution(data=iconv1,pad=(1,1),kernel=(3,3),stride=(1,1), \
                             num_filter=output_dim,name='pr1',
                             weight = param_dic['pr1_weight'],
                             bias=param_dic['pr1_bias'])
    stereo_scale = mx.sym.Convolution(data=iconv1,pad=(1,1),kernel=(3,3),stride=(1,1), \
                                      num_filter=512,name='stereo_scale',
                                      weight = param_dic['stereo_scale_weight'],
                                      bias=param_dic['stereo_scale_bias'])

    img1 = mx.sym.BlockGrad(data=data,name='img1_tmp')
    img2 = mx.sym.BlockGrad(data=data2,name='img2_tmp')
    corr = mx.sym.BlockGrad(data=corr,name='corr')
    conv3_img1 = mx.sym.BlockGrad(data=conv3_img1,name='conv2_img1_tmp')
    conv3_img2 = mx.sym.BlockGrad(data=conv3_img2,name='conv2_img2_tmp')
    net = mx.sym.Group([pr1, stereo_scale])

    return net


def flownets_half(img1, img2, loss_scale, net_type='stereo', is_sparse = False):

    name = 'flownets_half'

    # img1 = mx.sym.Variable('img1')
    # img2 = mx.sym.Variable('img2')

    if net_type == 'stereo':
        output_dim = 1
    elif net_type == 'flow':
        output_dim = 2

    # six loss functions with different output sizes
    labels = {'loss{}'.format(i): mx.sym.Variable('loss{}_label'.format(i)) for i in range(1, 7)}
    prediction = {}
    loss = []

    data = mx.sym.Concat(img1, img2)

    # The structure below is similar to VGG
    conv1 = mx.sym.Convolution(data, pad=(3, 3), kernel=(7, 7), stride=(2, 2), num_filter=32, name=name+'conv1')
    conv1 = mx.sym.LeakyReLU(data=conv1, act_type='leaky', slope=0.1)

    conv2 = mx.sym.Convolution(conv1, pad=(2, 2), kernel=(5, 5), stride=(2, 2), num_filter=64, name=name+'conv2')
    conv2 = mx.sym.LeakyReLU(data=conv2, act_type='leaky', slope=0.1)

    conv3a = mx.sym.Convolution(conv2, pad=(2, 2), kernel=(5, 5), stride=(2, 2), num_filter=128, name=name+'conv3a')
    conv3a = mx.sym.LeakyReLU(data=conv3a, act_type='leaky', slope=0.1)

    conv3b = mx.sym.Convolution(conv3a, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=128, name=name+'conv3b')
    conv3b = mx.sym.LeakyReLU(data=conv3b, act_type='leaky', slope=0.1)

    conv4a = mx.sym.Convolution(conv3b, pad=(1, 1), kernel=(3, 3), stride=(2, 2), num_filter=256, name=name+'conv4a')
    conv4a = mx.sym.LeakyReLU(data=conv4a, act_type='leaky', slope=0.1)

    conv4b = mx.sym.Convolution(conv4a, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=256, name=name+'conv4b')
    conv4b = mx.sym.LeakyReLU(data=conv4b, act_type='leaky', slope=0.1)

    conv5a = mx.sym.Convolution(conv4b, pad=(1, 1), kernel=(3, 3), stride=(2, 2), num_filter=256, name=name+'conv5a')
    conv5a = mx.sym.LeakyReLU(data=conv5a, act_type='leaky', slope=0.1)

    conv5b = mx.sym.Convolution(conv5a, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=256, name=name+'conv5b')
    conv5b = mx.sym.LeakyReLU(data=conv5b, act_type='leaky', slope=0.1)

    conv6a = mx.sym.Convolution(conv5b, pad=(1, 1), kernel=(3, 3), stride=(2, 2), num_filter=512, name=name+'conv6a')
    conv6a = mx.sym.LeakyReLU(data=conv6a, act_type='leaky', slope=0.1)

    conv6b = mx.sym.Convolution(conv6a, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=512, name=name+'conv6b')
    conv6b = mx.sym.LeakyReLU(data=conv6b, act_type='leaky', slope=0.1, )

    pr6 = mx.sym.Convolution(conv6b,pad= (1,1),kernel=(3,3),stride=(1,1),num_filter=output_dim,name=name+'pr6')
    prediction['loss6'] = pr6

    upsample_pr6to5 = mx.sym.Deconvolution(pr6, pad=(1,1), kernel=(4,4), stride=(2,2), num_filter=output_dim,
                                           name=name+'upsample_pr6to5',no_bias=True)
    upconv5 = mx.sym.Deconvolution(conv6b,pad=(1,1),kernel=(4,4),stride=(2,2),num_filter=512,name=name+'upconv5',no_bias=True)
    upconv5 = mx.sym.LeakyReLU(data = upconv5,act_type = 'leaky',slope  = 0.1)
    concat_tmp = mx.sym.Concat(conv5b,upconv5,upsample_pr6to5,dim=1)

    iconv5 = mx.sym.Convolution(concat_tmp,pad = (1,1),kernel=(3,3),stride=(1,1),num_filter = 512,name=name+'iconv5')

    pr5  = mx.sym.Convolution(iconv5, pad = (1,1),kernel=(3,3),stride=(1,1),num_filter = output_dim,name=name+'pr5')
    prediction['loss5'] = pr5

    upconv4 = mx.sym.Deconvolution(iconv5,pad = (1,1),kernel= (4,4),stride = (2,2),num_filter=256,name=name+'upconv4',no_bias=True)
    upconv4 = mx.sym.LeakyReLU(data = upconv4,act_type = 'leaky',slope  = 0.1 )

    upsample_pr5to4 = mx.sym.Deconvolution(pr5,pad = (1,1),kernel= (4,4),stride=(2,2),num_filter=output_dim,name=name+'upsample_pr5to4',no_bias=True)

    concat_tmp2 = mx.sym.Concat(conv4b,upconv4,upsample_pr5to4)
    iconv4  = mx.sym.Convolution(concat_tmp2,pad = (1,1),kernel = (3,3),stride=(1,1),num_filter=256,name=name+'iconv4')
    pr4 = mx.sym.Convolution(iconv4,pad=(1,1),kernel=(3,3),stride=(1,1),num_filter=output_dim,name=name+'pr4')
    prediction['loss4'] = pr4

    upconv3 = mx.sym.Deconvolution(iconv4,pad=(1,1),kernel=(4,4),stride=(2,2),num_filter=128,name=name+'upconv3',no_bias=True)
    upconv3 = mx.sym.LeakyReLU(data = upconv3,act_type = 'leaky',slope  = 0.1 )

    upsample_pr4to3 = mx.sym.Deconvolution(pr4,pad = (1,1),kernel= (4,4),stride=(2,2),num_filter=output_dim,name=name+'upsample_pr4to3',no_bias=True)
    concat_tmp3 = mx.sym.Concat(conv3b,upconv3,upsample_pr4to3)
    iconv3 = mx.sym.Convolution(concat_tmp3,pad=(1,1),kernel=(3,3),stride=(1,1),num_filter = 128,name=name+'iconv3')
    pr3 = mx.sym.Convolution(iconv3,pad = (1,1), kernel = (3,3), stride = (1,1),num_filter = output_dim,name=name+'pr3')
    prediction['loss3'] = pr3

    upconv2 = mx.sym.Deconvolution(iconv3,pad=(1,1),kernel=(4,4),stride=(2,2),num_filter=64,name=name+'upconv2',no_bias=True)
    upconv2 = mx.sym.LeakyReLU(data = upconv2,act_type = 'leaky',slope  = 0.1  )

    upsample_pr3to2 = mx.sym.Deconvolution(pr3,pad = (1,1),kernel= (4,4),stride=(2,2),num_filter=output_dim,name=name+'upsample_pr3to2',no_bias=True)
    concat_tmp4 = mx.sym.Concat(conv2, upconv2, upsample_pr3to2)
    iconv2 = mx.sym.Convolution(concat_tmp4,pad = (1,1),kernel = (3,3),stride= (1,1),num_filter = 64,name=name+'iconv2')
    pr2 = mx.sym.Convolution(iconv2,pad = (1,1),kernel=(3,3),stride = (1,1),num_filter = output_dim,name=name+'pr2')
    prediction['loss2'] = pr2

    upconv1 = mx.sym.Deconvolution(iconv2,pad=(1,1),kernel=(4,4),stride=(2,2),num_filter = 32,name=name+'upconv1',no_bias=True)
    upconv1 = mx.sym.LeakyReLU(data = upconv1,act_type = 'leaky',slope  = 0.1 )
    upsample_pr2to1 = mx.sym.Deconvolution(pr2,pad = (1,1),kernel= (4,4),stride=(2,2),num_filter=output_dim,name=name+'upsample_pr2to1',no_bias=True)
    concat_tmp5 = mx.sym.Concat(conv1, upconv1, upsample_pr2to1)
    iconv1 = mx.sym.Convolution(concat_tmp5,pad=(1,1),kernel = (3,3),stride=(1,1),num_filter=32,name=name+'iconv1')
    pr1 = mx.sym.Convolution(iconv1, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=output_dim,name=name+'pr1')
    prediction['loss1'] = pr1

    # scale layer
    stereo_scale = mx.sym.Convolution(iconv1, pad=(1,1), kernel=(3,3), \
                                      stride=(1,1), num_filter=512, \
                                      name='stereo_scale')


    # ignore the loss functions with loss scale of zero
    keys = loss_scale.keys()
    keys.sort()
    for key in keys:
        loss.append(get_loss(prediction[key], labels[key], loss_scale[key], name=key+name,
                             get_input=False, is_sparse = is_sparse, type=net_type))

    # net = mx.sym.Group(loss)
    net = mx.sym.Group([loss[0], stereo_scale])
    return net

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

def feature_propagate(relu5_3, data, data2):
    flownet = stereo_scale_net(data*config.FLOW_SCALE_FACTOR, \
                               data2*config.FLOW_SCALE_FACTOR,\
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
    return relu5_3_, flow, flow_avg

def feature_propagate_share(return_name, param_dic, relu5_3, data, data2):
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
    # print "+++++++++feature propagate:", relu5_3_.list_arguments()
    return relu5_3_, flow, flow_avg


def feature_propagate_share_half(return_name, param_dic, relu5_3, data, data2):
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
    # print "+++++++++feature propagate:", relu5_3_.list_arguments()
    return relu5_3_, flow, flow_avg

def feature_warp(relu5_3, data, data2):
    # flownet
    flownet = stereo_scale_net(data * 0.00392156, data2 * 0.00392156, net_type='flow')

    flow = flownet[0]
    scale = flownet[1]
    scale_avg = mx.sym.Pooling(
        data=scale*0.125, pool_type="avg", kernel=(8, 8), stride=(8, 8), name="flow_avg")
    flow_avg = mx.sym.Pooling(
        data=flow*0.125, pool_type="avg", kernel=(8, 8), stride=(8, 8), name="flow_avg")
    flow_transpose = mx.sym.transpose(
            data=flow_avg, axes=(0, 2, 3, 1), name="flow_transpose")

    relu5_3_transpose = mx.sym.transpose(
            data=relu5_3, axes=(0, 2, 3, 1), name="relu5_3_transpose")
    warp_res = mx.sym.Warp(
            data=relu5_3_transpose, grid=flow_transpose, name="warp")
    warp_transpose = mx.sym.transpose(
            data=warp_res, axes=(0, 3, 1, 2), name="warp_transpose")
    relu5_3_ = warp_transpose * scale_avg

    return relu5_3_, flow, flow_avg

def stereo_net(data, data2, net_type='flow', is_sparse = False):

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

    weights = [mx.sym.Variable('share{}_weight'.format(i)) for i in range(1,4)]
    bias    = [mx.sym.Variable('share{}_bias'.format(i)) for i in range(1,4)]

    conv1_img1, conv2_img1 = conv_unit(data, 'img1', weights, bias)
    conv1_img2, conv2_img2 = conv_unit(data2, 'img2', weights, bias)

    if net_type =='stereo':
        corr = mx.sym.Correlation1D(data1=conv2_img1, data2=conv2_img2, \
                                    pad_size=40, kernel_size=1, \
                                    max_displacement=40, stride1=1, stride2=1)

        conv_redir = mx.sym.Convolution(data=conv2_img1, pad=(0, 0), \
                                        kernel=(1, 1), stride=(1, 1), \
                                        num_filter=64, name='conv_redir')

        conv_redir = mx.sym.LeakyReLU(data=conv_redir,act_type='leaky',slope=0.1)
        concat = mx.sym.Concat(corr, conv_redir)
    else:
        conv3_img1 = mx.sym.Convolution(data=conv2_img1, pad=(2, 2), \
                                        kernel=(5, 5), stride=(2, 2), \
                                        num_filter=256, weight=weights[2], \
                                        bias=bias[2], name='conv3_img1')
        conv3_img1 = mx.sym.LeakyReLU(data=conv3_img1, act_type='leaky', slope=0.1)

        conv3_img2 = mx.sym.Convolution(data=conv2_img2, pad=(2, 2), \
                                        kernel=(5, 5), stride=(2, 2), \
                                        num_filter=256, weight=weights[2], \
                                        bias=bias[2], name='conv3_img2')
        conv3_img2 = mx.sym.LeakyReLU(data=conv3_img2, act_type='leaky', slope=0.1)

        corr = mx.sym.Correlation(data1=conv3_img1, data2=conv3_img2, \
                                  pad_size=20, kernel_size=1, \
                                  max_displacement=20, stride1=1, stride2=2)

        conv_redir = mx.sym.Convolution(data=conv3_img1, pad=(0, 0), \
                                        kernel=(1, 1), stride=(1, 1), \
                                        num_filter=64, name='conv_redir')
        conv_redir = mx.sym.LeakyReLU(data=conv_redir, act_type='leaky', slope=0.1)
        concat = mx.sym.Concat(corr, conv_redir)

    if net_type =='stereo':
        stride = (2,2)
    else:
        stride = (1,1)
    conv3a = mx.sym.Convolution(concat, pad=(2, 2), kernel=(5, 5), \
                                stride=stride, num_filter=256, name='conv3a')
    conv3a = mx.sym.LeakyReLU(data=conv3a, act_type='leaky', slope=0.1)

    conv3b = mx.sym.Convolution(conv3a, pad=(1, 1), kernel=(3, 3), \
                                stride=(1, 1), num_filter=256, name='conv3b')
    conv3b = mx.sym.LeakyReLU(data=conv3b, act_type='leaky', slope=0.1)

    conv4a = mx.sym.Convolution(conv3b, pad=(1, 1), kernel=(3, 3), \
                                stride=(2, 2), num_filter=512, name='conv4a')
    conv4a = mx.sym.LeakyReLU(data=conv4a, act_type='leaky', slope=0.1)

    conv4b = mx.sym.Convolution(conv4a, pad=(1, 1), kernel=(3, 3), \
                                stride=(1, 1), num_filter=512, name='conv4b')
    conv4b = mx.sym.LeakyReLU(data=conv4b, act_type='leaky', slope=0.1)

    conv5a = mx.sym.Convolution(conv4b, pad=(1, 1), kernel=(3, 3), \
                                stride=(2, 2), num_filter=512, name='conv5a')
    conv5a = mx.sym.LeakyReLU(data=conv5a, act_type='leaky', slope=0.1)

    conv5b = mx.sym.Convolution(conv5a, pad=(1, 1), kernel=(3, 3), \
                                stride=(1, 1), num_filter=512, name='conv5b')
    conv5b = mx.sym.LeakyReLU(data=conv5b, act_type='leaky', slope=0.1)

    conv6a = mx.sym.Convolution(conv5b, pad=(1, 1), kernel=(3, 3), \
                                stride=(2, 2), num_filter=1024, name='conv6a')
    conv6a = mx.sym.LeakyReLU(data=conv6a, act_type='leaky', slope=0.1)

    conv6b = mx.sym.Convolution(conv6a, pad=(1, 1), kernel=(3, 3), \
                                stride=(1, 1), num_filter=1024, name='conv6b')
    conv6b = mx.sym.LeakyReLU(data=conv6b, act_type='leaky', slope=0.1, )

    pr6 = mx.sym.Convolution(conv6b,pad= (1,1),kernel=(3,3),stride=(1,1),\
                             num_filter=output_dim,name='pr6')

    upsample_pr6to5 = mx.sym.Deconvolution(pr6, pad=(1,1), kernel=(4,4), \
                                           stride=(2,2), num_filter=1, \
                                           name='upsample_pr6to5',no_bias=True)
    upconv5 = mx.sym.Deconvolution(conv6b,pad=(1,1),kernel=(4,4),stride=(2,2), \
                                   num_filter=512,name='upconv5',no_bias=True)
    upconv5 = mx.sym.LeakyReLU(data = upconv5,act_type = 'leaky',slope  = 0.1)
    concat_tmp = mx.sym.Concat(conv5b,upconv5,upsample_pr6to5,dim=1)

    iconv5 = mx.sym.Convolution(concat_tmp,pad = (1,1),kernel=(3,3),\
                                stride=(1,1),num_filter = 512,name='iconv5')

    pr5  = mx.sym.Convolution(iconv5, pad = (1,1),kernel=(3,3),stride=(1,1),\
                              num_filter = output_dim,name='pr5')

    upconv4 = mx.sym.Deconvolution(iconv5,pad = (1,1),kernel= (4,4),\
                                    stride = (2,2),num_filter=256, \
                                    name='upconv4',no_bias=True)
    upconv4 = mx.sym.LeakyReLU(data = upconv4,act_type = 'leaky',slope  = 0.1 )

    upsample_pr5to4 = mx.sym.Deconvolution(pr5,pad = (1,1),kernel= (4,4),\
                                           stride=(2,2),num_filter=1,\
                                           name='upsample_pr5to4',no_bias=True)

    concat_tmp2 = mx.sym.Concat(conv4b,upconv4,upsample_pr5to4)
    iconv4  = mx.sym.Convolution(concat_tmp2,pad = (1,1),kernel = (3,3),\
                                 stride=(1,1),num_filter=256,name='iconv4')
    pr4 = mx.sym.Convolution(iconv4,pad=(1,1),kernel=(3,3),stride=(1,1),\
                             num_filter=output_dim,name='pr4')

    upconv3 = mx.sym.Deconvolution(iconv4,pad=(1,1),kernel=(4,4),stride=(2,2),\
                                   num_filter=128,name='upconv3',no_bias=True)
    upconv3 = mx.sym.LeakyReLU(data = upconv3,act_type = 'leaky',slope  = 0.1 )

    upsample_pr4to3 = mx.sym.Deconvolution(pr4,pad = (1,1),kernel= (4,4),stride=(2,2),num_filter=1,name='upsample_pr4to3',no_bias=True)
    concat_tmp3 = mx.sym.Concat(conv3b,upconv3,upsample_pr4to3)
    iconv3 = mx.sym.Convolution(concat_tmp3,pad=(1,1),kernel=(3,3),\
                                stride=(1,1),num_filter = 128,name='iconv3')
    pr3 = mx.sym.Convolution(iconv3,pad = (1,1), kernel = (3,3), \
                             stride = (1,1),num_filter = output_dim,name='pr3')


    upconv2 = mx.sym.Deconvolution(iconv3,pad=(1,1),kernel=(4,4),stride=(2,2),\
                                   num_filter=64,name='upconv2',no_bias=True)
    upconv2 = mx.sym.LeakyReLU(data = upconv2,act_type = 'leaky',slope  = 0.1  )

    upsample_pr3to2 = mx.sym.Deconvolution(pr3,pad = (1,1),kernel= (4,4),stride=(2,2),num_filter=1,name='upsample_pr3to2',no_bias=True)

    concat_tmp4 = mx.sym.Concat(conv2_img1,upconv2,upsample_pr3to2)

    iconv2 = mx.sym.Convolution(concat_tmp4,pad = (1,1),kernel = (3,3),\
                                stride= (1,1),num_filter = 64,name='iconv2')
    pr2 = mx.sym.Convolution(iconv2,pad = (1,1),kernel=(3,3),stride = (1,1), \
                             num_filter = output_dim,name='pr2')

    upconv1 = mx.sym.Deconvolution(iconv2,pad=(1,1),kernel=(4,4),stride=(2,2),\
                                   num_filter = 32,name='upconv1',no_bias=True)
    upconv1 = mx.sym.LeakyReLU(data = upconv1,act_type = 'leaky',slope  = 0.1 )

    upsample_pr2to1 = mx.sym.Deconvolution(pr2,pad = (1,1),kernel= (4,4),stride=(2,2),num_filter=1,name='upsample_pr2to1',no_bias=True)

    concat_tmp5 = mx.sym.Concat(conv1_img1,upconv1,upsample_pr2to1)
    iconv1 = mx.sym.Convolution(concat_tmp5,pad=(1,1),kernel = (3,3), \
                                stride=(1,1),num_filter=32,name='iconv1')
    pr1 = mx.sym.Convolution(iconv1,pad=(1,1),kernel=(3,3),stride=(1,1),\
                             num_filter=output_dim,name='pr1')


    img1 = mx.sym.BlockGrad(data=data,name='img1_tmp')
    img2 = mx.sym.BlockGrad(data=data2,name='img2_tmp')
    corr = mx.sym.BlockGrad(data=corr,name='corr')
    conv3_img1 = mx.sym.BlockGrad(data=conv3_img1,name='conv2_img1_tmp')
    conv3_img2 = mx.sym.BlockGrad(data=conv3_img2,name='conv2_img2_tmp')
    net = mx.sym.Group([pr1,conv3_img1,conv3_img2,corr,img1,img2])

    return pr1
