# coding:utf-8
## based on https://github.com/dmlc/mxnet/issues/1302
## Parses the model fit log file and generates a train/val vs epoch plot
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import numpy as np
import re
import argparse

parser = argparse.ArgumentParser(description='Parses log file and generates train/val curves')
parser.add_argument('--log_file', type=str,default="log_tr_va",
                    help='the path of log file')
parser.add_argument('--metric', type=str,default="L1Loss",
                    help='the path of log file')
args = parser.parse_args()


def plot_curve(metric, log_file):
    if metric == 'TRAIN_ACC':
        metric_name = ['Train-RPNAcc','Train-RCNNAcc']
    elif metric == 'ACC':
        metric_name = ['RPNAcc', 'RCNNAcc']
    elif metric == 'L1Loss':
        metric_name = ['RPNL1Loss', 'RCNNL1Loss']
    elif metric == 'LogLoss':
        metric_name = ['RPNLogLoss', 'RCNNLogLoss']
    else:
        assert 1==1, 'metric error!'

    if metric == 'TRAIN_ACC':
        RPN = re.compile('.*?]\s{}=([\d\.]+)'.format(metric_name[0]))
        RCNN = re.compile('.*?]\s{}=([\d\.]+)'.format(metric_name[1]))
    else:
        RPN = re.compile('.*{}=([\d\.]+).*?'.format(metric_name[0]))
        RCNN = re.compile('.*{}=([\d\.]+).*?'.format(metric_name[1]))
    log = open(log_file).read()
    log_rpn = [float(x) for x in RPN.findall(log)]
    log_rcnn = [float(x) for x in RCNN.findall(log)]

    idx = np.arange(len(log_rpn),dtype='float32')
    idx = idx / 186

    plt.figure(figsize=(8, 6))
    ax = plt.subplot(111)
    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.plot(idx, log_rpn, '-', linestyle='-', color="r",
             label=metric_name[0])

    plt.plot(idx, log_rcnn, '-', linestyle='-', color="b",
             label=metric_name[1])

    plt.legend(loc="best")

    # plt.xticks(np.arange(min(idx), max(idx)+1, 100), )
    # plt.yticks(np.arange(0, 1.2, 0.2))
    plt.ylim([0,1.2])

    xmajorLocator   = MultipleLocator(1) #将x主刻度标签设置为20的倍数
    xmajorFormatter = FormatStrFormatter('%5.1f') #设置x轴标签文本的格式
    xminorLocator   = MultipleLocator(0.2) #将x轴次刻度标签设置为5的倍数


    ymajorLocator   = MultipleLocator(0.1) #将y轴主刻度标签设置为0.5的倍数
    ymajorFormatter = FormatStrFormatter('%1.1f') #设置y轴标签文本的格式
    yminorLocator   = MultipleLocator(0.1) #将此y轴次刻度标签设置为0.1的倍数

    ax.xaxis.set_major_locator(xmajorLocator)
    ax.xaxis.set_major_formatter(xmajorFormatter)

    ax.yaxis.set_major_locator(ymajorLocator)
    ax.yaxis.set_major_formatter(ymajorFormatter)

    #显示次刻度标签的位置,没有标签文本
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)

    ax.xaxis.grid(True, which='major') #x坐标轴的网格使用主刻度
    ax.yaxis.grid(True, which='minor') #y坐标轴的网格使用次刻度

    plt.show()

plot_curve(args.metric, args.log_file)
