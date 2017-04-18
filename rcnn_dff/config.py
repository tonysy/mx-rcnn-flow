import numpy as np
from easydict import EasyDict as edict

config = edict()

config.RAWDATA_PAHT = '/data/syzhang/kitti_rawdata'
config.RAND_FILE = './devkit_object/mapping/train_rand.txt'
config.MAP_FILE = './devkit_object/mapping/train_mapping.txt'
config.SHARE_PARAMS_LIST = ['share1_weight', 'share1_bias', 'share2_weight', 'share2_bias', 'share3_weight', 'share3_bias', 'conv_redir_weight', 'conv_redir_bias', 'conv3a_weight', 'conv3a_bias', 'conv3b_weight', 'conv3b_bias', 'conv4a_weight', 'conv4a_bias', 'conv4b_weight', 'conv4b_bias', 'conv5a_weight', 'conv5a_bias', 'conv5b_weight', 'conv5b_bias', 'conv6a_weight', 'conv6a_bias', 'conv6b_weight', 'conv6b_bias', 'upconv5_weight', 'pr6_weight', 'pr6_bias', 'upsample_pr6to5_weight', 'iconv5_weight', 'iconv5_bias', 'upconv4_weight', 'pr5_weight', 'pr5_bias', 'upsample_pr5to4_weight', 'iconv4_weight', 'iconv4_bias', 'upconv3_weight', 'pr4_weight', 'pr4_bias', 'upsample_pr4to3_weight', 'iconv3_weight', 'iconv3_bias', 'upconv2_weight', 'pr3_weight', 'pr3_bias', 'upsample_pr3to2_weight', 'iconv2_weight', 'iconv2_bias', 'upconv1_weight', 'pr2_weight', 'pr2_bias', 'upsample_pr2to1_weight', 'iconv1_weight', 'iconv1_bias', 'pr1_weight', 'pr1_bias', 'stereo_scale_weight', 'stereo_scale_bias']

config.FLOW_SCALE_FACTOR = 0.00390625 # 0.00392156
config.FRAMES_FEATURE_AGGREGATION = 2
# network related params
config.PIXEL_MEANS = np.array([103.939, 116.779, 123.68])
config.IMAGE_STRIDE = 64
config.RPN_FEAT_STRIDE = 16 # 8
config.RCNN_FEAT_STRIDE = 16 # 2
config.FIXED_PARAMS = ['conv1', 'conv2']
config.FIXED_PARAMS_SHARED = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']

# dataset related params
# config.NUM_CLASSES = 21
config.NUM_CLASSES = 4
config.SCALES = [(384, 1280)]  # first is scale (the shorter side); second is max size
# config.SCALES = [(376, 1242)]  # first is scale (the shorter side); second is max size
# config.ANCHOR_SCALES = (8, 16, 32)
config.ANCHOR_SCALES = (4, 8, 16, 24)
config.ANCHOR_RATIOS = (0.5, 1, 2)
config.NUM_ANCHORS = len(config.ANCHOR_SCALES) * len(config.ANCHOR_RATIOS)

config.RPN_IOU_LOSS = False
config.RCNN_IOU_LOSS = False
config.RCNN_CTX_WINDOW = False

config.TRAIN = edict()

# R-CNN and RPN
# size of images for each device, 2 for rcnn, 1 for rpn and e2e
config.TRAIN.BATCH_IMAGES = 2
# e2e changes behavior of anchor loader and metric
config.TRAIN.END2END = False
# group images with similar aspect ratio
config.TRAIN.ASPECT_GROUPING = True

# crop
config.TRAIN.CROP = 'origin'  # can be origin, random, center
config.TRAIN.CROP_SHAPE = (360, 1000)
config.TRAIN.CROP_SCALE_RANGE = (1, 2.5)
config.TRAIN.CROP_BOX_SCALE_MIN = 24
config.TRAIN.CROP_BOX_SCALE_MAX = 384

# R-CNN
# rcnn rois batch size
config.TRAIN.BATCH_ROIS = 128
# rcnn rois sampling params
config.TRAIN.FG_FRACTION = 0.25
config.TRAIN.FG_THRESH = 0.5
config.TRAIN.BG_THRESH_HI = 0.5
config.TRAIN.BG_THRESH_LO = 0.0
# rcnn bounding box regression params
config.TRAIN.BBOX_REGRESSION_THRESH = 0.5
config.TRAIN.BBOX_WEIGHTS = np.array([1.0, 1.0, 1.0, 1.0])
# rcnn ohem training params
config.TRAIN.RCNN_OHEM = False
config.TRAIN.RCNN_OHEM_ROIS = 300
config.TRAIN.RCNN_OHEM_NMS = 0.7
config.TRAIN.RCNN_OHEM_TRANSFORM = False
config.TRAIN.RCNN_OHEM_IGNORE = 0

# RPN anchor loader
# rpn anchors batch size
config.TRAIN.RPN_BATCH_SIZE = 256
# rpn anchors sampling params
config.TRAIN.RPN_FG_FRACTION = 0.5
config.TRAIN.RPN_POSITIVE_OVERLAP = 0.7
config.TRAIN.RPN_NEGATIVE_OVERLAP = 0.3
config.TRAIN.RPN_CLOBBER_POSITIVES = False
# rpn bounding box regression params
config.TRAIN.RPN_BBOX_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
config.TRAIN.RPN_POSITIVE_WEIGHT = -1.0
# rpn ohem training params
config.TRAIN.RPN_OHEM = False
config.TRAIN.RPN_OHEM_ANCHORS = 2000
config.TRAIN.RPN_OHEM_NMS = 0.7
config.TRAIN.RPN_OHEM_TRANSFORM = False
config.TRAIN.RPN_OHEM_IGNORE = 0
config.TRAIN.RPN_OHEM_NP_RATIO = 2  # negative / positive ratio

# used for end2end training
# RPN proposal
config.TRAIN.CXX_PROPOSAL = True
config.TRAIN.RPN_NMS_THRESH = 0.7
config.TRAIN.RPN_PRE_NMS_TOP_N = 12000
config.TRAIN.RPN_POST_NMS_TOP_N = 2000
config.TRAIN.RPN_MIN_SIZE = config.RPN_FEAT_STRIDE
# approximate bounding box regression
config.TRAIN.BBOX_NORMALIZATION_PRECOMPUTED = False
config.TRAIN.BBOX_MEANS = (0.0, 0.0, 0.0, 0.0)
config.TRAIN.BBOX_STDS = (0.1, 0.1, 0.2, 0.2)

config.TEST = edict()

# R-CNN testing
# use rpn to generate proposal
config.TEST.HAS_RPN = False
# size of images for each device
config.TEST.BATCH_IMAGES = 1

# RPN proposal
config.TEST.CXX_PROPOSAL = True
config.TEST.RPN_NMS_THRESH = 0.7
config.TEST.RPN_PRE_NMS_TOP_N = 6000
config.TEST.RPN_POST_NMS_TOP_N = 300
config.TEST.RPN_MIN_SIZE = config.RPN_FEAT_STRIDE

# RPN generate proposal
config.TEST.PROPOSAL_NMS_THRESH = 0.7
config.TEST.PROPOSAL_PRE_NMS_TOP_N = 20000
config.TEST.PROPOSAL_POST_NMS_TOP_N = 2000
config.TEST.PROPOSAL_MIN_SIZE = config.RPN_FEAT_STRIDE

# RCNN nms
config.TEST.NMS = 0.3

# default settings
default = edict()

# default network
default.network = 'vgg'
default.pretrained = '/data/syzhang/model/vgg16-e2e'
default.pretrained_epoch = 10
default.base_lr = 0.001
# default dataset
default.dataset = 'PascalVOC'
default.image_set = '2007_trainval'
default.test_image_set = '2007_test'
default.root_path = 'data'
default.dataset_path = 'data/VOCdevkit'
# default training
default.frequent = 20
default.kvstore = 'device'
# default e2e
default.e2e_prefix = 'model/e2e'
default.e2e_epoch = 10
default.e2e_lr = default.base_lr
default.e2e_lr_step = '7'
# default rpn
default.rpn_prefix = 'model/rpn'
default.rpn_epoch = 8
default.rpn_lr = default.base_lr
default.rpn_lr_step = '6'
# default rcnn
default.rcnn_prefix = 'model/rcnn'
default.rcnn_epoch = 8
default.rcnn_lr = default.base_lr
default.rcnn_lr_step = '6'

# network settings
network = edict()

network.vgg = edict()

network.resnet = edict()
network.resnet.pretrained = 'model/resnet-101'
network.resnet.pretrained_epoch = 0
network.resnet.PIXEL_MEANS = np.array([0, 0, 0])
network.resnet.IMAGE_STRIDE = 0
network.resnet.RPN_FEAT_STRIDE = 16
network.resnet.RCNN_FEAT_STRIDE = 16
network.resnet.FIXED_PARAMS = ['conv0', 'stage1', 'gamma', 'beta']
network.resnet.FIXED_PARAMS_SHARED = ['conv0', 'stage1', 'stage2', 'stage3', 'gamma', 'beta']

network.enet = edict()
network.enet.pretrained = 'model/enet'
network.enet.pretrained_epoch = 140
network.enet.IMAGE_STRIDE = 2
network.enet.RPN_FEAT_STRIDE = 8
network.enet.RCNN_FEAT_STRIDE = 8
network.enet.FIXED_PARAMS = ['initial', 'bottleneck1', 'projection1', 'gamma', 'beta']
network.enet.FIXED_PARAMS_SHARED = ['initial', 'bottleneck1', 'projection1', 'bottleneck2', 'projection2', 'bottleneck3', 'projection3', 'gamma', 'beta']

# dataset settings
dataset = edict()

dataset.PascalVOC = edict()

dataset.Kitti = edict()
dataset.Kitti.image_set = 'train'
dataset.Kitti.test_image_set = 'val'
dataset.Kitti.root_path = 'data'
dataset.Kitti.dataset_path = 'data/kitti'
dataset.Kitti.NUM_CLASSES = 4
# dataset.Kitti.SCALES = [(376, 1242)]
dataset.Kitti.SCALES = [(384, 1280)]

# dataset.Kitti.ANCHOR_SCALES = (2, 4, 8, 16)
dataset.Kitti.ANCHOR_SCALES = (4, 8, 16, 24)
dataset.Kitti.ANCHOR_RATIOS = (0.5, 1, 2)
dataset.Kitti.NUM_ANCHORS = len(dataset.Kitti.ANCHOR_SCALES) * len(dataset.Kitti.ANCHOR_RATIOS)

dataset.Custom = edict()
dataset.Custom.image_set = 'train'
dataset.Custom.test_image_set = 'test0922'
dataset.Custom.root_path = 'data'
dataset.Custom.dataset_path = 'data/custom'
dataset.Custom.NUM_CLASSES = 4
dataset.Custom.SCALES = [(540, 960)]
dataset.Custom.ANCHOR_SCALES = (2, 4, 8, 16, 24)
dataset.Custom.ANCHOR_RATIOS = (0.5, 1, 2)
dataset.Custom.NUM_ANCHORS = len(dataset.Custom.ANCHOR_SCALES) * len(dataset.Custom.ANCHOR_RATIOS)


def generate_config(_network, _dataset):
    for k, v in network[_network].items():
        if k in config:
            config[k] = v
        elif k in default:
            default[k] = v
    for k, v in dataset[_dataset].items():
        if k in config:
            config[k] = v
        elif k in default:
            default[k] = v
