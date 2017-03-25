import numpy as np
from easydict import EasyDict as edict

dff_config = edict()

dff_config.EPS                                  = 1e-14
dff_config.PIXEL_MEANS                          = np.array([[[123.68, 116.779, 103.939]]])

dff_config.SCALES                               = [(376, 1242)]
dff_config.IMAGE_STRIDE                         = 64

# symbol
dff_config.NUM_CLASSES                          = 4
dff_config.RPN_FEAT_STRIDE                      = 16
dff_config.ANCHOR_SCALES                        = (4, 8, 16, 24)
dff_config.ANCHOR_RATIOS                        = (0.5, 1, 2)
dff_config.NUM_ANCHORS                          = len(dff_config.ANCHOR_RATIOS) * len(dff_config.ANCHOR_SCALES)
dff_config.RCNN_FEAT_STRIDE                     = 16
dff_config.FIXED_PARAMS                         = ['conv1', 'conv2'] + ['gamma', 'beta']
dff_config.FIXED_PARAMS_FINETUNE                = ['conv1','conv2', 'conv3', 'conv4', 'conv5'] + ['gamma', 'beta']

dff_config.TRAIN = edict()

# R-CNN and RPN
dff_config.TRAIN.BATCH_SIZE                     = 1
dff_config.TRAIN.END2END                        = False
dff_config.TRAIN.CROP                           = 'origin'
# R-CNN
dff_config.TRAIN.HAS_RPN                        = False
dff_config.TRAIN.BATCH_IMAGES                   = 2
dff_config.TRAIN.BATCH_ROIS                     = 128
dff_config.TRAIN.FG_FRACTION                    = 0.25
dff_config.TRAIN.FG_THRESH                      = 0.5
dff_config.TRAIN.BG_THRESH_HI                   = 0.5
dff_config.TRAIN.BG_THRESH_LO                   = 0.1

# r-cnn bounding box regression
dff_config.TRAIN.BBOX_REGRESSION_THRESH         = 0.5
dff_config.TRAIN.BBOX_INSIDE_WEIGHTS            = np.array([1.0, 1.0, 1.0, 1.0])

# RPN anchor loader
dff_config.TRAIN.RPN_BATCH_SIZE                 = 256
dff_config.TRAIN.RPN_FG_FRACTION                = 0.5
dff_config.TRAIN.RPN_POSITIVE_OVERLAP           = 0.7
dff_config.TRAIN.RPN_NEGATIVE_OVERLAP           = 0.3
dff_config.TRAIN.RPN_CLOBBER_WEIGHTS            = False
dff_config.TRAIN.RPN_BBOX_INSIDE_WEIGHTS        = (1.0, 1.0, 1.0, 1.0)
dff_config.TRAIN.RPN_POSITIVE_WEIGHT            = -1.0

dff_config.TRAIN.ASPECT_GROUPING                = True
# Used for end2end training
# RPN proposal
dff_config.TRAIN.CXX_PROPOSAL                   = True
dff_config.TRAIN.RPN_NMS_THRESH                 = 0.7
dff_config.TRAIN.RPN_PRE_NMS_TOP_N              = 12000
dff_config.TRAIN.RPN_POST_NMS_TOP_N             = 2000
dff_config.TRAIN.RPN_MIN_SIZE                   = 16

# approximate bounding box regression
dff_config.TRAIN.BBOX_NORMALIZATION_PRECOMPUTED = False
dff_config.TRAIN.BBOX_MEANS                     = (0.0, 0.0, 0.0, 0.0)
dff_config.TRAIN.BBOX_STDS                      = (0.1, 0.1, 0.2, 0.2)

# flownet
dff_config.TRAIN.FLOW_PRETRAINED                = './model/flow_model/flow'

dff_config.TEST = edict()

# R-CNN testing
dff_config.TEST.HAS_RPN                         = False
dff_config.TEST.BATCH_IMAGES                    = 1
dff_config.TEST.NMS                             = 0.3

# RPN proposal
dff_config.TEST.CXX_PROPOSAL                    = True
dff_config.TEST.RPN_NMS_THRESH                  = 0.7
dff_config.TEST.RPN_PRE_NMS_TOP_N               = 6000
dff_config.TEST.RPN_POST_NMS_TOP_N              = 300
dff_config.TEST.RPN_MIN_SIZE                    = 16
