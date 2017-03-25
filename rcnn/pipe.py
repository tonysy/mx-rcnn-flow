import ConfigParser
import pprint
import mxnet as mx
import numpy as np
from rcnn.config import config, generate_config
from rcnn.symbol import *
from rcnn.io.image import transform
from rcnn.core.tester import Predictor, im_detect, vis_all_detection
from rcnn.utils.load_model import load_param
from rcnn.processing.nms import wnms_wrapper


class Pipeline:
    def __init__(self, config_file):
        """
        model file: '%s-%04d.params' % (prefix, epoch)
        network: 'vgg_test'
        device_name: 'gpu:0' where 0 is device_id
        vis: False
        thresh: 0.8
        nms_thresh_lo: 0.5
        nms_thresh_hi: 0.7
        max_height * max_width: no image should be above this size
        need_preprocess: the input image
        num_classes: 4
        class_names: __background__:car:pedestrian:cyclist
        """
        cfg = ConfigParser.ConfigParser()
        cfg.read(config_file)
        prefix = cfg.get('model', 'prefix')
        epoch = cfg.getint('model', 'epoch')
        network = cfg.get('model', 'network')
        device_name = cfg.get('test', 'device_name')
        self.vis = cfg.getboolean('test', 'vis')
        self.thresh = cfg.getfloat('test', 'thresh')
        nms_thresh_lo = cfg.getfloat('test', 'nms_thresh_lo')
        nms_thresh_hi = cfg.getfloat('test', 'nms_thresh_hi')
        max_height = cfg.getint('test', 'max_height')
        max_width = cfg.getint('test', 'max_width')
        self.need_preprocess = cfg.getboolean('test', 'need_preprocess')
        self.classes = cfg.get('info', 'class_names').split(':')
        self.num_classes = len(self.classes)

        # set config
        config.TEST.HAS_RPN = True
        generate_config(network.split('_')[0], 'Custom')

        # print config
        pprint.pprint(config)

        # decide ctx
        if device_name == 'cpu':
            ctx = mx.cpu()
        else:
            device_id = int(device_name[4:])
            ctx = mx.gpu(device_id)

        # load model
        sym = eval('get_' + network)(num_classes=config.NUM_CLASSES, num_anchors=config.NUM_ANCHORS)
        arg_params, aux_params = load_param(prefix, epoch, convert=True, ctx=ctx, process=True)

        # get predictor
        data_names = ['data', 'im_info']
        label_names = ['cls_prob_label']
        data_shapes = [('data', (1, 3, max_height, max_width)), ('im_info', (1, 3))]
        label_shapes = []
        self.predictor = Predictor(sym, data_names, label_names, context=ctx,
                                   provide_data=data_shapes, provide_label=label_shapes,
                                   arg_params=arg_params, aux_params=aux_params)
        self.nms = wnms_wrapper(nms_thresh_lo, nms_thresh_hi)

    def generate_batch(self, im_array, im_scale):
        """
        preprocess image, return batch
        :param im_array:
        if need_preprocess: cv2.imread returns [height, width, channel] in BGR
        else: MXNet NDArray
        :param im_scale:
        :return:
        data_batch: MXNet input batch
        data_names: names in data_batch
        im_scale: float number
        """
        if self.need_preprocess:
            im_array = transform(im_array, config.PIXEL_MEANS)
        im_info = np.array([[im_array.shape[2], im_array.shape[3], im_scale]], dtype=np.float32)
        data_names = ['data', 'im_info']
        data = [mx.nd.array(im_array), mx.nd.array(im_info)]
        data_shapes = [(k, v.shape) for k, v in zip(data_names, data)]
        data_batch = mx.io.DataBatch(data=data, label=None, provide_data=data_shapes, provide_label=[])
        return data_batch, data_names, im_scale

    def process(self, im, im_scale=1.0):
        """
        process each input image
        :param im: input image
        if need_preprocess: cv2.imread returns [height, width, channel] in BGR
        else: MXNet NDArray
        :param im_scale: bounding boxes will be resized back with im_scale
        :return: [[], [car_dets], [ped_dets], [cyc_dets]]
        each dets is [[x1, y1, x2, y2]]
        """
        data_batch, data_names, im_scale = self.generate_batch(im, im_scale)
        scores, boxes, data_dict = im_detect(self.predictor, data_batch, data_names, im_scale)

        all_boxes = [[] for _ in self.classes]
        for cls_ind, cls in enumerate(self.classes):
            cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
            cls_scores = scores[:, cls_ind, np.newaxis]
            keep = np.where(cls_scores >= self.thresh)[0]
            dets = np.hstack((cls_boxes, cls_scores)).astype(np.float32)[keep, :]
            all_boxes[cls_ind] = self.nms(dets)

        boxes_this_image = [[]] + [all_boxes[j] for j in range(1, self.num_classes)]
        if self.vis:
            vis_all_detection(data_dict['data'].asnumpy(), boxes_this_image, self.classes, im_scale)
        return boxes_this_image
