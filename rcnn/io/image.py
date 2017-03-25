import numpy as np
import numpy.random as npr
import cv2
import os
import random
from ..config import config


def get_image(roidb, crop='origin'):
    """
    preprocess image and return processed roidb
    :param roidb: a list of roidb
    :param crop: ['origin', 'center', 'random']
    :return: list of img as in mxnet format
    roidb add new item['im_info']
    0 --- x (width, second dim of im)
    |
    y (height, first dim of im)
    """
    num_images = len(roidb)
    processed_ims = []
    processed_roidb = []
    for i in range(num_images):
        roi_rec = roidb[i]
        assert os.path.exists(roi_rec['image']), '%s does not exist'.format(roi_rec['image'])
        im = cv2.imread(roi_rec['image'])
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
        if crop == 'random':
            new_rec = roi_rec.copy()
            # resize image so that we can crop and the ratio is random between range_lo, range_hi
            size = config.TRAIN.CROP_SHAPE
            size_range = config.TRAIN.CROP_SCALE_RANGE
            rand_ratio = npr.uniform(size_range[0], size_range[1])
            im_scale = max(float(size[0]) / im.shape[0], float(size[1]) / im.shape[1], rand_ratio)
            im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
            # decide random crop start location
            crop_w = size[1]
            crop_h = size[0]
            start_w = npr.randint(0, im.shape[1] - crop_w + 1)
            start_h = npr.randint(0, im.shape[0] - crop_h + 1)
            im_cropped = im[start_h:start_h + crop_h, start_w:start_w + crop_w, :]
            # transform ground truth
            boxes = roi_rec['boxes'].copy() * im_scale
            ctrs_x = (boxes[:, 2] + boxes[:, 0]) / 2.0
            ctrs_y = (boxes[:, 3] + boxes[:, 1]) / 2.0
            keep = np.where((ctrs_x > start_w) & (ctrs_y > start_h) &
                            (ctrs_x < start_w + crop_w) & (ctrs_y < start_h + crop_h))
            boxes = boxes[keep].copy()
            boxes[:, [0, 2]] -= start_w
            boxes[:, [1, 3]] -= start_h
            # create new roidb
            new_rec['boxes'] = boxes
            # transform bbox_targets
            if config.RCNN_IOU_LOSS:
                bbox_targets = roi_rec['bbox_targets'].copy()
                bbox_targets = bbox_targets[keep].copy()
                bbox_targets[:, 1:] *= im_scale
                bbox_targets[:, [1, 3]] -= start_w
                bbox_targets[:, [2, 4]] -= start_h
                new_rec['bbox_targets'] = bbox_targets
                items = ['gt_classes', 'gt_overlaps', 'max_classes', 'max_overlaps']
            else:
                items = ['gt_classes', 'gt_overlaps', 'max_classes', 'max_overlaps', 'bbox_targets']

            for k in items:
                if k in new_rec:
                    new_rec[k] = new_rec[k][keep].copy()
            # fill in info
            im_tensor = transform(im_cropped, config.PIXEL_MEANS)
            im_info = [im_tensor.shape[2], im_tensor.shape[3], im_scale]
            new_rec['im_info'] = im_info
            processed_ims.append(im_tensor)
            processed_roidb.append(new_rec)
        elif crop == 'center':
            new_rec = roi_rec.copy()
            scale_ind = random.randrange(len(config.SCALES))
            rescale_size = config.SCALES[scale_ind]
            im_scale = max(float(rescale_size[0]) / im.shape[0], float(rescale_size[1]) / im.shape[1])
            crop_size = config.TRAIN.CROP_SHAPE
            crop_h = float(crop_size[0] + im_scale)
            crop_w = float(crop_size[1] + im_scale)
            boxes = roi_rec['boxes'].copy()
            gt_inds = np.where(roi_rec['gt_classes'] != 0)[0]
            gt_boxes = boxes[gt_inds, :]
            if gt_boxes.size > 0:
                # random select box
                rand_index = npr.randint(gt_boxes.shape[0])
                box = gt_boxes[rand_index, :]
                area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)

                # build scale pyramid
                scale_pyramid = np.array([(i / 2.) for i in range(int(2 * np.log(config.TRAIN.CROP_BOX_SCALE_MAX / config.TRAIN.CROP_BOX_SCALE_MIN) / np.log(2) + 1))])
                box_scale_pyramid = (config.TRAIN.CROP_BOX_SCALE_MIN * np.power(2, scale_pyramid)) ** 2

                # choose scale to match box area
                scale_overlap = np.minimum(float(area), box_scale_pyramid) / np.maximum(float(area), box_scale_pyramid)
                match_scale_ind = np.argmax(scale_overlap)
                match_scale = scale_pyramid[match_scale_ind]

                # subtract min scale
                scale_pyramid -= match_scale
                scale_pyramid = np.power(2, scale_pyramid)

                # random select scale
                box_scales = scale_pyramid[max(0, match_scale_ind - 2):min(match_scale_ind + 2, len(scale_pyramid))]
                box_scale = npr.choice(box_scales)
                im_scale *= box_scale

                # prevent crop is bigger than whole image
                im_scale = max(crop_h / float(im.shape[0]), crop_w / float(im.shape[1]), im_scale)

                # decide small crop shape
                origin_crop_h = int(round(crop_h / im_scale))
                origin_crop_w = int(round(crop_w / im_scale))

                # decide start point
                ctr_x = (box[2] + box[0]) / 2.0
                ctr_y = (box[3] + box[1]) / 2.0
                noise_h = npr.randint(-10, 10)
                noise_w = npr.randint(-30, 30)
                start_h = int(round(ctr_y - origin_crop_h / 2)) + noise_h
                start_w = int(round(ctr_x - origin_crop_w / 2)) + noise_w
                end_h = start_h + origin_crop_h
                end_w = start_w + origin_crop_w

                # prevent crop cross border
                if start_h < 0:
                    off = -start_h
                    start_h += off
                    end_h += off
                if start_w < 0:
                    off = -start_w
                    start_w += off
                    end_w += off
                if end_h > im.shape[0]:
                    off = end_h - im.shape[0]
                    end_h -= off
                    start_h -= off
                if end_w > im.shape[1]:
                    off = end_w - im.shape[1]
                    end_w -= off
                    start_w -= off
            else:
                # random crop from image
                origin_crop_h = int(round(crop_h / im_scale))
                origin_crop_w = int(round(crop_w / im_scale))
                start_h = npr.randint(0, im.shape[0] - origin_crop_h)
                start_w = npr.randint(0, im.shape[1] - origin_crop_w)
                end_h = start_h + origin_crop_h
                end_w = start_w + origin_crop_w
            # crop then resize
            im_cropped = im[start_h:end_h, start_w:end_w, :]
            im_rescaled = cv2.resize(im_cropped, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
            assert im_rescaled.shape[0] >= crop_size[0] and im_rescaled.shape[1] >= crop_size[1], \
                'cropped {} scaled {} scale {}'.format(im_cropped.shape, im_rescaled.shape, im_scale)
            im_rescaled = im_rescaled[:crop_size[0], :crop_size[1], :]
            # transform ground truth
            ctrs_x = (boxes[:, 2] + boxes[:, 0]) / 2.0
            ctrs_y = (boxes[:, 3] + boxes[:, 1]) / 2.0
            keep = np.where((ctrs_y > start_h) & (ctrs_x > start_w) &
                            (ctrs_y < end_h) & (ctrs_x < end_w))
            boxes = boxes[keep].copy()
            boxes[:, [0, 2]] -= start_w
            boxes[:, [1, 3]] -= start_h
            boxes *= im_scale
            # create new roidb
            new_rec['boxes'] = boxes
            # transform bbox_targets
            if config.RCNN_IOU_LOSS:
                bbox_targets = roi_rec['bbox_targets'].copy()
                bbox_targets = bbox_targets[keep].copy()
                bbox_targets[:, 1:] *= im_scale
                bbox_targets[:, [1, 3]] -= start_w
                bbox_targets[:, [2, 4]] -= start_h
                new_rec['bbox_targets'] = bbox_targets
                items = ['gt_classes', 'gt_overlaps', 'max_classes', 'max_overlaps']
            else:
                items = ['gt_classes', 'gt_overlaps', 'max_classes', 'max_overlaps', 'bbox_targets']

            for k in items:
                if k in new_rec:
                    new_rec[k] = new_rec[k][keep].copy()
            # fill in info
            im_tensor = transform(im_rescaled, config.PIXEL_MEANS)
            im_info = [im_tensor.shape[2], im_tensor.shape[3], im_scale]
            new_rec['im_info'] = im_info
            processed_ims.append(im_tensor)
            processed_roidb.append(new_rec)
        else:
            new_rec = roi_rec.copy()
            scale_ind = random.randrange(len(config.SCALES))
            target_size = config.SCALES[scale_ind][0]
            max_size = config.SCALES[scale_ind][1]
            im, im_scale = resize(im, target_size, max_size, stride=config.IMAGE_STRIDE)
            im_tensor = transform(im, config.PIXEL_MEANS)
            processed_ims.append(im_tensor)
            im_info = [im_tensor.shape[2], im_tensor.shape[3], im_scale]
            new_rec['boxes'] = roi_rec['boxes'].copy() * im_scale
            new_rec['im_info'] = im_info
            if config.RCNN_IOU_LOSS:
                if 'bbox_targets' in new_rec.keys():
                    bbox_targets = roi_rec['bbox_targets'].copy()
                    bbox_targets[:, 1:] *= im_scale
                    new_rec['bbox_targets'] = bbox_targets
            processed_roidb.append(new_rec)
    return processed_ims, processed_roidb


def resize(im, target_size, max_size, stride=0):
    """
    only resize input image to target size and return scale
    :param im: BGR image input by opencv
    :param target_size: one dimensional size (the short side)
    :param max_size: one dimensional max size (the long side)
    :param stride: if given, pad the image to designated stride
    :return:
    """
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)

    if stride == 0:
        return im, im_scale
    else:
        # pad to product of stride
        im_height = int(np.ceil(im.shape[0] / float(stride)) * stride)
        im_width = int(np.ceil(im.shape[1] / float(stride)) * stride)
        im_channel = im.shape[2]
        padded_im = np.zeros((im_height, im_width, im_channel))
        padded_im[:im.shape[0], :im.shape[1], :] = im
        return padded_im, im_scale


def transform(im, pixel_means):
    """
    transform into mxnet tensor
    substract pixel size and transform to correct format
    :param im: [height, width, channel] in BGR
    :param pixel_means: [B, G, R pixel means]
    :return: [batch, channel, height, width]
    """
    im_tensor = np.zeros((1, 3, im.shape[0], im.shape[1]))
    for i in range(3):
        im_tensor[0, i, :, :] = im[:, :, 2 - i] - pixel_means[2 - i]
    return im_tensor


def transform_inverse(im_tensor, pixel_means):
    """
    transform from mxnet im_tensor to ordinary RGB image
    im_tensor is limited to one image
    :param im_tensor: [batch, channel, height, width]
    :param pixel_means: [B, G, R pixel means]
    :return: im [height, width, channel(RGB)]
    """
    assert im_tensor.shape[0] == 1
    im_tensor = im_tensor.copy()
    # put channel back
    channel_swap = (0, 2, 3, 1)
    im_tensor = im_tensor.transpose(channel_swap)
    im = im_tensor[0]
    assert im.shape[2] == 3
    im += pixel_means[[2, 1, 0]]
    im = im.astype(np.uint8)
    return im


def tensor_vstack(tensor_list, pad=0):
    """
    vertically stack tensors
    :param tensor_list: list of tensor to be stacked vertically
    :param pad: label to pad with
    :return: tensor with max shape
    """
    ndim = len(tensor_list[0].shape)
    dtype = tensor_list[0].dtype
    islice = tensor_list[0].shape[0]
    dimensions = []
    first_dim = sum([tensor.shape[0] for tensor in tensor_list])
    dimensions.append(first_dim)
    for dim in range(1, ndim):
        dimensions.append(max([tensor.shape[dim] for tensor in tensor_list]))
    if pad == 0:
        all_tensor = np.zeros(tuple(dimensions), dtype=dtype)
    elif pad == 1:
        all_tensor = np.ones(tuple(dimensions), dtype=dtype)
    else:
        all_tensor = np.full(tuple(dimensions), pad, dtype=dtype)
    if ndim == 1:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind*islice:(ind+1)*islice] = tensor
    elif ndim == 2:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind*islice:(ind+1)*islice, :tensor.shape[1]] = tensor
    elif ndim == 3:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind*islice:(ind+1)*islice, :tensor.shape[1], :tensor.shape[2]] = tensor
    elif ndim == 4:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind*islice:(ind+1)*islice, :tensor.shape[1], :tensor.shape[2], :tensor.shape[3]] = tensor
    else:
        raise Exception('Sorry, unimplemented.')
    return all_tensor
