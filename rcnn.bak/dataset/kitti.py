"""
KITTI Object Detection Database
Image list and annotation format follow the multicustom format.
"""

import cv2
import os
import numpy as np
import cPickle
from imdb import IMDB


class Kitti(IMDB):
    def __init__(self, image_set, root_path, dataset_path):
        """
        fill basic information to initialize imdb
        :param image_set: train or val or trainval or test
        :param root_path: 'cache' and 'rpn_data'
        :param dataset_path: data and results
        :return: imdb object
        """
        super(Kitti, self).__init__('kitti', image_set, root_path, dataset_path)
        self.image_set = image_set
        self.root_path = root_path
        self.data_path = dataset_path

        self.classes = ['__background__', 'car', 'pedestrian', 'cyclist']
        self.num_classes = len(self.classes)
        self.image_set_index = self.load_image_set_index()
        self.num_images = len(self.image_set_index)
        print 'num_images', self.num_images

    def load_image_set_index(self):
        """
        find out which indexes correspond to given image set (train or val)
        :return:
        """
        image_set_index_file = os.path.join(self.data_path, 'imglists', self.image_set + '.lst')
        assert os.path.exists(image_set_index_file), 'Path does not exist: {}'.format(image_set_index_file)
        image_set_index = []
        with open(image_set_index_file, 'r') as f:
            for line in f:
                if len(line) > 1:
                    label = line.strip().split(':')
                    image_set_index.append(label[0])
        return image_set_index

    def image_path_from_index(self, index):
        """
        given image index, find out full path
        :param index: index of a specific image
        :return: full path of this image
        """
        image_file = os.path.join(self.data_path, 'images', index)
        assert os.path.exists(image_file), 'Path does not exist: {}'.format(image_file)
        return image_file

    def gt_roidb(self):
        """
        return ground truth image regions database
        :return: imdb[image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                roidb = cPickle.load(f)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb
        gt_roidb = self.load_kitti_annotations()
        with open(cache_file, 'wb') as f:
            cPickle.dump(gt_roidb, f, cPickle.HIGHEST_PROTOCOL)

        return gt_roidb

    def load_kitti_annotations(self):
        """
        for a given index, load image and bounding boxes info from a single image list
        :return: list of record['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        """
        annotation_file = os.path.join(self.data_path, 'imglists', self.image_set + '.lst')
        assert os.path.exists(annotation_file), 'Path does not exist: {}'.format(annotation_file)
        total_box_list = []
        with open(annotation_file, 'r') as f:
            for line in f:
                box_list = []
                label = line.strip().split(':')
                bbox = label[1:]
                for i in range(self.num_classes - 1):
                    if len(bbox[i]) == 0:
                        box_list.append([])
                        continue
                    else:
                        class_i_box = map(float, bbox[i].strip().split(' '))
                        box_list.append(class_i_box)
                total_box_list.append(box_list)

        assert len(total_box_list) == self.num_images, 'number of boxes matrix must match number of images'
        roidb = []
        for im in range(self.num_images):
            roi_rec = dict()
            roi_rec['image'] = self.image_path_at(im)
            size = cv2.imread(roi_rec['image']).shape
            roi_rec['height'] = size[0]
            roi_rec['width'] = size[1]
            box_list = total_box_list[im]
            boxes = np.concatenate([np.array(box_list[i], dtype=np.float32) for i in range(self.num_classes - 1)], axis=0)
            boxes = boxes.reshape(-1, 4)
            num_objs_list = [len(box_list[i]) / 4 for i in range(self.num_classes - 1)]
            total_num_objs = np.sum(num_objs_list)
            gt_classes = np.zeros((total_num_objs, ), dtype=np.int32)
            overlaps = np.zeros((total_num_objs, self.num_classes), dtype=np.float32)
            for ix in range(total_num_objs):
                for j in range(self.num_classes - 1):
                    if ix < np.sum(num_objs_list[:j+1]):
                        gt_classes[ix] = j + 1
                        overlaps[ix, j+1] = 1
                        break
            roi_rec.update({'boxes': boxes,
                            'gt_classes': gt_classes,
                            'gt_overlaps': overlaps,
                            'max_classes': overlaps.argmax(axis=1),
                            'max_overlaps': overlaps.max(axis=1),
                            'flipped': False})
            roidb.append(roi_rec)
        return roidb

    def evaluate_detections(self, detections):
        """
        write to cache and generate kitti format
        :param detections: result matrix, [bbox, confidence]
        :return:
        """
        res_folder = os.path.join(self.cache_path, 'results')
        if not os.path.exists(res_folder):
            os.makedirs(res_folder)
        # write out all results
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing preliminary {} results'.format(cls)
            filename = os.path.join(self.cache_path, 'results', self.image_set + '_' + cls + '.txt')
            with open(filename, 'w') as f:
                for im_ind, index in enumerate(self.image_set_index):
                    dets = detections[cls_ind][im_ind]
                    if len(dets) == 0:
                        continue
                    for k in range(dets.shape[0]):
                        f.write('{:s} {:.8f} {:.2f} {:.2f} {:.2f} {:.2f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0], dets[k, 1], dets[k, 2], dets[k, 3]))
        # write kitti format
        self.gen_eval()

    def gen_eval(self):
        """
        save to kitti format
        :return:
        """
        import shutil
        res_dir = os.path.join(self.data_path, 'results/')
        if os.path.exists(res_dir):
            shutil.rmtree(res_dir)
        if not os.path.exists(res_dir):
            os.mkdir(res_dir)

        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing final {} results'.format(cls)
            filename = os.path.join(self.cache_path, 'results', self.image_set + '_' + cls + '.txt')
            with open(filename, 'r') as f:
                dets = f.readlines()
            for l in dets:
                im_ind = l.split(' ')[0]
                det = map(float, l.split(' ')[1:])
                res_dir_det = os.path.dirname(res_dir + im_ind)
                if not os.path.exists(res_dir_det):
                    os.makedirs(res_dir_det)
                with open(os.path.join(res_dir_det, os.path.basename(im_ind).split('.')[0] + '.txt'), 'a') as fo:
                    fo.write('%s -1 -1 -10 ' % cls)
                    fo.write('%.2f ' % det[1])
                    fo.write('%.2f ' % det[2])
                    fo.write('%.2f ' % det[3])
                    fo.write('%.2f ' % det[4])
                    fo.write('-1 -1 -1 -1000 -1000 -1000 -10 ')
                    fo.write('%.8f\n' % det[0])

        with open(os.path.join(self.data_path, 'imglists', self.image_set + '.lst')) as f:
            img_list = f.readlines()
        img_list = [item.split(':')[0] for item in img_list]
        for im_ind in img_list:
            res_dir_det = os.path.dirname(res_dir + im_ind)
            if not os.path.exists(res_dir_det):
                os.makedirs(res_dir_det)
            filename = os.path.join(res_dir_det, os.path.basename(im_ind).split('.')[0] + '.txt')
            if not os.path.exists(filename):
                print 'creating', filename
                open(filename, 'a').close()

