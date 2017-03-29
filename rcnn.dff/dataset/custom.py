import cPickle
import cv2
import os
import numpy as np

from imdb import IMDB


class Custom(IMDB):
    def __init__(self, image_set, root_path, dataset_path):
        """
        fill basic information to initialize imdb
        :param image_set: train or val or trainval or test
        :param root_path: 'cache' and 'rpn_data'
        :param dataset_path: data and results
        :return: imdb object
        """
        super(Custom, self).__init__('Custom', image_set, root_path, dataset_path)
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
                    label = line.strip().split(' ')
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
        gt_roidb = self.load_custom_annotations()
        with open(cache_file, 'wb') as f:
            cPickle.dump(gt_roidb, f, cPickle.HIGHEST_PROTOCOL)

        return gt_roidb

    def load_custom_annotations(self):
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
                for i in range(self.num_classes - 1):
                    box_list.append([])
                label = line.strip().split(' ')
                for i in range(1, len(label), 5):
                    l = int(label[i + 4]) - 1
                    box_list[l].extend([float(label[j]) for j in range(i, i + 4)])
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
        write to cache and generate custom format
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
            print 'Writing {} custom results'.format(cls)
            filename = os.path.join(self.cache_path, 'results', self.image_set + '_' + cls + '.txt')
            with open(filename, 'w') as f:
                for im_ind, index in enumerate(self.image_set_index):
                    dets = detections[cls_ind][im_ind]
                    if len(dets) == 0:
                        continue
                    for k in range(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0], dets[k, 1], dets[k, 2], dets[k, 3]))
