"""
ImageNet VID 15(IS) ILSVRC15
This class loads ground truth natations from standard ILSVRC15 XML data formats
and transform them into IMDB format. Selective search is used for proposals, see roid
"""
import cv2
import os
import numpy as np

from imdb import IMDB

class ImageNetVID(IMDB):
    def __init__(self, image_set, root_path, dataset_path):
        super(ImageNetVID, self).__init__('vid', image_set, root_path, dataset_path)
        self.image_set = image_set
        self.root_path = root_path
        self.data_path = dataset_path

        self.classes = ['__background__', # always index 0
                        'airplane','antelope','bear','bicycle',
                        'bird','bus','car','cattle','dog',
                        'domestic_cat','elephant','fox',
                        'giant_panda','hamster','horse',
                        'lion','lizard','monkey','motorcycle',
                        'rabbit','red_panda','sheep','snake',
                        'squirrel','tiger','train','turtle',
                        'watercraft','whale', 'zebra']
        self.num_classes = len(self.classes)
        self.image_set_index = self.load_image_set_index()
        self.num_images = len(self.image_set_index)
        print 'num_images', self.num_images

    def load_image_set_index(self):
