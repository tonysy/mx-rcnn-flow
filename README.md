# Faster R-CNN

This version is based on [mx-rcnn](https://github.com/precedenceguo/mx-rcnn).
History of this implementation is:

* Fast R-CNN (v1)
* Faster R-CNN (v2)
* Faster R-CNN with module training (v3)
* Faster R-CNN with end-to-end training (v3.5, tornadomeet/mx-rcnn)
* Faster R-CNN with end-to-end training and module testing (v4)
* Faster R-CNN with accelerated training and resnet (v5)

mxnet/example/rcnn was v1, v3 and now v3.5.The basis of this version is v5.

## Deep Feature flow
modified the orginal Faster R-CNN mxnet version.

## Results Record
### 1. RCNN
#### Results

kitti | easy | medium | hard
:---: | :---: | :----: | :---:
car | 0.8782325455 | 0.7572223636 | 0.6334450909
pedestrian | 0.7038934545 | 0.6196088182 | 0.5497168182
cyclist | 0.7561599091 | 0.5126151818 | 0.4845294545

#### Methods
- optimizer: sgd
#### Training Logs

```
04-05 01:28 root         INFO     Epoch[9] Train-RPNAcc=0.998872
04-05 01:28 root         INFO     Epoch[9] Train-RPNLogLoss=0.003826
04-05 01:28 root         INFO     Epoch[9] Train-RPNL1Loss=0.118832
04-05 01:28 root         INFO     Epoch[9] Train-RCNNAcc=0.980337
04-05 01:28 root         INFO     Epoch[9] Train-RCNNLogLoss=0.051186
04-05 01:28 root         INFO     Epoch[9] Train-RCNNL1Loss=0.653532
04-05 01:28 root         INFO     Epoch[9] Time cost=1299.938
04-05 01:28 root         INFO     Saved checkpoint to "model/e2e-rcnn-log-0010.params"
```

### 2. RCNN-DFF-fix-flow
#### Method:
- Use `mx.symbol.GridGenerator` and `mx.symbol.BilinearSampler`
- `config.FLOW_SCALE_FACTOR = 0.00390625`

#### Results
kitti | easy | medium | hard
:---: | :---: | :----: | :---:
car |	0.707715727273	|  0.545954545455	| 0.470983818182
pedestrian | 	0.495135727273 |	0.414654636364	| 0.337167090909
cyclist |	0.504706545455 |	0.310941636364 |	0.304307272727

### 3. RCNN-DFF-non-fix-flow
#### Method:
- Use `mx.symbol.GridGenerator` and `mx.symbol.BilinearSampler`
- `config.FLOW_SCALE_FACTOR = 0.00390625`

#### Results
1. Epoch 8

kitti | easy | medium | hard
:---: | :---: | :----: | :---:
car | 0.793966090909 |	0.637243 |	0.553690181818
pedestrian |	0.504789090909	| 0.419259909091 | 	0.402404181818
cyclist |	0.529568454545	| 0.317203 |	0.313888272727

1. Epoch 10


## Structure
This repository provides Faster R-CNN as a package named `rcnn`.
  * `rcnn.core`: core routines in Faster R-CNN training and testing.
  * `rcnn.cython`: cython speedup from py-faster-rcnn.
  * `rcnn.dataset`: dataset library. Base class is `rcnn.dataset.imdb.IMDB`.
  * `rcnn.io`: prepare training data.
  * `rcnn.processing`: data and label processing library.
  * `rcnn.symbol`: symbol and operator.
  * `rcnn.test`: test utilities.
  * `rcnn.tools`: training and testing wrapper.
  * `rcnn.utils`: utilities in training and testing, usually overloads mxnet functions.

## Getting Started
* Install python package `cython`, `cv2`, `easydict`, `matplotlib`, `numpy`.
* Install [TuSimple/MXNet](https://github.com/TuSimple/mxnet) and [Python interface](http://mxnet.io/get_started/ubuntu_setup.html).
* Run `make` in `HOME`
* Suppose `HOME` represents where this file is located. All commands, unless stated otherwise, should be started from `HOME`.
* Make a folder `model` in `HOME`. `model` folder will be used to place model checkpoints along the training process.
  It is recommended to make `model` as a symbolic link to some place in hard disk.
* `prefix` refers to the first part of a saved model file name and `epoch` refers to a number in this file name.
  In `e2e-0001.params`, `prefix` is `"e2e"` and `epoch` is `1`.
* `begin_epoch` means the start of your training process, which will apply to all saved checkpoints.

## Training Faster R-CNN
All scripts default to VOC data and VGG network. They have common parameters `--network` and
`--dataset`.

### Prepare Training Data
All dataset have three attributes, `image_set`, `root_path` and `dataset_path`.  
* `image_set` could be `2007_trainval` or something like `2007trainval+2012trainval`.  
* `root_path` is usually `data`, where `cache`, `selective_search_data`, `rpn_data` will be stored.  
* `dataset_path` could be something like `data/VOCdevkit`, where images, annotations and results can be put so that many copies of datasets can be linked to the same actual place.    
See `doc/kitti.md` and `doc/custom.md`.

### Prepare Pretrained Models
* Download VGG16 pretrained model from [MXNet model gallery](https://github.com/dmlc/mxnet-model-gallery/blob/master/imagenet-1k-vgg.md),
  rename `vgg16-0000.params` to `vgg16-0001.params` and place it in `model` folder.
* Download ResNet pretrained model from [ResNet](https://github.com/tornadomeet/ResNet).
  Download `resnet-101-0000.params` to `model` folder. Other networks like resnet-152 require change in `rcnn.symbol.resnet`.

### Alternate Training
* Start training by running `python train_alternate.py`. This will train the VGG network on the VOC07 trainval.
  More control of training process can be found in the argparse help.
* Start testing by running `python test.py --prefix model/final --epoch 0` after completing the training process.
  This will test the VGG network on the VOC07 test with the model in `HOME/model/final-0000.params`.
  Adding a `--vis` will turn on visualization and `-h` will show help as in the training process.

### End-to-end Training (approximate process)
* Start training by running `python train_end2end.py`. This will train the VGG network on VOC07 trainval.
* Start testing by running `python test.py`. This will test the VGG network on the VOC07 test.

## Disclaimer
This repository used code from [MXNet](https://github.com/dmlc/mxnet),
[Fast R-CNN](https://github.com/rbgirshick/fast-rcnn),
[Faster R-CNN](https://github.com/rbgirshick/py-faster-rcnn),
[caffe](https://github.com/BVLC/caffe),
[tornadomeet/mx-rcnn](https://github.com/tornadomeet/mx-rcnn).  
Training data are from
[Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/),
[ImageNet](http://image-net.org/).  
Model comes from
[VGG16](http://www.robots.ox.ac.uk/~vgg/research/very_deep/),
[ResNet](https://github.com/tornadomeet/ResNet).  
Thanks to tornadomeet for end-to-end experiments and MXNet contributers for helpful discussions.

## References
1. Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. "Faster R-CNN: Towards real-time object detection with region proposal networks." In Advances in Neural Information Processing Systems, 2015.
2. Karen Simonyan, and Andrew Zisserman. "Very deep convolutional networks for large-scale image recognition." arXiv preprint arXiv:1409.1556 (2014).
3. Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. "Deep Residual Learning for Image Recognition". In Computer Vision and Pattern Recognition, IEEE Conference on, 2016.
