# Faster R-CNN

This version is based on [mx-rcnn](https://github.com/precedenceguo/mx-rcnn).

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

kitti | easy | medium | hard
:---: | :---: | :----: | :---:
car 	| 0.796542363636	| 0.638948727273	| 0.555344363636
pedestrian 	| 0.496717818182	| 0.414531636364	| 0.401392
cyclist 	| 0.532543363636	| 0.317411909091	| 0.316632636364

### 4. Finetuning RCNN-DFF-non-fix-flow
#### Method:
- Finetuning(use prev frame) on pre-trained(use next frame) model
- Use `mx.symbol.GridGenerator` and `mx.symbol.BilinearSampler`
- `config.FLOW_SCALE_FACTOR = 0.00390625`

#### Results
1. Epoch 15

kitti | easy | medium | hard
:---: | :---: | :----: | :---:
car  | 0.839839454545 | 0.661167545455 | 0.576057181818
pedestrian  | 0.581605454545 | 0.429004272727 | 0.416503818182
cyclist  | 0.510314818182 | 0.305621090909 | 0.301150545455

## References
1. Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. "Faster R-CNN: Towards real-time object detection with region proposal networks." In Advances in Neural Information Processing Systems, 2015.
2. Karen Simonyan, and Andrew Zisserman. "Very deep convolutional networks for large-scale image recognition." arXiv preprint arXiv:1409.1556 (2014).
3. Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. "Deep Residual Learning for Image Recognition". In Computer Vision and Pattern Recognition, IEEE Conference on, 2016.
