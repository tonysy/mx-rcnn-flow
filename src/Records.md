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

### 5. Finetuning RCNN-Cycle-Flow
#### RCNN-Cycle-Flow Structure
```
Prev_Image   ------>   Curr_Image
                           |
                           |
                           v
Prev_Feature <-----  Curr_Feature

Curr_Image   ------>   Prev_Image
                           |
                           |
                           v
Curr_Feature <-----  Prev_Feature
```

#### Results
kitti	| easy	| medium	| hard
:---: | :---: | :----: | :---:
car 	| 0.872972	| 0.751366636364 | 	0.602904090909
pedestrian 	| 0.583609090909 | 	0.501473727273 | 	0.420809636364
cyclist 	| 0.574077454545	| 0.393106181818	| 0.334247727273

### 6. RCNN-FFA(RCNN-Flow-Feature-Aggregation)
#### RCNN-Train:
- K = 2

```
Prev_2 ---- Prev_1 ---- Curr_Image ---- Next_1 ---- Next_2

```
#### Results
- K = 2, epoch = 10

kitti	| easy	| medium	| hard
:---: | :---: | :----: | :---:
car  |	0.865556363636 |	0.675827363636 |	0.587821727273
pedestrian | 	0.602180909091 |	0.513320181818 |	0.430714636364
cyclist | 	0.553921363636 |	0.377344090909	 | 0.325980272727


- K = 3

kitti	| easy	| medium	| hard
:---: | :---: | :----: | :---:
car 	  |	0.853539909091  |		0.669660545455  |		0.580927818182
pedestrian   |		0.598009545455  |		0.437202181818  |		0.429093
cyclist   |		0.537169181818  |		0.322616454545  |		0.314224

- K = 4

kitti	| easy	| medium	| hard
:---: | :---: | :----: | :---:
car 	  |	0.834104636364	  |	0.653766818182  |		0.564943
pedestrian 	  |	0.525435727273  |		0.438181090909  |		0.430116727273
cyclist   |		0.484703  |		0.315549818182	  |	0.313029727273

- K = 0
kitti	| easy	| medium	| hard
:---: | :---: | :----: | :---:
car 	| 0.750215181818		| 0.566817727273	| 	0.490937727273
pedestrian	|  	0.582343363636	| 	0.496310545455		| 0.418782909091
cyclist 	| 	0.443371090909		| 0.287288636364		| 0.285888181818


## References
1. Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. "Faster R-CNN: Towards real-time object detection with region proposal networks." In Advances in Neural Information Processing Systems, 2015.
2. Karen Simonyan, and Andrew Zisserman. "Very deep convolutional networks for large-scale image recognition." arXiv preprint arXiv:1409.1556 (2014).
3. Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. "Deep Residual Learning for Image Recognition". In Computer Vision and Pattern Recognition, IEEE Conference on, 2016.
