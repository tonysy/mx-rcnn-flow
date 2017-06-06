# Deep Feature R-CNN

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
