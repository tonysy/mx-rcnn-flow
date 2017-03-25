1. fixed_param_prefix: see rcnn/core/module.py line 52. it is not prefix
2. config.IMAGE_STRIDE: pad image to multiples of IMAGE_STRIDE
3. max_data_shape, max_label_shape: you must allocate maximum memory before training. must be multiples of IMAGE_STRIDE
4. kitti_eval will output the last result if no results are supplied
