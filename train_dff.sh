# python train_end2end_dff.py --dataset Kitti --gpu 6,7 --image_set train --dataset_path data/kitti --prefix model/e2e-fix-flow-anchorloader  --end_epoch 36 --pretrained model/e2e-rcnn --pretrained_epoch 10
#--begin_epoch 6  --resume

# python train_end2end_dff.py --dataset Kitti --gpu 6,7 --image_set train --dataset_path data/kitti --prefix /data/syzhang/rcnn_dff/model/e2e-non-fix-flow --end_epoch 37 --pretrained /data/syzhang/rcnn_dff/model/e2e-fix-flow-anchorloader-train --pretrained_epoch 22 --resume  --begin_epoch 21

python train_end2end_dff.py --dataset Kitti --gpu  7 --image_set train --dataset_path data/kitti --prefix /data/syzhang/rcnn_dff/model/e2e-ffa-new-k1 --end_epoch 15 --pretrained /data/syzhang/model/e2e-rcnn-log --pretrained_epoch 10
