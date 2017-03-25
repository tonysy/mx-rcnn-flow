1. Make folder `images`, `imglists` in `data/kitti`.
2. Get images  
Download data from /rawdata/kitti_data/data_object_image_2 or http://www.cvlibs.net/download.php?file=data_object_image_2.zip
Put all training images in `images` folder, put all testing images in `images/testing` folder.
3. Get imglists  
Download example imglists from https://www.dropbox.com/s/x0gh5fxt0kn9e87/kitti_imglists.tar.gz?dl=0
4. Get evaluation toolkit  
Download from https://www.dropbox.com/s/5z562s7c1ns8umf/kitti_eval.tar.gz?dl=0
Extract it. Run `make` in `kitti_eval`.
Each time you evaluate, run
```bash
cp data/kitti/results/* kitti_eval/results/frcnn/data
./test.sh
```
5. Reference results

kitti | easy | medium | hard
:---: | :---: | :----: | :---:
car | 0.8782325455 | 0.7572223636 | 0.6334450909
pedestrian | 0.7038934545 | 0.6196088182 | 0.5497168182
cyclist | 0.7561599091 | 0.5126151818 | 0.4845294545

6. imglist format  
Multiple classes must be stated separately. The annotation file does not contain names of classes. 
Each line can be separated by a semicolon to (n + 1) parts where n is the number of classes. 
The first part is a path to an image, where the root of this path is images folder. 
Followed are bounding box coordinates of each class, a series of numbers in the (x1, y1, x2, y2) order and the numbers must come in multiple of 4. Numbers are separated by space. 
An example of annotation would be `I00000.jpg:0 0 1 1 :1 1 2 2 :`. Four columns means four classes and in the corresponding image, and there is no instance of the last two classes.
```
filename:car annotation:ped annotation:cyc annotation
annotation = "x1 y1 x2 y2 " --> {left, top, right, bottom}
```
