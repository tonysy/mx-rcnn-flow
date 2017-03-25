1. Make folder `images`, `imglists` in `data/custom`.
2. imglist format  
Multiple classes must be stated separately. The annotation file does not contain names of classes.
Each line can be separated by a semicolon to (n + 1) parts where n is the number of classes.
The first part is a path to an image, where the root of this path is `images` folder.
Followed are bounding box coordinates of each class, a series of numbers in the (x1, y1, x2, y2, class_ind) order and the numbers must come in multiple of 5. Numbers are separated by space.
An example of annotation would be `I00000.jpg 0 0 1 1 1 1 1 2 2 2`, which indicates the appearance of bbox [0, 0, 1, 1] of class 1 and [1, 1, 2, 2] of class 2.
```
filename annotations
annotation = "x1 y1 x2 y2 cls_ind" --> {left, top, right, bottom, cls_ind}
cls_ind = index from {__background__, car, ped, cyc}
```
