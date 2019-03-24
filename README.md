# M2det_pytorch
re-implementation M2Det in pytorch based on official M2Det codes [here](https://github.com/qijiezhao/M2Det), currently it only support 512x512 training like origianl repo.
<div align=center><img src="https://github.com/ximitiejiang/m2det_pytorch/blob/master/data/m2det_structure_1.jpg"/></div>
<div align=center><img src="https://github.com/ximitiejiang/m2det_pytorch/blob/master/data/m2det_structure_2.jpg"/></div>
<div align=center><img src="https://github.com/ximitiejiang/m2det_pytorch/blob/master/data/base_anchors_for_6_scales.png"/></div>

## Features
+ SSD algorithm is base on [simple_ssd_pytorch](https://github.com/ximitiejiang/simple_ssd_pytorch) which is extracted from mmdetection. 
+ pretrained vgg16 model weight was from caffe on [here](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/vgg16_caffe-292e1171.pth), so some slightly difference on backbone vgg16 model & mean/std setting as well.
+ support single-GPU/multi-GPU training
+ compatiable to mmdetection framework, the MLFPN module can be integrate to mmdetection lib. but for some reason, the interface was little different, little change may needed.
+ from author's paper, it takes 6days for training with VGG16/512x512 on 4 Titan X, 3days for VGG16/320x320 on 4 Titan X.

## Todo
+ [ ] compare performance with original implementation
+ [ ] support 320x320 input size
+ [ ] support eval on voc
+ [ ] support distributed training
