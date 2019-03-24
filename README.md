# M2det_pytorch
re-implementation M2Det in pytorch based on official M2Det codes [here](https://github.com/qijiezhao/M2Det)
<div align=center><img src="https://github.com/ximitiejiang/m2det_pytorch/blob/master/data/m2det_structure_1"/></div>
<div align=center><img src="https://github.com/ximitiejiang/m2det_pytorch/blob/master/data/m2det_structure_2"/></div>
<div align=center><img src="https://github.com/ximitiejiang/m2det_pytorch/blob/master/data/base_anchors_for_6_scales.png"/></div>

## features
+ SSD algorithm is base on [simple_ssd_pytorch](https://github.com/ximitiejiang/simple_ssd_pytorch) which is extracted from mmdetection. 
+ pretrained vgg16 model weight was from caffe on [here](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/vgg16_caffe-292e1171.pth), so some slightly difference on backbone vgg16 model & mean/std setting as well.
+ support single-GPU/multi-GPU training/test
+ compatiable to mmdetection framework, the MLFPN module can be integrate to mmdetection lib. but for some reason, the interface was little different, little change may needed.

## TODO
+ [ ] compare performance with original implementation
+ [ ] support eval on voc
+ [ ] support distributed training
