# model settings
input_size = 512
model = dict(
    type='M2detDetector',
#    pretrained='weights/m2det/vgg16_reducedfc.pth', # 这里不能采用原有M2det的预训练模型，原因是1.原有预训练模型是pytorch模型，跟mean/std不一样；2.原有预训练参数跟当前使用的VGG模型层的命名不一致，不带'feature'，导致初始化实际上没有用
    pretrained='weights/m2det/vgg16_caffe-292e1171.pth',
    backbone=dict(
        type='M2detVGG',
        input_size=input_size,
        depth=16,
        with_last_pool=False,
        ceil_mode=True,
        out_indices=(3, 4),
        out_feature_indices=(22, 34),
        l2_norm_scale=20),
    neck=dict(
        type='MLFPN',
        backbone_type='M2detVGG',
        input_size=input_size,
        planes=256,       # 代表每个tum的输出layers
        smooth=True,      # 代表tum是否包含smooth conv(1x1)
        num_levels=8,     # 代表多少个tums
        num_scales=6,     # 代表每个tum输出多少scales
        side_channel=512,
        sfam=False,     # 是否含sfam模块
        compress_ratio=16),
    bbox_head=dict(
        type='M2detHead',
        input_size=input_size,
        planes = 256,
        num_levels=8,
        num_classes=81,
        anchor_strides=(8, 16, 32, 64, 100, 300),
        size_pattern = [0.06, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05],  # 代表anchors尺寸跟img的比例, 前6个数是min_size比例，后6个数是max_size比例，可以此计算anchor最小最大尺寸
        size_featmaps = [(64,64), (32,32), (16,16), (8,8), (4,4), (2,2)],
        anchor_ratio_range = ([2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]),
        target_means=(.0, .0, .0, .0),
        target_stds=(1.0, 1.0, 1.0, 1.0)))
cudnn_benchmark = True
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.5,
        min_pos_iou=0.,
        ignore_iof_thr=-1,
        gt_max_assign_all=False),
    smoothl1_beta=1.,
    allowed_border=-1,
    pos_weight=-1,
    neg_pos_ratio=3,
    debug=False)
test_cfg = dict(
    nms=dict(type='nms', iou_thr=0.45),
    min_bbox_size=0,
    score_thr=0.02,
    max_per_img=200)
# model training and testing settings
# dataset settings
dataset_type = 'CocoDataset'
data_root = './data/coco/'
# 预训练模型是否跟该mean/std不匹配？？？当前预训练模型是ssd的pytorch版本，是否应该换成caffe版本？这样mean/std不用换
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[1, 1, 1], to_rgb=True)
data = dict(
    imgs_per_gpu=2,  # 从4改成2
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/instances_train2017.json',
            img_prefix=data_root + 'train2017/',
            img_scale=(512, 512),
            img_norm_cfg=img_norm_cfg,
            size_divisor=None,
            flip_ratio=0.5,
            with_mask=False,
            with_crowd=False,
            with_label=True,
            test_mode=False,
            extra_aug=dict(
                photo_metric_distortion=dict(
                    brightness_delta=32,
                    contrast_range=(0.5, 1.5),
                    saturation_range=(0.5, 1.5),
                    hue_delta=18),
                expand=dict(
                    mean=img_norm_cfg['mean'],
                    to_rgb=img_norm_cfg['to_rgb'],
                    ratio_range=(1, 4)),
                random_crop=dict(
                    min_ious=(0.1, 0.3, 0.5, 0.7, 0.9), min_crop_size=0.3)),
            resize_keep_ratio=False)),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        img_scale=(512, 512),
        img_norm_cfg=img_norm_cfg,
        size_divisor=None,
        flip_ratio=0,
        with_mask=False,
        with_label=False,
        test_mode=True,
        resize_keep_ratio=False),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        img_scale=(512, 512),
        img_norm_cfg=img_norm_cfg,
        size_divisor=None,
        flip_ratio=0,
        with_mask=False,
        with_label=False,
        test_mode=True,
        resize_keep_ratio=False))
# optimizer
optimizer = dict(type='SGD', lr=4e-4, momentum=0.9, weight_decay=5e-4)  # 学习率是8块GPU的，所以在1块GPU下从2e-3改为了2e-4
optimizer_config = dict()
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[16, 22])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
gpus=2
total_epochs = 10
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/ssd300_coco'
load_from = None
resume_from = None
workflow = [('train', 1)]
