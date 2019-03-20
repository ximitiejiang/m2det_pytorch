#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 10:16:38 2019

@author: ubuntu
"""

from model.mlfpn import MLFPN
from model.m2det_head import M2detHead
import torch


if __name__ == '__main__':
    cfg_fpn = dict(backbone_type = 'SSDVGG',
                   phase = 'train',
                   size = 512,
                   planes = 256,  
                   smooth = True,
                   num_levels = 8,
                   num_scales = 6,
                   side_channel = 512
                   )
    
    mlfpn = MLFPN(**cfg_fpn)
    mlfpn.init_weights()

    feat_shallow = [torch.randn(2,512,64,64)]
    feat_deep = [torch.randn(2,1024,32,32)]
    feats = feat_shallow + feat_deep
    
    sources = mlfpn(feats)
    
    cfg_m2dethead = dict(input_size = 512,      # 相同保留
                         in_channels = [64, 32, 16, 8, 4, 2],
                         num_classes = 81,
                         step_pattern = [8, 16, 32, 64, 128, 256],  
                         size_pattern = [0.06, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05], 
                         size_featmaps = [64, 32, 16, 8, 4, 2],
                         anchor_ratio_range = ([2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]),
                         target_means=(.0, .0, .0, .0),
                         target_stds=(1.0, 1.0, 1.0, 1.0))
    head = M2detHead(**cfg_m2dethead)
    bbox_preds, cls_scores = head(sources)
    
    
    
    