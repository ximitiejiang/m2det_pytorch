#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 10:16:38 2019

@author: ubuntu
"""

from model.mlfpn import MLFPN
from model.m2det_head import M2detHead
import torch
import matplotlib.pyplot as plt
from addict import Dict

def draw_base_anchors(base_anchors_list):
    """base_anchors show
    base anchors_list(list): (6,) with (6,4) for each scale with coordinate of (xmin,ymin,xmax,ymax)
    """
    def draw_rect(anchor):
        plt.plot([anchor[0],anchor[2]], [anchor[1],anchor[1]])
        plt.plot([anchor[0],anchor[0]], [anchor[1],anchor[3]])
        plt.plot([anchor[0],anchor[2]], [anchor[3],anchor[3]])
        plt.plot([anchor[2],anchor[2]], [anchor[1],anchor[3]])
    # base anchors_list (6,)
    length = len(base_anchors_list)    
    for i,bas in enumerate(base_anchors_list):
        # base (6,4)
        plt.subplot(2, length/2, i+1)
        for anchor in bas: # anchor (4,)
            draw_rect(anchor)
        

if __name__ == '__main__':
    
    # 创建MLFPN
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
    
    # 创建m2det head
    cfg_m2dethead = dict(input_size = 512,      # 相同保留
                         planes = 256,
                         num_classes = 81,
                         step_pattern = [8, 16, 32, 64, 128, 256],  
                         size_pattern = [0.06, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05], 
                         size_featmaps = [64, 32, 16, 8, 4, 2],
                         anchor_ratio_range = ([2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]),
                         target_means=(.0, .0, .0, .0),
                         target_stds=(1.0, 1.0, 1.0, 1.0))
    head = M2detHead(**cfg_m2dethead)
    
    # draw base anchors for 6 scales
    base_anchors_list = [head.anchor_generators[i].base_anchors for i in range(len(head.anchor_generators))]
    draw_base_anchors(base_anchors_list)
    
    # check head outputs
    head.init_weights()
    outs = head(sources)  # outs is a tuple
    
    # check loss
    gt_bboxes = torch.tensor([[20,20,120,120]])
    gt_labels = torch.tensor([1])
    img_metas = [Dict(img_shape=(800,600))]
    train_cfg = Dict(neg_pos_ratio=3, smoothl1_beta=1)
    
    loss_inputs = outs + (gt_bboxes, gt_labels, img_metas, train_cfg)
    losses = head.loss(*loss_inputs)
    
    
    
    