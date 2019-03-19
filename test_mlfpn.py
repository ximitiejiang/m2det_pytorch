#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 10:16:38 2019

@author: ubuntu
"""

from model.mlfpn import MLFPN
import torch


if __name__ == '__main__':
    cfg_fpn = dict(backbone_type = 'SSDVGG',
                   phase = 'train',
                   size = 512,
                   planes = 
                   smooth = True,
                   num_levels = 8,
                   num_scales = 6,
                   side_channel = 512
                   )
    
    mlfpn = MLFPN(**cfg_fpn)
    mlfpn.init_weights()

    feat_shallow = torch.randn(2,512,64,64)
    feat_deep = torch.randn(2,1024,32,32)
    feats = [feat_shallow, feat_deep]
    mlfpn(feats)
    
    