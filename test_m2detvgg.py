#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 10:16:38 2019

@author: ubuntu
"""

from model.m2detvgg import M2detVGG
from utils.config import Config
import torch

if __name__ == '__main__':
    path = 'config/cfg_m2det512_vgg16_coco.py'
    cfg = Config.fromfile(path)
    
    cfg.model.backbone.pop('type')
    m2detvgg = M2detVGG(**cfg.model.backbone)
    print(m2detvgg)
    
    # data container是什么时候去掉的？
    img = torch.randn(3,512,512)
    img_meta
    
    data = dict()
    m2detvgg(img)