#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 16:05:16 2019

@author: ubuntu
"""

from .one_stage_detector import OneStageDetector
from utils.registry_build import registered


@registered.register_module
class M2detDetector(OneStageDetector):

    def __init__(self, cfg):  # 输入参数修改成cfg，同时预训练模型参数网址可用了
        super(M2detDetector, self).__init__(cfg)
