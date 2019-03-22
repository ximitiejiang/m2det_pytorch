#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 17:44:36 2019

@author: ubuntu
"""

from model.checkpoint import load_checkpoint
from model.vgg import VGG


model = VGG(
            depth=16,
            with_bn=False,
            num_classes=-1,
            num_stages=5,
            dilations=(1, 1, 1, 1, 1),
            out_indices=(0, 1, 2, 3, 4),
            frozen_stages=-1,
            bn_eval=True,
            bn_frozen=False,
            ceil_mode=False,
            with_last_pool=)

            depth,
            with_last_pool=with_last_pool,
            ceil_mode=ceil_mode,
            out_indices=out_indices

checkpoint = load_checkpoint(
        model, filename, map_location=None, strict=False, logger=None):