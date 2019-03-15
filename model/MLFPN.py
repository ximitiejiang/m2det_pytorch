#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 15:00:07 2019

@author: ubuntu
"""
import torch.nn as nn


class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, 
                 padding=0, dilation=1, groups=1, relu=True, bn=True, 
                 bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, 
                stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class TUM(nn.Module):
    def __init__(self, first_level=True, input_planes=128, is_smooth=True, side_channel=512, scales=6):
        super(TUM, self).__init__()
        self.is_smooth = is_smooth
        self.side_channel = side_channel
        self.input_planes = input_planes
        self.planes = 2 * self.input_planes
        self.first_level = first_level
        self.scales = scales
        self.in1 = input_planes + side_channel if not first_level else input_planes

        self.layers = nn.Sequential()
        self.layers.add_module('{}'.format(len(self.layers)), BasicConv(self.in1, self.planes, 3, 2, 1))
        for i in range(self.scales-2):
            if not i == self.scales - 3:
                self.layers.add_module(
                        '{}'.format(len(self.layers)),
                        BasicConv(self.planes, self.planes, 3, 2, 1)
                        )
            else:
                self.layers.add_module(
                        '{}'.format(len(self.layers)),
                        BasicConv(self.planes, self.planes, 3, 1, 0)
                        )
        self.toplayer = nn.Sequential(BasicConv(self.planes, self.planes, 1, 1, 0))
        
        self.latlayer = nn.Sequential()
        for i in range(self.scales-2):
            self.latlayer.add_module(
                    '{}'.format(len(self.latlayer)),
                    BasicConv(self.planes, self.planes, 3, 1, 1)
                    )
        self.latlayer.add_module('{}'.format(len(self.latlayer)),BasicConv(self.in1, self.planes, 3, 1, 1))

        if self.is_smooth:
            smooth = list()
            for i in range(self.scales-1):
                smooth.append(
                        BasicConv(self.planes, self.planes, 1, 1, 0)
                        )
            self.smooth = nn.Sequential(*smooth)

    def _upsample_add(self, x, y, fuse_type='interp'):
        _,_,H,W = y.size()
        if fuse_type=='interp':
            return F.interpolate(x, size=(H,W), mode='nearest') + y
        else:
            raise NotImplementedError
            #return nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1)

    def forward(self, x, y):
        if not self.first_level:
            x = torch.cat([x,y],1)
        conved_feat = [x]
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            conved_feat.append(x)
        
        deconved_feat = [self.toplayer[0](conved_feat[-1])]
        for i in range(len(self.latlayer)):
            deconved_feat.append(
                    self._upsample_add(
                        deconved_feat[i], self.latlayer[i](conved_feat[len(self.layers)-1-i])
                        )
                    )
        if self.is_smooth:
            smoothed_feat = [deconved_feat[0]]
            for i in range(len(self.smooth)):
                smoothed_feat.append(
                        self.smooth[i](deconved_feat[i+1])
                        )
            return smoothed_feat
        return deconved_feat

class SFAM(nn.Module):
    def      



class MLFPN(nn.Module):
    """创建Multi Layers Feature Pyramid Net
    1. TUM: 类似与unet/fpn的多级特征融合模块
    """
    def __init__(self, phase, size, planes, smooth=True, num_scales=6, side_channel=512):
        super().__init__()
        self.phase = phase
        self.size = size
        self.planes = planes
        self.smooth = smooth
        self.num_scales = num_scales
        self.side_channel = side_channel
        
        for i in range(self.num_levels):
            if i == 0:
                TUM(first_level=True,
                    input_planes=self.planes//2,
                    is_smooth=self.smooth,
                    scales=self.num_scales,
                    side_channel=512)
            else:
                TUM(first_level=False,
                    input_planes=self.planes//2,
                    is_smooth=self.smooth,
                    scales=self.num_scales,
                    side_channel=self.side_channel)
        
        shollow_in, shallow_out = 512, 256
        deep_in, deep_out = 1024, 512
        self.reduce= BasicConv(
                shallow_in, shallow_out, kernel_size=3, stride=1, padding=1)
        self.up_reduce= BasicConv(
                deep_in, deep_out, kernel_size=1, stride=1)
        
        if self.phase == 'test':
            self.softmax = nn.Softmax()
        self.Norm = nn.BatchNorm2d(256*8)
        self.leach = nn.ModuleList([BasicConv(deep_out+shallow_out,
                                              self.planes//2,
                                              kernel_size=(1,1),
                                              stride=(1,1))]*self.num_levels)
        # localization layer and recognition layers
        loc_ = list()
        conf_ = list()
        for i in range(self.num_scales):
            loc_.append(nn.Conv2d(self.planes*self.num_levels,
                                       4 * 6, # 4 is coordinates, 6 is anchors for each pixels,
                                       3, 1, 1))
            conf_.append(nn.Conv2d(self.planes*self.num_levels,
                                       self.num_classes * 6, #6 is anchors for each pixels,
                                       3, 1, 1))
        self.loc = nn.ModuleList(loc_)
        self.conf = nn.ModuleList(conf_)

        # construct SFAM module
        if self.sfam:
            self.sfam_module = SFAM(self.planes, self.num_levels, self.num_scales, compress_ratio=16)

    def forward
               