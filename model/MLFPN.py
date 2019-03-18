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
    
    def __init__(self):
        pass



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
        
        # build FFM
        # TODO: need parameter backbone type
        if backbone_type == 'SSDVGG':
            shollow_in, shallow_out = 512, 256  # for vgg shallow layer output
            deep_in, deep_out = 1024, 512       # for vgg deep layer output
        self.reduce= BasicConv(
            shallow_in, shallow_out, kernel_size=3, stride=1, padding=1)
        self.up_reduce= BasicConv(
            deep_in, deep_out, kernel_size=1, stride=1)
        
        # build FFM2
        self.leach = nn.ModuleList([
            BasicConv(deep_out + shallow_out, self.planes//2, 
                      kernel_size=(1,1),stride=(1,1))]*self.num_levels)
        
        # build TUM
        self.tums = []
        for i in range(self.num_levels):
            if i == 0:
                self.tums.append(
                        TUM(first_level=True, 
                            input_planes=self.planes//2,
                            is_smooth=self.smooth,
                            scales=self.num_scales,
                            side_channel=512))
            else:
                self.tums.append(
                        TUM(first_level=False,
                            input_planes=self.planes//2,
                            is_smooth=self.smooth,
                            scales=self.num_scales,
                            side_channel=self.side_channel))
        
        # build sfam:
        if self.sfam:
            self.sfam_module = SFAM(self.planes, self.num_levels, self.num_scales, compress_ratio=16)

        
    def forward(self, x):
        """
        Args:
            x(list): feature list from vgg16, (512, 64, 64), (1024 , 32, 32)
        """
        x_shallow = self.reduce(x)
        x_deep = self.up_reduce(x)
        base_feature = torch.cat(x_shallow, 
            F.interpolate(x_deep, scale_factor=2, mode='nearest'), 1)
        tum_outs = 
        

class FPN(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 normalize=None,
                 activation=None):
        super(FPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.activation = activation
        self.with_bias = normalize is None

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                normalize=normalize,
                bias=self.with_bias,
                activation=self.activation,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                normalize=normalize,
                bias=self.with_bias,
                activation=self.activation,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

            # lvl_id = i - self.start_level
            # setattr(self, 'lateral_conv{}'.format(lvl_id), l_conv)
            # setattr(self, 'fpn_conv{}'.format(lvl_id), fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                in_channels = (self.in_channels[self.backbone_end_level - 1]
                               if i == 0 else out_channels)
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    normalize=normalize,
                    bias=self.with_bias,
                    activation=self.activation,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] += F.interpolate(
                laterals[i], scale_factor=2, mode='nearest')

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                orig = inputs[self.backbone_end_level - 1]
                outs.append(self.fpn_convs[used_backbone_levels](orig))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    # BUG: we should add relu before each extra conv
                    outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)
