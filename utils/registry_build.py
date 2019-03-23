#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 16:12:54 2019

@author: ubuntu
"""
import torch.nn as nn

class Registry():
    """用于作为接收方在类初始化前把类加入字典
    1. 通过Registery类方法register_module()对module进行注册，存入module_dict中
    2. 实例化Registery类
    3. 把实例化Registery作为参数传入build_module函数，进行module创建
    """
    def __init__(self):
#        self._name = name
        self._module_dict = dict()   # 用于存放{类名：类}
    
    @property
    def module_dict(self):
        return self._module_dict
        
    def register_module(self, cls):
        """用于作为装饰器接收类，把类加入module dict，最终装饰器也返回该类"""
        if not issubclass(cls, nn.Module):
            raise TypeError(
                'module must be a child of nn.Module, but got {}'.format(type(cls)))
        
        module_name = cls.__name__    # 获得类的__name__属性
        if module_name in self._module_dict:
            raise KeyError('{} is already registered in registered dict'.format(module_name))
        self._module_dict[module_name] = cls
        
        return cls  # 装饰器返回该类

registered = Registry()  # 实例化注册器


def build_module(cfg, registered):
    args = cfg.copy()
    obj_type = args.pop('type')   # 提取type进行判断，同时从args中去除type字符串
    if obj_type not in registered.module_dict:
        raise KeyError('{} is not in the registered dict'.format(obj_type))
    else:
        obj_type = registered.module_dict[obj_type]  # 获得字符串类名对应的类
    return obj_type(**args)  # 实例化类

@registered.register_module
class TestClass(nn.Module):
    
    def __init__(self, name):
        print(name)
        
        
if __name__ == '__main__':
    from addict import Dict
    cfg = Dict(type = 'TestClass',
               name = 'MLFPN')
    mlfpn = build_module(cfg, registered)