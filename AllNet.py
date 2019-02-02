#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2018-2021, Node Supply Chain Manager Corporation Limited.
@contact: 1243049371@qq.com
@software:PyCharm
@file: AllNet
@time: 2019/2/2 23:08
@desc:
'''

import tensorflow as tf
import numpy as np

class NeuralNetwork:

    __slots__ = ('__x', '__y')

    def __init__(self):
        '''
        神经网络构造函数
        '''
        self.__x = None
        self.__y = None

    @property
    def x(self):
        return self.__x
    @x.setter
    def x(self, x):
        self.__x = x

    @property
    def y(self):
        return self.__y
    @y.setter
    def y(self, y):
        self.__y = y

class FNN(NeuralNetwork):

    def __init__(self):
        '''
        全连接网络构造函数
        :param x: 单一数据特征
        :param y: 单一数据标签
        :param w: 参数矩阵
        '''
        super(FNN, self).__init__()
        self.__w = None

    @property
    def w(self):
        return self.__w
    @w.setter
    def w(self, w):
        self.__w = w

n = NeuralNetwork()
f = FNN()
f.w = 3
f.x = 2
print(f.x)