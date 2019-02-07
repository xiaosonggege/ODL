#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2018-2021, Node Supply Chain Manager Corporation Limited.
@contact: 1243049371@qq.com
@software:PyCharm
@file: tests
@time: 2019/2/6 11:57
@desc:
'''
import numpy as np
import tensorflow as tf

class A:
    def __init__(self, x, y):
        self.__x = x
        self.y = y

class B(A):
    def __init__(self, z, x, y):
        super(B, self).__init__(x, y)
        self.z = z
        print(self.__x)

if __name__ == '__main__':
    a = A(1, 2)
    b = B(3, 1, 2)

