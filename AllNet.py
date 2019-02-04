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

    __slots__ = ('__x', '__y', '__loss', '__optimizer')

    def __init__(self, x, y, loss, optimizer):
        '''
        神经网络构造函数
        :param x: 单一数据特征
        :param y: 单一数据标签
        :param loss: type= Function, 损失函数对象
        :param optimizer: type= Function, 优化函数对象
        '''
        self.__x = x
        self.__y = y
        self.__loss = loss
        self.__optimizer = optimizer

    def AttributePrint(self):
        print('x: %s, y: %s, loss: %s, optimizer: %s' % (self.__x, self.__y, self.__loss, self.__optimizer))

    def loss(self): #待更新
        ''''''
        loss = None
        return loss

    def optimizer(self): #待更新
        ''''''
        optimizer = None
        return optimizer

class FNN(NeuralNetwork):
    __slots__ = ('__x', '__y', '__w')

    @staticmethod
    def fc_layer(para, w, b, keep_prob):
        '''
        :param para: shape= (1, den)单层输入
        :param w: shape= (den, den_w), 参数矩阵
        :param b: shape= (den_w, )偏置矩阵
        单层全连接层,加入dropout和relu操作
        :return: op, 单层节点
        '''
        h = tf.matmul((para, w)) + b
        h = tf.nn.dropout(h, keep_prob)
        h = tf.nn.relu(h)
        return h

    def __init__(self, x, y, loss, optimizer, w):
        '''
        全连接网络构造函数
        :param x: 单一数据特征
        :param y: 单一数据标签
        :param loss: type= Function, 损失函数对象
        :param optimizer: type= Function, 优化函数对象
        :param w: types = ((W, bia),..., ), W, b为参数矩阵和偏置矩阵
        '''
        super(FNN, self).__init__(x, y, loss, optimizer)
        self.__w = w

    def fc_concat(self, keep_prob):
        '''
        构建全连接网络部分组合
        :return: op, 全连接网络部分输出节点
        '''
        initial = 1
        fc_ops = None
        for parameters in self.__w:
            w, b = parameters
            if not initial:
                fc_ops = FNN.fc_layer(para= self.__x, w= w, b= b, keep_prob= keep_prob)
                initial = 0
            else:
                fc_ops = FNN.fc_layer(para= fc_ops, w= w, b= b, keep_prob= keep_prob)

        return fc_ops


class CNN(NeuralNetwork):
    __slots__ = ('__x', '__y', '__w_conv', '__w_pool', '__stride_conv', '__stride_pool')

    @staticmethod
    def reshape(f_vector, new_shape):
        '''
        对输入ndarray类型张量进行维度变换
        :param f_vector: type= array(ndarray), 待处理特征向量
        :param new_shape: 变换后维度
        :return: 处理后的特征向量
        '''
        return f_vector.reshape(new_shape)

    @staticmethod
    def flat(tensor):
        '''
        将高维张量维度降至二维
        :param tensor: type= Variable, 待处理张量
        :return: 维度坍塌后的低维张量
        '''
        #张量维度
        dimension = tensor.get_shape().as_list() #type= list
        all_dim = np.multiply.reduce(np.array(dimension))
        return tf.reshape(tensor, shape= (all_dim, )) #type= Variable

    def __init__(self, x, y, w_conv, w_pool, stride_conv, stride_pool, loss, optimizer):
        '''
        卷积神经网络构造函数
        :param x: 单一数据特征
        :param y: 单一数据标签
        :param w_conv: type= list, 单个卷积核维度矩阵
        :param w_pool: type= list, 单个池化核维度矩阵
        :param stride_conv: 卷积核移动步伐
        :param stride_pool: 池化核移动步伐
        :param loss: type= Function, 损失函数对象
        :param optimizer: type= Function, 优化函数对象
        '''
        super(CNN, self).__init__(x, y, loss, optimizer)
        self.__w_conv = w_conv
        self.__w_pool = w_pool
        self.__stride_conv = stride_conv
        self.__stride_pool = stride_pool

    def convolution(self):
        '''
        构建卷积核
        :return:
        '''





