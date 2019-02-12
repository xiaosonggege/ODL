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

    @staticmethod
    def flat(tensor):
        '''
        将高维张量维度降至二维
        :param tensor: type= Variable, 待处理张量
        :return: 维度坍塌后的低维张量
        '''
        # 张量维度
        dimension = tensor.get_shape().as_list()  # type= list
        all_dim = np.multiply.reduce(np.array(dimension))
        return tf.reshape(tensor, shape=(all_dim,))  # type= Variable

    def __init__(self, x, y, loss, optimizer):
        '''
        神经网络构造函数
        :param x: 单一数据特征
        :param y: 单一数据标签
        :param loss: type= Function, 损失函数对象
        :param optimizer: type= Function, 优化函数对象
        '''
        self.x = x
        self.y = y
        self.__loss = loss
        self.__optimizer = optimizer

    # def AttributePrint(self):
    #     print('x: %s, y: %s, loss: %s, optimizer: %s' % (self.__x, self.__y, self.__loss, self.__optimizer))

    def loss(self): #待更新
        ''''''
        loss = None
        return loss

    def optimizer(self): #待更新
        ''''''
        optimizer = None
        return optimizer

class FNN(NeuralNetwork):

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
        :param x: Tensor, 单一数据特征
        :param y: Tensor, 单一数据标签
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
                fc_ops = FNN.fc_layer(para= self.x, w= w, b= b, keep_prob= keep_prob)
                initial = 0
            else:
                fc_ops = FNN.fc_layer(para= fc_ops, w= w, b= b, keep_prob= keep_prob)

        return fc_ops


class CNN(NeuralNetwork):

    @staticmethod
    def reshape(f_vector, new_shape):
        '''
        对输入Tensor类型张量进行维度变换
        :param f_vector: type= Tensor, 待处理特征向量
        :param new_shape: iterable, 变换后维度
        :return: 处理后的特征向量
        '''
        return tf.reshape(f_vector, new_shape)

    def __init__(self, x, y, w_conv, stride_conv, stride_pool, loss, optimizer):
        '''
        卷积神经网络构造函数
        :param x: Tensor, 单一数据特征
        :param y: Tensor, 单一数据标签
        :param w_conv: type= iterable, 单个卷积核维度(4维)
        :param stride_conv: 卷积核移动步伐
        :param stride_pool: 池化核移动步伐
        :param loss: type= Function, 损失函数对象
        :param optimizer: type= Function, 优化函数对象
        '''
        super(CNN, self).__init__(x, y, loss, optimizer)
        self.__w_conv = w_conv
        self.__stride_conv = stride_conv
        self.__stride_pool = stride_pool

    def convolution(self, op_outside= 'x'):
        '''
        单层卷积操作
        :param op_outside: setdefult:x, 输入待进行卷积操作节点
        :return: ops, 单层卷积操作后节点
        '''
        input = op_outside if op_outside != 'x' else self.x
        filter_initial = tf.Variable(tf.truncated_normal(shape= self.__w_conv, mean= 0, stddev= 1)) #mean、stddev可更改
        return tf.nn.conv2d(input= input, filter= filter_initial, strides= [1, self.__stride_conv, self.__stride_conv, 1], padding= 'SAME')

    def pooling(self, pool_fun, input):
        '''
        单层池化操作
        :param input: 输入节点
        :param pool_fun: 池化函数
        :return: 单层池化操作后节点
        '''
        return pool_fun(value= input, ksize= [1, self.__stride_pool, self.__stride_pool, 1],
                        strides= [1, self.__stride_pool, self.__stride_pool, 1], padding= 'SAME')

class RNN(NeuralNetwork):

    @staticmethod
    def get_a_cell(num_units, style):
        '''
        制作一个LSTM/GRU节点
        :param num_units: 隐藏层向量维度
        :param style: 网络名称
        :return: ops, 循环网络节点
        '''

        return tf.nn.rnn_cell.LSTMCell(num_units= num_units) if style == 'LSTM' else tf.nn.rnn_cell.GRUCell(num_units= num_units)

    @staticmethod
    def reshape(x, max_time):
        '''
        对输入Tensor特征进行维度转换
        :param x: type: Tensor, 单一特征数据
        :param max_time: 最大循环次数
        :return: 维度转换后的特征
        '''
        den_3 = x.get_shape().as_list()[-1] // max_time
        para_shape = (-1, max_time, den_3)
        return tf.reshape(x, para_shape)

    def __init__(self, x, y, loss, optimizer, max_time, num_units):
        '''
        循环网络构造函数
        :param x: Tensor, 单一特征数据
        :param y: Tensor, 单一数据标签
        :param loss: 损失函数
        :param optimizer: 优化函数
        :param max_time: 最大循环次数
        :param num_units: 隐藏层向量维度
        '''
        super(RNN, self).__init__(x, y, loss, optimizer)
        self.__max_time = max_time
        self.__num_units = num_units

    def dynamic_rnn(self, style, output_keep_prob):
        '''
        按时间步展开计算循环网络
        :param style: LSTM/GRU
        :param output_keep_prob: rnn节点中dropout概率
        :return: 各个时间步输出值和最终时间点输出值
        '''
        cell = RNN.get_a_cell(num_units= self.__num_units, style= style)
        #添加在循环网络中加入dropout操作
        cell = tf.nn.rnn_cell.DropoutWrapper(cell= cell, input_keep_prob= 1.0, output_keep_prob= output_keep_prob)
        #将原始输入数据变换维度
        x_in = RNN.reshape(x= self.x, max_time= self.__max_time)
        outputs, fin_state = tf.nn.dynamic_rnn(cell, x_in, dtype= tf.float32)
        return outputs, fin_state

if __name__ == '__main__':
    rnn = RNN(1, 2, 3, 4, 5, 6)






