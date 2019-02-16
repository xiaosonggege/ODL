#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2018-2021, Node Supply Chain Manager Corporation Limited.
@contact: 1243049371@qq.com
@software:PyCharm
@file: cnn-rnn
@time: 2019/2/11 22:16
@desc:
'''
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from AllNet import NeuralNetwork, RNN, CNN, FNN
import pandas as pd
import pickle
import os

def LoadFile(p):
    '''
    读取文件
    :param p: 数据集绝对路径
    :return: 数据集
    '''
    data = np.array([0])
    try:
        with open(p, 'rb') as file:
            data = pickle.load(file)
    except:
        print('文件不存在!')
    finally:
        return data

def onehot(label):
    '''
    将标签数据转换为one-hot编码
    :param label: 实际标签数据
    :return: 标签数据的one-hot编码形式
    '''
    one_hot = np.zeros(shape= (label.shape[0], 8))
    row = np.arange(label.shape[0], dtype= np.int8)
    column = label.astype(np.int8)
    one_hot[row, column] = 1
    return one_hot

def Databatch(dataset):
    '''
    批数据生成器
    :param dataset: 训练集数据（带标签）
    :return: type= (feature, label), 批数据
    '''
    data_size = dataset.shape[0]
    batch_size = 500
    for i in range(0, data_size-batch_size, 500):
        yield dataset[i:i+500, :225], dataset[i:i+500, -1]

def cnn(x, is_training):
    '''
    :param x: 数据特征占位符
    :param is_training: 标志训练和测试时段
    搭建cnn模块
    :return: cnn最后一层flat后的节点
    '''
    #1
    cnn_1 = CNN(x= x, y= None, w_conv= (3, 3, 1, 96), stride_conv= 1, stride_pool= 2, loss= None, optimizer= None)
    x_new = cnn_1.reshape(f_vector= x, new_shape= (-1, 15, 15, 1))
    conv1 = cnn_1.convolution(op_outside= x_new)# + tf.Variable(tf.truncated_normal(shape= ([96]), mean= 0, stddev= 1), dtype= tf.float32) #-1*15*15*96
    relu1 = tf.nn.relu(conv1)
    bn1 = cnn_1.batch_normoalization(is_training= is_training)
    pool1 = cnn_1.pooling(pool_fun= tf.nn.max_pool, input= relu1) #-1*8*8*96
    #2
    cnn_2 = CNN(x= pool1, y= None, w_conv= (3, 3, 96, 256), stride_conv= 1, stride_pool= 2, loss= None, optimizer= None)
    conv2 = cnn_2.convolution()# + tf.Variable(tf.truncated_normal(shape= ([256]), mean= 0, stddev= 1), dtype= tf.float32) #-1*8*8*256
    relu2 = tf.nn.relu(conv2)
    bn2 = cnn_2.batch_normoalization(is_training= is_training)
    pool2 = cnn_1.pooling(pool_fun=tf.nn.max_pool, input=relu2) #-1*4*4*256
    #3
    cnn_3 = CNN(x=pool2, y=None, w_conv=(3, 3, 256, 384), stride_conv=1, stride_pool=2, loss=None, optimizer=None)
    conv3 = cnn_3.convolution()# + tf.Variable(tf.truncated_normal(shape=([384]), mean=0, stddev=1), dtype=tf.float32) #-1*4*4*384
    relu3 = tf.nn.relu(conv3)
    bn3 = cnn_3.batch_normoalization(is_training= is_training)
    pool3 = cnn_1.pooling(pool_fun=tf.nn.max_pool, input=relu3) #-1*2*2*384
    x, y, z = pool3.get_shape().as_list()[1:]
    output = tf.reshape(pool3, shape= (-1, x*y*z)) #-1*2*2*384
    return output

def lstm(x, max_time, num_units):
    '''
    搭建lstm模块
    :param x: Tensor, 卷积模块输入节点
    :param max_time: 最大循环次数
    :param num_units: 隐藏层向量维度\
    :return: lstm最后时刻输出节点，数据标签占位符
    '''
    rnn = RNN(x= x, y= None, loss= None, optimizer= None, max_time= max_time, num_units= num_units)
    outputs, _ = rnn.dynamic_rnn(style= 'LSTM', output_keep_prob= 0.8)
    return outputs[:, -1, :]

def main(dataset_train, dataset_test):
    '''
    循环网络后的输出层以及训练过程
    :param dataset_train: ndarray, shape= (None, 226), 实际训练数据特征和标签
    :param dataset_test: ndarray, shape= (None, 226), 实际测试数据特征
    :return: None
    '''
    # 建立计算图
    g1 = tf.Graph()
    with g1.as_default():
        x = tf.placeholder(dtype=tf.float32, shape=(None, 225))
        y = tf.placeholder(dtype=tf.float32, shape=(None, 8))  # 需要对数据数据进行one-hot编码
        is_training = tf.placeholder(dtype=tf.bool)

        # lstm网络输出与类别交互
        para_size = {
            'size_in': 192,
            'size_out': 8
        }
        fc_para = {
            'w1': tf.Variable(
                tf.truncated_normal(shape=(para_size['size_in'], para_size['size_out']), mean=0, stddev=1),
                dtype=tf.float32),
            'b1': tf.Variable(tf.truncated_normal(shape=([para_size['size_out']]), mean=0, stddev=1), dtype=tf.float32)
        }

        # cnn-rnn
        output = cnn(x, is_training)
        op = lstm(x=output, max_time=8, num_units=192)
        fc = tf.matmul(op, fc_para['w1']) + fc_para['b1']
        fc = tf.nn.relu(fc)

        # 定义softmax交叉熵和损失函数以及精确度函数
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fc, labels=y))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-4).minimize(loss)
        acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(tf.nn.softmax(fc), 1), tf.argmax(y, 1)), tf.float32))

        init = tf.global_variables_initializer()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph= g1) as sess:
        sess.run(init)
        for epoch in range(100000):
            #制作一个数据的生成器可以像mnist的next_batch函数一样
            for xs, ys in Databatch(dataset= dataset_train):
                #将标签转化成one-hot编码
                ys = onehot(label= ys)
                _, loss_ = sess.run([optimizer, loss], feed_dict= {x: xs, y: ys, is_training: True})
            if not (epoch % 100):
                x_test = dataset_test[:, :225]
                y_test = onehot(label= dataset_test[:, -1])
                acc_ = sess.run(acc, feed_dict= {x: x_test, y: y_test, is_training: False})
                print('第%s轮训练后测试集上的预测精度为: %s' % (epoch, acc_))


if __name__ == '__main__':
    dataset_train = LoadFile(p= r'F:\ODL\dataset\data_train')
    dataset_test = LoadFile(p= r'F:\ODL\dataset\data_test')
    main(dataset_train= dataset_train, dataset_test= dataset_test)





#     def SaveFile(data, savepickle_p):
#         '''
#         存储整理好的数据
#         :param data: 待存储数据
#         :param savepickle_p: pickle后缀文件存储绝对路径
#         :return: None
#         '''
#         if not os.path.exists(savepickle_p):
#             with open(savepickle_p, 'wb') as file:
#                 pickle.dump(data, file)
#
#     def LoadFile(p):
#         '''
#         读取文件
#         :param p: 数据集绝对路径
#         :return: 数据集
#         '''
#         data = np.array([0])
#         try:
#             with open(p, 'rb') as file:
#                 data = pickle.load(file)
#         except:
#             print('文件不存在!')
#         finally:
#             return data
#
#     p_1 = r'F:\GraduateDesigning\featrure\original_data_Label1_features.xlsx'
#     dataset = pd.read_excel(p_1)
#     data = np.array(dataset)
#     np.random.shuffle(data)
#     data = data[:3000, :]
#     alldata = np.hstack((data, np.ones(shape=(data.shape[0], 1))))
#     print('alldata维度为:', alldata.shape)
#     data_train, data_test = data[:2500, :], data[2500:3000, :]
#     for num in range(2, 9):
#         p = r'F:\GraduateDesigning\featrure\original_data_Label%s_features.txt' % num
#         dataset = np.loadtxt(p)
#         # np.random.shuffle(dataset)
#         dataset = dataset[:3000, :]
#         data_fin = np.hstack((dataset, np.ones(shape=(dataset.shape[0], 1)) * num))
#         print('data_fin的维度为:', data_fin.shape)
#         data_fin_train, data_fin_test = data[:2500, :], data[2500:3000, :]
#         data_train = np.vstack((data_train, data_fin_train))
#         data_test = np.vstack((data_test, data_fin_test))
#
#     print('data_train, data_test的维度分别为:', data_train.shape, data_test.shape)
#
#     p_train = r'F:\ODL\dataset\data_train'
#     p_test = r'F:\ODL\dataset\data_test'
#     SaveFile(data= data_train, savepickle_p= p_train)
#     SaveFile(data= data_test, savepickle_p= p_t
