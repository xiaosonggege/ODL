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
from TestEvaluation import Evaluation
import pandas as pd
import pickle
from collections import Counter
import os
from sklearn.preprocessing import Imputer
from Routine_operation import SaveFile, LoadFile, SaveRestore_model, SaveImport_model

def onehot(label):
    '''
    将标签数据转换为one-hot编码
    :param label: 实际标签数据
    :return: 标签数据的one-hot编码形式
    '''
    one_hot = np.zeros(shape= (label.shape[0], 8))
    column = label.astype(np.int8)
    for row in range(label.shape[0]):
        one_hot[row, column[row]] = 1
    return one_hot

def Databatch(dataset):
    '''
    批数据生成器
    :param dataset: 训练集数据（带标签）
    :return: type= (feature, label), 批数据
    '''
    data_size = dataset.shape[0]
    batch_size = 500
    for i in range(0, data_size-batch_size+1, batch_size):
        yield dataset[i:i+batch_size, :225], dataset[i:i+batch_size, -1]

def denoising(dataset, training_time, is_finishing):
    '''
    对所有训练和测试数据进行去噪预处理
    :param dataset: shape:(-1, 225), 原始数据(不包含标签)
    :param training_time: 标记当前训练次数
    :param is_finishing: 标记是否已经训练完成
    :return: None
    '''
    LEARNING_RATE = 1e-3
    g_denoising = tf.Graph()
    with g_denoising.as_default():
        with tf.name_scope('placeholder'):
            x = tf.placeholder(shape=(None, 225), dtype=tf.float32)
        with tf.name_scope('encoder'):
            x_input = tf.reshape(tensor=x, shape=(-1, 15, 15, 1))
            # 输入(-1, 15, 15, 1)
            conv1 = tf.layers.conv2d(inputs=x_input, filters=32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu,
                                     name='conv1')  # (-1, 15, 15, 32)
            maxpool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=(2, 2), strides=(2, 2), padding='same',
                                               name='maxpool1')  # (-1, 8, 8, 32)
            conv2 = tf.layers.conv2d(inputs=maxpool1, filters=32, kernel_size=(3, 3), padding='same',
                                     activation=tf.nn.relu, name='conv2')  # (-1, 8, 8, 32)
            maxpool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=(2, 2), strides=(2, 2), padding='same',
                                               name='maxpool2')  # (-1, 4, 4, 32)
            conv3 = tf.layers.conv2d(inputs=maxpool2, filters=16, kernel_size=(3, 3), padding='same',
                                     activation=tf.nn.relu, name='conv3')  # (-1, 4, 4, 16)
            encoded = tf.layers.max_pooling2d(inputs=conv3, pool_size=(2, 2), strides=(2, 2), padding='same',
                                              name='encoded')  # (-1, 2, 2, 16)
        with tf.name_scope('decoder'):
            upsample1 = tf.image.resize_nearest_neighbor(images=encoded, size=(4, 4),
                                                         name='upsample1')  # (-1, 4, 4, 16)
            conv4 = tf.layers.conv2d(inputs=upsample1, filters=16, kernel_size=(3, 3), padding='same',
                                     activation=tf.nn.relu, name='conv4')  # (-1, 4, 4, 16)
            upsample2 = tf.image.resize_nearest_neighbor(images=conv4, size=(8, 8), name='upsample2')  # (-1, 8, 8, 16)
            conv5 = tf.layers.conv2d(inputs=upsample2, filters=32, kernel_size=(3, 3), padding='same',
                                     activation=tf.nn.relu, name='conv5')  # (-1, 8, 8, 32)
            upsample3 = tf.image.resize_nearest_neighbor(images=conv5, size=(15, 15),
                                                         name='upsample3')  # (-1, 15, 15, 32)
            conv6 = tf.layers.conv2d(inputs=upsample3, filters=32, kernel_size=(3, 3), padding='same',
                                     activation=tf.nn.relu, name='conv6')  # (-1, 15, 15, 32)
            logits = tf.layers.conv2d(inputs=conv6, filters=1, kernel_size=(3, 3), padding='same', activation=None,
                                      name='logits')  # (-1, 15, 15, 1)
        with tf.name_scope('loss-op'):
            # loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=x_input, logits=logits)
            loss = tf.reduce_mean(tf.square(x_input-logits))
            cost = tf.reduce_mean(loss)
            opt = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)
        with tf.name_scope('etc'):
            init = tf.global_variables_initializer()
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
            EPOCH = 400

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph= g_denoising) as sess:
        sess.run(init)
        # 建立checkpoint节点保存对象
        saverestore_model = SaveRestore_model(sess=sess, save_file_name='denoiseing', max_to_keep=1)
        saver = saverestore_model.saver_build()
        if training_time != 0:
            # 导入checkpoint节点，继续训练
            saverestore_model.restore_checkpoint(saver=saver)
        for epoch in range(EPOCH):
            for xs, _ in Databatch(dataset=dataset):
                _ = sess.run(opt, feed_dict={x: xs})
            if not (epoch % 100):
                cost_ = sess.run(cost, feed_dict={x: dataset})
                print('第%s轮训练后数据集上的损失值为: %s' % (epoch, cost_))
            # 保存checkpoint节点
            saverestore_model.save_checkpoint(saver=saver, epoch=epoch, is_recording_max_acc=False)
        if is_finishing:
            x_denoising = sess.run(logits, feed_dict={x: dataset})
            axis = x_denoising.shape
            x_denoising = tf.reshape(tensor=x_denoising, shape=(-1, axis[1]*axis[2]*axis[3]))
            SaveFile(data=x_denoising, savepickle_p=r'F:\ODL\dataset\data_denoising.pickle')


def cnn(x, is_training):
    '''
    :param x: 数据特征占位符
    :param is_training: 标志训练和测试时段
    搭建cnn模块
    :return: cnn最后一层flat后的节点
    '''
    #1
    cnn_1 = CNN(x= x, w_conv= (3, 3, 1, 96), stride_conv= 1, stride_pool= 2)
    x_new = cnn_1.reshape(f_vector= x, new_shape= (-1, 15, 15, 1))
    conv1 = cnn_1.convolution(op_outside= x_new)# + tf.Variable(tf.truncated_normal(shape= ([96]), mean= 0, stddev= 1), dtype= tf.float32) #-1*15*15*96
    # conv11 = cnn_1.convolution(op_outside= conv1)
    # conv111 = cnn_1.convolution(op_outside= conv11)
    relu1 = tf.nn.relu(conv1)
    bn1 = cnn_1.batch_normoalization(input= relu1, is_training= is_training)
    pool1 = cnn_1.pooling(pool_fun= tf.nn.max_pool, input= bn1) #-1*8*8*96
    #2
    cnn_2 = CNN(x= pool1, w_conv= (3, 3, 96, 256), stride_conv= 1, stride_pool= 2)
    conv2 = cnn_2.convolution()# + tf.Variable(tf.truncated_normal(shape= ([256]), mean= 0, stddev= 1), dtype= tf.float32) #-1*8*8*256
    # conv22 = cnn_2.convolution(op_outside= conv2)
    relu2 = tf.nn.relu(conv2)
    bn2 = cnn_2.batch_normoalization(input= relu2, is_training= is_training)
    pool2 = cnn_1.pooling(pool_fun=tf.nn.max_pool, input=bn2) #-1*4*4*256
    #3
    cnn_3 = CNN(x=pool2, w_conv=(3, 3, 256, 384), stride_conv=1, stride_pool=2)
    conv3 = cnn_3.convolution()# + tf.Variable(tf.truncated_normal(shape=([384]), mean=0, stddev=1), dtype=tf.float32) #-1*4*4*384
    relu3 = tf.nn.relu(conv3)
    bn3 = cnn_3.batch_normoalization(input= relu3, is_training= is_training)
    pool3 = cnn_1.pooling(pool_fun=tf.nn.max_pool, input=bn3) #-1*2*2*384
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
    rnn = RNN(x= x, max_time= max_time, num_units= num_units)
    outputs, _ = rnn.dynamic_rnn(style= 'LSTM', output_keep_prob= 0.8)
    return outputs[:, -1, :]

def main(dataset_train, dataset_test, train_time):
    '''
    循环网络后的输出层以及训练过程
    :param dataset_train: ndarray, shape= (None, 239), 实际训练数据特征和标签
    :param dataset_test: ndarray, shape= (None, 239), 实际测试数据特征
    :param train_time: 标记当前训练次数
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
        # acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(tf.nn.softmax(fc), 1), tf.argmax(y, 1)), tf.float32))
        evaluation = Evaluation(one_hot= True, logit= tf.nn.softmax(fc), label= y, regression_pred= None, regression_label= None)
        acc = evaluation.acc_classification()
        init = tf.global_variables_initializer()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph= g1) as sess:
        sess.run(init)
        # 建立checkpoint节点保存对象
        saverestore_model = SaveRestore_model(sess=sess, save_file_name='denoiseing', max_to_keep=1)
        saver = saverestore_model.saver_build()
        if train_time != 0:
            # 导入checkpoint节点，继续训练
            saverestore_model.restore_checkpoint(saver=saver)
        x_test = dataset_test[:, :225]
        y_test = onehot(label=dataset_test[:, -1])
        for epoch in range(100000):
            #制作一个数据的生成器可以像mnist的next_batch函数一样
            for xs, ys in Databatch(dataset= dataset_train):
                #将标签转化成one-hot编码
                ys = onehot(label= ys)
                # print(xs.dtype)
                _, loss_ = sess.run([optimizer, loss], feed_dict= {x: xs, y: ys, is_training: True})
            if not (epoch % 100):
                acc_ = sess.run(acc, feed_dict= {x: x_test, y: y_test, is_training: False})
                print('第%s轮训练后测试集上的预测精度为: %s' % (epoch, acc_))
            # 保存checkpoint节点
            saverestore_model.save_checkpoint(saver=saver, epoch=epoch, is_recording_max_acc=False)

        #定义计算pre, recall, F1参数的结点以便传递到下一个计算图
        op_logit, op_label = sess.run([tf.nn.softmax(fc), y], feed_dict= {x: x_test, y: y_test, is_training: False})
        # print(op_label)
    #定义Evaluation类对象
    evalulation_2 = Evaluation(one_hot= True, logit= op_logit, label= op_label, regression_pred= None, regression_label= None)
    PRF_dict_ = evalulation_2.PRF_tables(mode_num= 8)
    # _, PRF_dict_ = evalulation_2.session_PRF(acc= None, prf_dict= PRF_dict)
    print(PRF_dict_)

if __name__ == '__main__':
    rng = np.random.RandomState(0)
    dataset_train = LoadFile(p= r'F:\ODL\dataset\data_train.pickle')
    dataset_test = LoadFile(p= r'F:\ODL\dataset\data_test.pickle')
    dataset_train = (dataset_train - np.min(dataset_train, axis=0)) / (np.max(dataset_train, axis=0) - np.min(dataset_train, axis=0))
    dataset_test = (dataset_test - np.min(dataset_test, axis=0)) / (np.max(dataset_test, axis=0) - np.min(dataset_test, axis=0))
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0, verbose=0, copy=True)
    dataset_train = imp.fit_transform(dataset_train)
    dataset_test = imp.fit_transform(dataset_test)
    rng.shuffle(dataset_train)
    rng.shuffle(dataset_test)
    print(dataset_train.shape, dataset_test.shape)
    #检查数据
    # print(dataset_train.shape, dataset_test.shape)
    # num_train = Counter(dataset_train[:, -1])
    # num_test = Counter(dataset_test[:, -1])
    # print(num_train)
    # print(num_test)
    #数据预处理去噪
    dataset = np.vstack((dataset_train, dataset_test))
    dataset = dataset[:, :225]
    rng.shuffle(dataset)
    denoising(dataset, training_time=1, is_finishing=True)
    #训练数据
    denoise = LoadFile(p=r'F:\ODL\dataset\data_denoising.pickle')
    print(denoise.shape, dataset.shape)
    dataset = np.hstack((denoise, dataset[:, -1][:, np.newaxis]))
    dataset_train, dataset_test = dataset[:4000, :], dataset[4000:, :]
    main(dataset_train= dataset_train, dataset_test= dataset_test, train_time=0)


