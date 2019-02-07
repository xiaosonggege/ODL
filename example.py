#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2018-2021, Node Supply Chain Manager Corporation Limited.
@contact: 1243049371@qq.com
@software:PyCharm
@file: example
@time: 2019/2/7 17:02
@desc:
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 载入数据
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 输入图片是28
n_input = 28
max_time = 28
lstm_size = 100  # 隐藏单元
n_class = 10  # 10个分类
batch_size = 50  # 每次50个样本
n_batch_size = mnist.train.num_examples // batch_size  # 计算一共有多少批次

# 这里None表示第一个维度可以是任意长度
# 创建占位符
x = tf.placeholder(tf.float32, [None, 28 * 28])
# 正确的标签
y = tf.placeholder(tf.float32, [None, 10])

# 初始化权重 ,stddev为标准差
weight = tf.Variable(tf.truncated_normal([lstm_size, n_class], stddev=0.1))
# 初始化偏置层
biases = tf.Variable(tf.constant(0.1, shape=[n_class]))


# 定义RNN网络
def RNN(X, weights, biases):
    #  原始数据为[batch_size,28*28]
    # input = [batch_size, max_time, n_input]
    input = tf.reshape(X, [-1, max_time, n_input])
    # 定义LSTM的基本单元
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    # final_state[0] 是cell state
    # final_state[1] 是hidden stat
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, input, dtype=tf.float32)
    results = tf.nn.softmax(tf.matmul(final_state[1], weights) + biases)
    return results


# 计算RNN的返回结果
prediction = RNN(x, weight, biases)
# 损失函数
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
# 使用AdamOptimizer进行优化
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
# 将结果存下来
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
# 计算正确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# 初始化
init = tf.global_variables_initializer()
gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(init)
    for epoch in range(6):
        for batch in range(n_batch_size):
            # 取出下一批次数据
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
            if (batch % 100 == 0):
                print(str(batch) + "/" + str(n_batch_size))
            acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys})
            print("Iter" + str(epoch) + " ,Testing Accuracy = " + str(acc))
