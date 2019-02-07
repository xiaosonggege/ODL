#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2018-2021, Node Supply Chain Manager Corporation Limited.
@contact: 1243049371@qq.com
@software:PyCharm
@file: mnist
@time: 2019/2/6 10:11
@desc:
'''
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from AllNet import RNN

data = input_data.read_data_sets('F:\ODL\MNIST_data', one_hot= True)
#一共有多少批次
batch_all = data.train.num_examples // 100
x = tf.placeholder(dtype= tf.float32, shape= (100, 784))
y = tf.placeholder(dtype= tf.float32, shape= (100, 10))
#循环网络层
rnn = RNN(x, y, loss= 0, optimizer= 0, max_time= 28, num_units= 128)
outputs, fin_state = rnn.dynamic_rnn(style= 'LSTM', output_keep_prob= 0.8)
rnn_output = outputs[:, -1, :]
w_1 = tf.Variable(tf.truncated_normal(shape= (128, 10), mean= 0, stddev= 1), dtype= tf.float32)
w_2 = tf.Variable(tf.truncated_normal(shape= (64, 10), mean= 0, stddev= 1), dtype= tf.float32)
b_1 = tf.Variable(tf.truncated_normal(shape= (10, ), mean= 0, stddev= 1), dtype= tf.float32)
b_2 = tf.Variable(tf.truncated_normal(shape= (10, ), mean= 0, stddev= 1), dtype= tf.float32)
fc_1 = tf.matmul(rnn_output, w_1) + b_1
fc_1 = tf.nn.relu(fc_1)
fc_2 = tf.nn.dropout(fc_1, keep_prob= 0.8)
# fc_2 = tf.matmul(fc_1, w_2) + b_2
# fc_2 = tf.nn.relu(fc_2)
# fc_2 = tf.nn.dropout(fc_2, keep_prob= 0.8)
loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels= y, logits= fc_2))
optimizer = tf.train.GradientDescentOptimizer(learning_rate= 1e-4).minimize(loss= loss)
acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(fc_2, 1)), tf.float32))
init = tf.global_variables_initializer()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(init)
    for i in range(60):
        for batch_num in range(batch_all):
            xs, ys = data.train.next_batch(100)
            _, loss_s = sess.run([optimizer, loss], feed_dict={x: xs, y: ys})
            if not batch_num % 1000:
                # print(loss_s)
                acc_c = sess.run(acc, feed_dict={x: xs, y: ys})
                print(acc_c)




