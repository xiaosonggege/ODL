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
from numpy import random
import numpy as np
from AllNet import RNN

rng = random.RandomState(0)
a = rng.randint(low= 0, high= 10, size= (10000, 20))
a = a.astype(np.float32)
b = rng.randint(low= 1, high= 4, size= (10000, 1))
b = b.astype(np.float32)
rng.shuffle(a)
rng.shuffle(b)
# print(a.shape, a.dtype)
# print(b.shape, b.dtype)
x = tf.placeholder(dtype= tf.float32, shape= (None, 20))
y = tf.placeholder(dtype= tf.float32, shape= (None, 1))
rnn = RNN(x= x, y= y, loss= 0, optimizer= 0, max_time= 4, num_units= 128)
output, fin_state = rnn.dynamic_rnn(style= 'LSTM', output_keep_prob= 0.8)
fc_in = output[:, -1, :]
w_1 = tf.Variable(tf.truncated_normal(shape= (128, 1), mean= 0, stddev= 1), dtype= tf.float32)
b_1 = tf.Variable(tf.truncated_normal(shape= (1, ), mean= 0, stddev= 1), dtype= tf.float32)
fc_1 = tf.matmul(fc_in, w_1) + b_1
fc_2 = tf.nn.relu(fc_1) #(batch_size, 1)
loss = tf.reduce_mean(tf.square(fc_2 - y))
optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss= loss)
#回归问题准确率计算
acc = tf.reduce_mean(tf.cast(tf.abs(fc_2 - y) < 0.2, tf.float32))
init = tf.global_variables_initializer()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(init)
    for i in range(100000):
        _, loss_s = sess.run([optimizer, loss], feed_dict= {x: a, y: b})
        acc_ = sess.run(acc, feed_dict= {x: a, y: b})
        if not i % 100:
            print(loss_s, acc_)

