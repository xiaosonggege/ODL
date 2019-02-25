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
from TestEvaluation import Evaluation
from numpy import random
import numpy as np
from AllNet import RNN
from Routine_operation import Summary_Visualization

#建立摘要对象
summary_visalization = Summary_Visualization()
rng = random.RandomState(0)
a = rng.randint(low= 0, high= 10, size= (10000, 20))
a = a.astype(np.float32)
b = rng.randint(low= 1, high= 4, size= (10000, 1))
b = b.astype(np.float32)
rng.shuffle(a)
rng.shuffle(b)
# print(a.shape, a.dtype)
# print(b.shape, b.dtype)
with tf.name_scope('x-y'):
    x = tf.placeholder(dtype=tf.float32, shape=(None, 20), name= 'x')
    y = tf.placeholder(dtype=tf.float32, shape=(None, 1), name= 'y')

with tf.name_scope('rnn'):
    rnn = RNN(x=x, max_time=4, num_units=128)
    output, fin_state = rnn.dynamic_rnn(style='LSTM', output_keep_prob=0.8)
    fc_in = output[:, -1, :]
with tf.name_scope('para'):
    w_1 = tf.Variable(tf.truncated_normal(shape=(128, 1), mean=0, stddev=1), dtype=tf.float32, name= 'w_1')
    b_1 = tf.Variable(tf.truncated_normal(shape=(1,), mean=0, stddev=1), dtype=tf.float32, name= 'b_1')
    summary_visalization.variable_summaries(var= w_1, name= 'w_1')
    summary_visalization.variable_summaries(var= b_1, name= 'b_1')
with tf.name_scope('fc'):
    fc_1 = tf.matmul(fc_in, w_1) + b_1
    fc_2 = tf.nn.relu(fc_1)  # (batch_size, 1)
with tf.name_scope('opt'):
    loss = tf.reduce_mean(tf.square(fc_2 - y))
    optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss=loss)
    # 回归问题准确率计算
    # acc = tf.reduce_mean(tf.cast(tf.abs(fc_2 - y) < 0.2, tf.float32))
    evaluation = Evaluation(one_hot=False, logit=None, label=None, regression_pred=fc_2, regression_label=y)
    acc = evaluation.acc_regression(Threshold=0.2)

merge = summary_visalization.summary_merge()
init = tf.global_variables_initializer()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(init)
    #摘要文件
    summary_writer = summary_visalization.summary_file(p= 'logs/', graph= sess.graph)
    for i in range(100000):
        summary = sess.run(merge, feed_dict= {x: a, y: b})
        _, loss_s = sess.run([optimizer, loss], feed_dict= {x: a, y: b})
        acc_ = sess.run(acc, feed_dict= {x: a, y: b})
        if not i % 100:
            print(loss_s, acc_)

        summary_visalization.add_summary(summary_writer= summary_writer, summary= summary, summary_information= i)
    summary_visalization.summary_close(summary_writer= summary_writer)

