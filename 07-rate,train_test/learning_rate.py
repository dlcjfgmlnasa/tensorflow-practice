# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np

xy = np.loadtxt('train.txt', dtype=np.float, unpack=True)
x_Data = np.transpose(xy[:3])
y_Data = np.transpose(xy[3:])

X = tf.placeholder('float', [None, 3])
Y = tf.placeholder('float', [None, 3])
W = tf.Variable(tf.random_uniform([3, 3], -0.1, 0.1))
hypothesis = tf.nn.softmax(tf.matmul(X, W))

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), reduction_indices=1))
"""
10 : learning_rate 가 너무 커지면 오버 슈팅이 된다.
0.0001 : 너무 값이 떨어지지 않는다.
"""
learning_rate = 0.01
train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for step in xrange(2001):
    sess.run(train, feed_dict={X: x_Data, Y: y_Data})
    if step % 20 == 0:
        print step, sess.run(cost, feed_dict={X: x_Data, Y: y_Data}), sess.run(W)