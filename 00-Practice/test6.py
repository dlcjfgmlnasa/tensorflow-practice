# -*- coding:utf-8 -*-
import tensorflow as tf

x_data = [[1, 1, 1, 1, 1],
          [0., 2., 0., 4., 0.],
          [1., 0., 3., 0., 5.]]
y_data = [1, 2, 3, 4, 5]

W = tf.Variable(tf.random_uniform([1, 3], -0.1, 0.1))

hypothesis = tf.matmul(W, x_data)
cost = tf.reduce_mean(tf.square(hypothesis - y_data))
a = tf.Variable(0.1)
train = tf.train.GradientDescentOptimizer(a).minimize(cost)
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)

    for step in xrange(2001):
        sess.run(train)
        if step % 20 == 0:
            print step, sess.run(W)
