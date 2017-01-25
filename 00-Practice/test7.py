# -*-coding:utf-8 -*-
import tensorflow as tf
import numpy as np
# unpack 어떤 행태로 pack을 잡을 것인가
xy = np.loadtxt('train.txt', dtype=np.float32, unpack=True)
x_data = xy[:-1]
y_data = xy[-1]

print 'x', x_data
print 'y', y_data

W = tf.Variable(tf.random_uniform([1, len(x_data)], -0.1, 0.1))

hypothesis = tf.matmul(W, x_data)
cost = tf.reduce_mean(tf.square(hypothesis - y_data))

a = tf.Variable(0.1)
train = tf.train.GradientDescentOptimizer(a).minimize(cost)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for step in xrange(2001):
    sess.run(train)
    if step % 20 == 0:
        print step, sess.run(cost), sess.run(W)