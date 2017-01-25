import tensorflow as tf
import numpy as np

x_data = [1,2,3]
y_data = [1,2,3]

W = tf.Variable(tf.random_uniform([1],-1.0,1.0))
b = tf.Variable(tf.random_uniform([1],-1.0,1.0))

hypothesis = x_data * W + b

cost = tf.reduce_mean(tf.square(hypothesis - y_data))
a = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

with tf.Session() as sess :
    init = tf.initialize_all_variables()
    sess.run(init)

    for step in xrange(2001):
        sess.run(train)
        if step % 20 == 0:
            print step, sess.run(cost), sess.run(W), sess.run(b)