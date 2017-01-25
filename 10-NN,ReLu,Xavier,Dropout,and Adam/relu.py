# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np

xy = np.loadtxt('train.txt', unpack=True)
x_data = np.transpose(xy[:-1])
y_data = np.reshape(xy[-1], (4, 1))

X = tf.placeholder(tf.float32, name='X-Input')
Y = tf.placeholder(tf.float32, name='Y-Input')

W1 = tf.Variable(tf.random_uniform([2, 5], -1.0, 1.0), name='Weight1')
W2 = tf.Variable(tf.random_uniform([5, 4], -1.0, 1.0), name='Weight2')
W3 = tf.Variable(tf.random_uniform([4, 3], -1.0, 1.0), name='Weight3')
W4 = tf.Variable(tf.random_uniform([3, 4], -1.0, 1.0), name='Weight4')
W5 = tf.Variable(tf.random_uniform([4, 1], -1.0, 1.0), name='Weight5')

b1 = tf.Variable(tf.random_uniform([5], -1.0, 1.0), name='bias1')
b2 = tf.Variable(tf.random_uniform([4], -1.0, 1.0), name='bias2')
b3 = tf.Variable(tf.random_uniform([3], -1.0, 1.0), name='bias3')
b4 = tf.Variable(tf.random_uniform([4], -1.0, 1.0), name='bias4')
b5 = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name='bias5')

with tf.name_scope('layer2') as scope:
    L2 = tf.nn.relu(tf.matmul(X, W1) + b1)

with tf.name_scope('layer3') as scope:
    L3 = tf.nn.relu(tf.matmul(L2, W2) + b2)

with tf.name_scope('layer4') as scope:
    L4 = tf.nn.relu(tf.matmul(L3, W3) + b3)

with tf.name_scope('layer5') as scope:
    L5 = tf.nn.relu(tf.matmul(L4, W4) + b4)

with tf.name_scope('layer6') as scope:
    hypothesis = tf.sigmoid(tf.matmul(L5, W5) + b5)

with tf.name_scope('cost') as scope:
    cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
    tf.scalar_summary('cost', cost)

with tf.name_scope('train') as scope:
    optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

with tf.name_scope('accuracy') as scope:
    answer = tf.equal(tf.floor(hypothesis + 0.5), Y)
    accuracy = tf.reduce_mean(tf.cast(answer, 'float'))
    tf.scalar_summary('accuracy', accuracy)

w1_hist = tf.histogram_summary('Weight1', W1)
w2_hist = tf.histogram_summary('Weight2', W2)
w3_hist = tf.histogram_summary('Weight3', W3)
w4_hist = tf.histogram_summary('Weight4', W4)
w5_hist = tf.histogram_summary('Weight5', W5)

b1_hist = tf.histogram_summary('bias1', b1)
b2_hist = tf.histogram_summary('bias2', b2)
b3_hist = tf.histogram_summary('bias3', b3)
b4_hist = tf.histogram_summary('bias4', b4)
b5_hist = tf.histogram_summary('bias5', b5)
y_hist = tf.histogram_summary('y', Y)


with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess.run(init)

    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter('./log/xor_log', sess.graph_def)

    for step in xrange(2001):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        if step % 100 == 0:
            summary = sess.run(merged, feed_dict={X: x_data, Y: y_data})
            writer.add_summary(summary, step)




