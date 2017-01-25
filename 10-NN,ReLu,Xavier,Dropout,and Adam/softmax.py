# -*- coding:utf-8 -*-
import datetime as dt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# Get timestamp
timestamp = dt.datetime.now().strftime('%Y%m%d%H%M%S')

# Import MINST data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parameters
learning_rate = 0.1
training_epochs = 8000
batch_size = 100
display_step = 10

X = tf.placeholder(tf.float32, [None, 784], name='X_input')
Y = tf.placeholder(tf.float32, [None, 10], name='Y-input')

W = tf.Variable(tf.random_uniform([784, 10]), name='X-input')
b = tf.Variable(tf.random_uniform([10]), name='Y-input')

tf.histogram_summary('Weight', W)
tf.histogram_summary('bias', b)

with tf.name_scope('Layer') as sess:
    hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

with tf.name_scope('Cost') as sess:
    cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), reduction_indices=1))
    tf.scalar_summary('Cost', cost)

with tf.name_scope('Train') as sess:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

with tf.name_scope('Accuracy') as sess:
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1)), tf.float32))
    tf.scalar_summary('Accuracy', accuracy)

with tf.Session() as sess:
    writer = tf.train.SummaryWriter('./minist/softmax', sess.graph)
    merged = tf.merge_all_summaries()

    tf.initialize_all_variables().run()

    for step in xrange(training_epochs):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)

        _, summary = sess.run([optimizer, merged], feed_dict={X: batch_xs, Y: batch_ys})
        writer.add_summary(summary, step)

    writer.close()
