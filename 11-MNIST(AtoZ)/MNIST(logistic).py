import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./log/MNIST_data', one_hot=True)

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)
cost = tf.reduce_mean(tf.reduce_sum(-Y * tf.log(hypothesis), reduction_indices=1))
optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(cost)
init = tf.initialize_all_variables()

training_epoch = 25
display_step = 1
batch_size = 100

with tf.Session() as sess:
    sess.run(init)

    for epoch in xrange(training_epoch):

        total_batch = mnist.train.num_examples / 100
        for i in xrange(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)