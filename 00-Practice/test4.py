import tensorflow as tf

x1_data = [1, 0, 3, 0, 5]
x2_data = [0, 2, 0, 4, 0]
y_data = [1, 2, 3, 4, 5]

W1 = tf.Variable(tf.random_uniform([1], -0.1, 0.1))
W2 = tf.Variable(tf.random_uniform([1], -0.1, 0.1))
b = tf.Variable(tf.random_uniform([1], -0.1, 0.1))

hypothesis = W1 * x1_data + W2 * x2_data + b

cost = tf.reduce_mean(tf.square(hypothesis - y_data))

a = tf.Variable(0.1)
train = tf.train.GradientDescentOptimizer(a).minimize(cost)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for step in xrange(2001) :
    sess.run(train)
    if step % 20 == 0 :
        print step, sess.run(cost), sess.run(W1), sess.run(W2), sess.run(b)