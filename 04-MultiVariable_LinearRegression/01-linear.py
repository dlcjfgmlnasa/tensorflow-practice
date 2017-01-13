#-*- coding:utf-8 -*-
import tensorflow as tf

x1_data = [1,0,3,0,5]
x2_data = [0,2,0,4,0]
y_data = [1,2,3,4,5]

#Try to find values for W and b that compute y_data = W * x_data + b
#(We know that W should be 1 and 0, but Tensorflow will
#figure that out for us)

W1 = tf.Variable(tf.random_uniform([1],-1.0,1.0))
W2 = tf.Variable(tf.random_uniform([1],-1.0,1.0))
b = tf.Variable(tf.random_uniform([1],-1.0,1.0))

#Our hypothesis
hypothesis = W1 * x1_data + W2 * x2_data + b

#Simplified cost function
cost = tf.reduce_mean(tf.square(hypothesis - y_data))

#Minimize
a = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)


#Before starting, initialize the variables, We will run this first
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in xrange(2001) :
    sess.run(train)
    if step % 20 == 0:
        print step, sess.run(W1), sess.run(W2), sess.run(b)
