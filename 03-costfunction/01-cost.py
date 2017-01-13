#-*- coding:utf-8 -*-
import tensorflow as tf

X = [1.,2.,3.]
Y = [1.,2.,3.]
m = n_samples = len(X)

W = tf.placeholder(tf.float32)

#Construct a linear model
hypothesis = tf.mul(W ,X)

#Cost function
cost = tf.reduce_mean(tf.pow(hypothesis-Y,2))

#Initializing the variables
init = tf.initialize_all_variables()

#For graphs
W_val = []
cost_val = []

#Launch the graphs
sess = tf.Session()
sess.run(init)
for i in xrange(-30,50) :
    print i*0.1, sess.run(cost, feed_dict={W:0.1*i})
    W_val.append(i*0.1)
    cost_val.append(sess.run(cost, feed_dict={W:0.1*i}))

#Graphic display
"""



"""
