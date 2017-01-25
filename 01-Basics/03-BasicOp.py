#-*- coding:utf-8 -*-

import tensorflow as tf

a = tf.constant(2)
b = tf.constant(3)


with tf.Session() as sess :
    print "a=2,b=3"
    print "Addition with constants : %i" % sess.run(a+b)
    print "Multiplication with constants : %i" % sess.run(a*b)

