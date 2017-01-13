#-*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np

xy = np.loadtxt('train.txt',unpack=True,dtype='float32')
x_data = xy[0:-1]
y_data = xy[-1]

print 'x',x_data
print 'y',y_data

#처음 Weight값은 -1과 1사이의 랜덤한 값을 주어도 되고
#-5와 5사이의 값을 주어도된다. 큰 의미는 없다.
W = tf.Variable(tf.random_uniform([1,len(x_data)],-0.1,0.1))

hypothsis = tf.matmul(W,x_data)

cost = tf.reduce_mean(tf.square(hypothsis - y_data))

#Minimize
a = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in xrange(2001) :
    sess.run(train)
    if step % 20 == 0 :
        print step, sess.run(cost), sess.run(W)
