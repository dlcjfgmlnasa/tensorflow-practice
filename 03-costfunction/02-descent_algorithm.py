#-*- coding:utf-8 -*-
"""
직접구현한 descent 알고리즘
"""
import tensorflow as tf

x_data = [1.,2.,3.]
y_data = [1.,2.,3.]

W = tf.Variable(tf.random_uniform([1],-10.0,10.0))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

#Our hypothesis
hypothesis = W * X

cost = tf.reduce_mean(tf.square(hypothesis - Y))

#Minimize
descent = W - tf.mul(0.1, tf.reduce_mean(tf.mul( tf.mul(W,X) - Y  , X)))
update = W.assign(descent) #W의 값을 업데이트를 시킨다.
                           #assign : 할당하다.
                           #아직 실행을 하지 않았으므로 node이다.

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

#python의 공통된 변수를 쓰니까 하나를 실행하면 다른 하나도 실행됨
for step in xrange(100) :
    sess.run(update,feed_dict={X:x_data,Y:y_data})
    print step, sess.run(cost,feed_dict={X:x_data,Y:y_data}) , sess.run(W)
