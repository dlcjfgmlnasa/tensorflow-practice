#-*- coding:utf-8 -*-
#placeholder를 이용하면 모델의 제사용이 가능하다.
import tensorflow as tf

x_data = [1,2,3]
y_data = [1,2,3]

W = tf.Variable(tf.random_uniform([1],-1.0,1.0))
b = tf.Variable(tf.random_uniform([1],-1.0,1.0))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypothesis = W * X + Y

cost = tf.reduce_mean(tf.square(hypothesis - Y))

a = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)


for step in xrange(2001) :
    sess.run(train, feed_dict={X:x_data,Y:y_data})
    if step % 20 == 0 :
        print step, sess.run(cost,feed_dict={X:x_data,Y:y_data}), \
                sess.run(W), sess.run(b)

