# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np

xy = np.loadtxt('train.txt', dtype=np.float, unpack=True)
x_data = np.transpose(xy[:3])
y_data = np.transpose(xy[3:])

X = tf.placeholder('float', [None, 3])  # x1, x2 and 1
Y = tf.placeholder('float', [None, 3])  # A, B, C
W = tf.Variable(tf.random_uniform([3, 3], -0.1, 0.1))

hypothesis = tf.nn.softmax(tf.matmul(X, W))
learning_rate = 0.001

# cross entropy cost function
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), reduction_indices=1))
"""
Y * tf.log(hypothesis) 결과는 행 단위로 더해야 한다.
그림에서 보면, 최종 cost 를 계산하기 전에 행 단위로 결과를 더하고 있다.
이것을 가능하게 하는 옵션이 reduction_indices 매개변수다.
0을 전달하면 열 합계, 1을 전달하면 행 합계, 아무 것도 전달하지 않으면 전체 합계.
이렇게 행 단위로 더한 결과에 대해 전체 합계를 내서 평균을 구하기 위해 reduce_mean 함수가 사용됐다.
"""
train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for step in xrange(2001):
    sess.run(train, feed_dict={X: x_data, Y: y_data})
    if step % 20 == 0:
        print step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W)

print '----------------------------------------'
# one-hot encoding
a = sess.run(hypothesis, feed_dict={X: [[1, 11, 7]]})
print a, sess.run(tf.arg_max(a, 1))
a = sess.run(hypothesis, feed_dict={X: [[1, 3, 4]]})
print a, sess.run(tf.arg_max(a, 1))
