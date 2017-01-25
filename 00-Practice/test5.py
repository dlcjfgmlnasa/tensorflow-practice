# -*- coding:utf-8 -*-
import tensorflow as tf

x_data = [[1., 0., 3., 0., 5.],
           [0., 2., 0., 4., 0.]]
y_data = [1, 2, 3, 4, 5]

# 1곱하기 2의 배열로 선언이 되었다.
W = tf.Variable(tf.random_uniform([1, 2], -0.1, 0.1))
b = tf.Variable(tf.random_uniform([1], -0.1, 0.1))

# hypothesis = x_data * W + b
# ValueError: Incompatible shapes for broadcasting: (2, 5) and (1, 2)
# 이렇게 되는 이유는 메트릭스끼리 곱하는 거기 때문이다.
# 메트릭스 끼리 곱하기 위해서는 matmul을 사용해야한다.

# hypothesis = tf.matmul(x_data, W) 이거는 올바는 행렬의 개산이 아니다.

hypothesis = tf.matmul(W, x_data) + b
cost = tf.reduce_mean(tf.square(hypothesis - y_data))

a = tf.Variable(0.1)
train = tf.train.GradientDescentOptimizer(a).minimize(cost)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for step in xrange(2001) :
    sess.run(train)
    if step % 20 == 0 :
        print step, sess.run(cost), sess.run(W), sess.run(b)
