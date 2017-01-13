#-*- coding:utf-8 -*-
import tensorflow as tf
#Everything is operation

sess = tf.Session()

a = tf.constant(2)
b = tf.constant(3)

c = a+b #노드이 operation이다 노드의 형태만 가지고 있다.

print c #의미없는 값이 나온다.
print sess.run(c)
