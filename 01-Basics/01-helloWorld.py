#-*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np


hello = tf.constant('Hello, TensorFlow!')

print hello # Tensor("Const:0", shape=(), dtype=string)
            # hello라는 것 자체가 하나의 Tensor이다.

sess = tf.Session()
print sess.run(hello) #run이라는 것을 실행을 할때 의미를 가진다.
