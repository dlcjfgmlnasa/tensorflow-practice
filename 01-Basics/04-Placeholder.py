#-*- coding:utf-8 -*-
import tensorflow as tf

#placeholder란 프로그래밍의 매개변수처럼 값을넣어줄것이라고 지정해두고
#나중에 값을 넣는 방식
a = tf.placeholder(tf.int16) #데이터 타입만 정해둔다.
b = tf.placeholder(tf.int16)

add = tf.add(a,b)
mul = tf.mul(a,b)

#실행하는 시점에서 값들을 바꿀수 있는 파워풀한 기능이다. 마치 함수처럼
with tf.Session() as sess :
    #Run every operation with variable input
    print "Addition with variables : %i" % sess.run(add, feed_dict={a:2,b:3})
    print "Multiplication with variables : %i" % sess.run(mul, feed_dict={a:2,b:3})
