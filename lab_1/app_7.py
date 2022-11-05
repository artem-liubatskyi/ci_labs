import tensorflow
import os

tf = tensorflow.compat.v1
tf.disable_eager_execution()

session = tf.Session()

A = tf.zeros([2, 3])
print(session.run(A))

B = tf.ones([4, 3])
print(session.run(B))

C = tf.fill([2, 3], 13)
print(session.run(C))

D = tf.diag([4, -3, 2])
print(session.run(D))

E = tf.constant([5, 2, 4, 2])
print(session.run(E))

G = tf.range(start=6, limit=45, delta=3)
print(session.run(G))

H = tf.linspace(10.0, 92.0, 5)
print(session.run(H))

R1 = tf.random_uniform([2, 3], minval=0, maxval=4)
print(session.run(R1))

R2 = tf.random_normal([2, 3], mean=0, stddev=4)
print(session.run(R2))

I = tf.nn.tanh([10, 2, 1, 0, 0.5, 0, -0.5, -1., -2., -10.])
print(session.run(I))

J = tf.nn.sigmoid([10, 2, 1, 0.5, 0, -0.5, -1, -2, -10])
print(session.run(J))

K = tf.nn.relu([-1, 1, -3, 13])
print(session.run(K))
