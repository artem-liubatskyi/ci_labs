import tensorflow
import os

tf = tensorflow.compat.v1
tf.disable_eager_execution()

session = tf.Session()
A = tf.zero([2, 3])
print(session.run(A))
