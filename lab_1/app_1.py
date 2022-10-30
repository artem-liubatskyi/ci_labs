import tensorflow

tf = tensorflow.compat.v1
tf.disable_eager_execution()

x = tf.constant(12, dtype='float32')
session = tf.Session()
print(session.run(x))
