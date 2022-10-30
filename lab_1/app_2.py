import tensorflow

tf = tensorflow.compat.v1
tf.disable_eager_execution()

x = tf.constant(12, dtype='float32')
y = tf.Variable(x+11)
model = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(model)
    print(session.run(y))
