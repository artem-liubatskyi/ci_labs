import tensorflow

tf = tensorflow.compat.v1
tf.disable_eager_execution()

x = tf.placeholder('float32', None)
y = x*10 + 1
with tf.Session() as session:
    dataX = [[12, 2, 0, -2], [14, 4, 1, 0]]
    placeX = session.run(y, feed_dict={x: dataX})
    print(placeX)
