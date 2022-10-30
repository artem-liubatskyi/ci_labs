import tensorflow

tf = tensorflow.compat.v1
tf.disable_eager_execution()

x = tf.placeholder('float32', None)
y = x*10 + 500
with tf.Session() as session:
    placeX = session.run(y, feed_dict={x: [0, 5, 15, 25]})
    print(placeX)
