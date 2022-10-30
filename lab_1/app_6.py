import tensorflow
import os

tf = tensorflow.compat.v1
tf.disable_eager_execution()

dir_path = os.path.dirname(os.path.realpath(__file__))
filename = dir_path + '/assets/image.jpg'
image = tf.image.decode_jpeg(tf.read_file(filename), channels=3)

session = tf.InteractiveSession()
print(session.run(image[10:15,0:4,1]))
