import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, losses, metrics, optimizers
from keras.layers import Dense, Dropout
from keras.models import Sequential

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

directory = "lab_2/input/mnist_images_csv/"
df = pd.read_csv(directory + "train.csv")

file_paths = df["file_name"].values
labels = df["label"].values
ds_train = tf.data.Dataset.from_tensor_slices((file_paths, labels))


def read_image(image_file, label):
    image = tf.io.read_file(directory + image_file)
    image = tf.image.decode_image(image, channels=1, dtype=tf.float32)
    return image, label


ds_train = ds_train.map(read_image).batch(2)

images = list()
for element in ds_train.as_numpy_iterator():
    images.append(element[0][0])
    images.append(element[0][1])

model = Sequential(
    [
        layers.Input((28, 28, 1)),
        layers.Flatten(),
        layers.Dense(512, activation="relu"),
        layers.Dense(10, activation="softmax")
    ]
)
model.summary()
model.compile(
    loss=[losses.SparseCategoricalCrossentropy(from_logits=True)],
    optimizer=optimizers.Adam(),
    metrics=[metrics.Accuracy().name])
model.fit(ds_train, epochs=10, verbose=2)

score = model.evaluate(ds_train, verbose=0)
print("Test loss: ", score[0])
print("Test accuracy: ", score[1])

to_predict = list(ds_train.as_numpy_iterator())[0][0]
prediction = model.predict(to_predict)

class_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def plot_image(i, predictions_array, labels, images):
    predictions_array, true_label, img = predictions_array[0], labels[i], images[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100*np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(i, predictions, labels):
    predictions_array, true_label = predictions[0], labels[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plot = plt.bar(
        range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    plot[predicted_label].set_color('red')
    plot[true_label].set_color('blue')


predictions = list(map(lambda img: model.predict(img), images))

rows_count = 5
cols_count = 5
images_count = rows_count*cols_count

plt.figure(figsize=(2*2*cols_count, 2*rows_count))
for i in range(images_count):
    plt.subplot(rows_count, 2*cols_count, 2*i+1)
    plot_image(i, predictions[i], labels, images)
    plt.subplot(rows_count, 2*cols_count, 2*i+2)
    plot_value_array(i, predictions[i], labels)
plt.show()
