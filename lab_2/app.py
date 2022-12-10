from __future__ import print_function

import keras
import numpy as np
from keras import losses, metrics, optimizers
from keras.datasets import fashion_mnist
from keras.layers import Dense, Dropout
from keras.models import Sequential

batch_size = 128
num_classes = 10
epochs = 2

(train_images, train_labels), (test_images,
                               test_labels) = fashion_mnist.load_data()

train_images = train_images.reshape(60000, 784)
test_images = test_images.reshape(10000, 784)
train_images = train_images.astype("float32")
test_images = test_images.astype("float32")

train_images = (train_images-np.mean(train_images))/np.std(train_images)
test_images = (test_images-np.mean(test_images))/np.std(test_images)

train_labels = keras.utils.to_categorical(train_labels, num_classes)
test_labels = keras.utils.to_categorical(test_labels, num_classes)

model = Sequential()
model.add(Dense(256, activation="relu", input_shape=(784,)))
model.add(Dropout(0.4))
model.add(Dense(512, activation="relu"))
model.add(Dense(num_classes, activation="softmax"))

model.summary()

model.compile(
    loss="categorical_crossentropy",
    optimizer=optimizers.Adam(),
    metrics=[metrics.BinaryAccuracy()],
)

model.fit(train_images, train_labels,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(test_images, test_labels))

model.save("my_model.h5")

score = model.evaluate(test_images, test_labels, verbose=0)
print("Test loss: ", score[0])
print("Test accuracy: ", score[1])
