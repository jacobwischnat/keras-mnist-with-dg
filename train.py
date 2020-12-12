from os import stat

import gzip
import math
import keras
import numpy as np
import urllib.request
import matplotlib.pyplot as plot
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

batch_size=64
image_size = 28
label_count = 10
validation_percentage = 20
input_shape = (28, 28, 1)
image_byte_size = image_size * image_size

train_data = urllib.request.urlretrieve('/train-images.dat', "train-images.dat")

data_train = open("train-images.dat", mode="rb")
data_train.read(16) # read the header?

train_size = stat(data_train.fileno()).st_size - 16
image_count = math.floor(train_size / (28 * 28))
valid_image_count = math.floor((validation_percentage / 100) * image_count)
train_indexes = range(valid_image_count, image_count)
valid_indexes = range(0, valid_image_count)

print("Image count", image_count)

valid_data = urllib.request.urlretrieve('/train-labels.gz', "train-labels.gz")

label_train = gzip.open("train-labels.gz")
label_train.read(8)
label_size = stat(label_train.fileno()).st_size - 8
print("Label size", label_size)

def get_input(index):
    byte_index = index * image_byte_size
    data_train.seek(byte_index)
    data = np.frombuffer(data_train.read(image_byte_size), dtype=np.uint8).astype(np.float32)
    return data.reshape(image_size, image_size, 1)

def get_output(index):
    label_train.seek(index)
    return np.asarray([int.from_bytes(label_train.read(1), "big")])

def image_generator(indexes, batch_size=64):
    while True:
        batch_values = np.random.choice(indexes, size=batch_size)

        batch_input = []
        batch_output = []

        for input_index in batch_values:
            input = get_input(input_index)
            output = get_output(input_index)

            batch_input += [ input ]
            batch_output += [ output ]

        batch_x = np.array(batch_input)
        batch_y = np.array(batch_output)

        yield(batch_x, batch_y)

model = keras.models.Sequential()
model.add(keras.Input(shape=input_shape))
model.add(keras.layers.convolutional.Conv2D(
    32,
    strides=(1, 1),
    kernel_size=(5, 5),
    activation="relu"
))
model.add(keras.layers.pooling.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(keras.layers.convolutional.Conv2D(
    64,
    kernel_size=(5, 5),
    activation="relu"
))
model.add(keras.layers.pooling.MaxPool2D(pool_size=(2, 2)))
model.add(keras.layers.core.Flatten())
model.add(keras.layers.core.Dense(1000, activation="relu"))
model.add(keras.layers.core.Dense(label_count, activation="softmax"))

opt = keras.optimizers.SGD(lr=0.01)
model.compile(
    loss="categorical_crossentropy",
    optimizer=opt,
    metrics=['accuracy']
)

model.fit(
    image_generator(train_indexes),
    validation_data=image_generator(valid_indexes)
)