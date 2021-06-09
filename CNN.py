# -*- coding: utf-8 -*-
"""Assignment.ipynb
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

def load_dataset():

    # Loading the data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    # Printing the shape of the dataset
    print("Before reshaping the data: ")
    print("X_train shape: ", x_train.shape)
    print("y_train shape: ", y_train.shape)
    print("X_test shape: ", x_test.shape)
    print("y_test shape: ", y_test.shape)

    # Visualizing some of the images in the dataset
    print("\n\nVisualising the data: ")
    plt.subplot(131)
    plt.imshow(x_train[0])

    plt.subplot(132)
    plt.imshow(x_train[200])

    plt.subplot(133)
    plt.imshow(x_train[599])

    plt.show()

    # Reshaping the data
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    # Printing the shape of the dataset
    print("\n\nAfter reshaping the data: ")
    print("X_train shape: ", x_train.shape)
    print("y_train shape: ", y_train.shape)
    print("X_test shape: ", x_test.shape)
    print("y_test shape: ", y_test.shape)

    # Standardizing the data
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train = x_train /255
    x_test = x_test/255

    return (x_train, y_train), (x_test, y_test)

(x_train, y_train), (x_test, y_test) = load_dataset()

"""### Model 1

The first architecture we will try is pretty straight forward. We have
 - 2 Convolution layers
 - Max Pool layer
 - Dropout layer
 - Dense layer
"""

# Model 1
# We'll include all the available layers to us in the model 1 architecture.

model1 = tf.keras.Sequential()

model1.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(28, 28, 1)))
model1.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
model1.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
model1.add(tf.keras.layers.Dropout(0.5))
model1.add(tf.keras.layers.Flatten())
model1.add(tf.keras.layers.Dense(128, activation='relu'))
model1.add(tf.keras.layers.Dropout(0.5))
model1.add(tf.keras.layers.Dense(10, activation='softmax'))

model1.summary()

model1.compile(loss="categorical_crossentropy", metrics=['accuracy'], optimizer='Adam')

start_time_m1 = time.time()
m1 = model1.fit(x_train, y_train, batch_size = 128, epochs=12, verbose=1)
end_time_m1 = time.time()

print("Total time take is: ", end_time_m1 - start_time_m1, " seconds")

# Making predictions

score1 = model1.evaluate(x_test, y_test, verbose=0)
print("Testing accuracy is: ", score1[1])

# Saving the model
model1.save("model1.h5")

"""### Model 2

In this architecture, I have tried a full Convolution layer approach. This model consists of:
 - 4 Convolution Layers
 - Dense Layers
 - **No Max Pool layers**
 - **No Dropout layers**

"""

# Model 2

model2 = tf.keras.Sequential()

model2.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(28, 28, 1)))
model2.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
model2.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
model2.add(tf.keras.layers.Flatten())
model2.add(tf.keras.layers.Dense(128, activation='relu'))
model2.add(tf.keras.layers.Dense(10, activation='softmax'))

model2.summary()

model2.compile(loss="categorical_crossentropy", metrics=['accuracy'], optimizer='Adam')

start_time_m2 = time.time()
m2 = model2.fit(x_train, y_train, batch_size = 128, epochs=12, verbose=1)
end_time_m2 = time.time()

print("Total time take is: ", end_time_m2 - start_time_m2, " seconds")

# Making predictions

score2 = model2.evaluate(x_test, y_test, verbose=0)
print("Testing accuracy is: ", score2[1])

# Saving the model
model2.save("model2.h5")

"""### Model 3

In this architecture, I have used Pooling layers. This model consists of:
 - 2 Convolution Layers
 - Dense Layers
 - Max Pool layers
 - **No Dropout layers**

"""

# Model 2

model3 = tf.keras.Sequential()

model3.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(28, 28, 1)))
model3.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
model3.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
model3.add(tf.keras.layers.Flatten())
model3.add(tf.keras.layers.Dense(128, activation='relu'))
model3.add(tf.keras.layers.Dense(10, activation='softmax'))

model3.summary()

model3.compile(loss="categorical_crossentropy", metrics=['accuracy'], optimizer='Adam')

start_time_m3 = time.time()
m3 = model3.fit(x_train, y_train, batch_size = 128, epochs=12, verbose=1)
end_time_m3 = time.time()

print("Total time take is: ", end_time_m3 - start_time_m3, " seconds")

# Making predictions

score3 = model3.evaluate(x_test, y_test, verbose=0)
print("Testing accuracy is: ", score3[1])

# Saving the model
model3.save("model3.h5")