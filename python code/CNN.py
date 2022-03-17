

#loading the dataset

from keras.datasets import cifar10
(train_images, train_labels), (test_images, test_labels)= cifar10.load_data() # here we can dataset assinging to the train abd test

#labels before applying the function
# Training set labels
print(train_labels)
print(train_labels.shape)

# Testing set labels
print(test_labels)
print(test_labels.shape)

# Applying the function to training set labels and testing set labels
from keras.utils.np_utils import to_categorical
train_labels = to_categorical(train_labels, dtype ="uint8")
test_labels = to_categorical(test_labels, dtype ="uint8")

# Labels after applying the function
# Training set labels
print(train_labels)
print(train_labels.shape)

# Testing set labels
print(test_labels)
print(test_labels.shape)
# Initializing Input vector
class_vector=[2, 5, 6, 1, 4, 2, 3, 2]
print(class_vector)

# Applying the function on input class vector
from keras.utils.np_utils import to_categorical
output_matrix = to_categorical(class_vector, num_classes = 7, dtype ="int32")

print(output_matrix)

import tensorflow as tf

tf.test.is_gpu_available(
cuda_only=False, min_cuda_compute_capability=None
)


from tensorflow.keras import layers
from tensorflow.keras import models
model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.summary()

model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.summary()

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, batch_size=128)
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
test_acc


