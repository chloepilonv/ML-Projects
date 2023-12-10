from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#Visualization
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(12, 12))
shuffle_indices = np.random.permutation(9)
for i in range(0, 9):
    plt.subplot(3, 3, i+1)
    image = train_images[shuffle_indices[i]]
    plt.imshow(image/255,cmap='gray')

plt.tight_layout()
plt.show()

#Reshaping it lower res, 28 pixels x 28 pixels
train_images = train_images.reshape((60000,28,28,1))
train_images = train_images.astype('float32')/255 #Because pixel grayscale is from 0 to 255, 0 being black

test_images = test_images.reshape((10000,28,28,1))
test_images = test_images.astype('float32')/255 #lower

from keras.utils import to_categorical

#one-hot encoding
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

#Creating the layers
from keras import layers
from keras import models

model = models.Sequential()

#First layer, with 32 filters, sample size 3x3, activation function that makes the behaviour non-linear
model.add(layers.Conv2D(32,(3,3), activation='relu', input_shape = (28,28,1)))

#Layer #2,'compresses' 2 by 2. Highlights the difference.
model.add(layers.MaxPooling2D(2,2))

#Layer #3, Conv layer
model.add(layers.Conv2D(64, (3,3), activation='relu'))

model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
#Flatten, put the information on a single array
model.add(layers.Flatten())
#Fully connect it, dense it to size 64
model.add(layers.Dense(64,activation = 'relu'))
#Fully connect it, dense it to size 10
model.add(layers.Dense(10, activation= 'softmax'))

model.summary()

from keras.utils import plot_model

#Visualize the model
keras.utils.plot_model(
    model,
    to_file="model.png",
    show_shapes=False,
    show_dtype=False,
    show_layer_names=False,
    rankdir="TB",
    expand_nested=False,
    dpi=200,
    show_layer_activations=False,
    show_trainable=False,
)

#How to reduce the loss function
model.compile(optimizer = 'rmsprop',
             loss = 'categorical_crossentropy',
             metrics = ['accuracy'])

#over 5 iteration, 
model.fit(train_images, train_labels, epochs=5, batch_size = 64)

acc = model.evaluate(test_images, test_labels)[1]
acc = model.evaluate(test_images, test_labels)[2]

print('accuracy is '+ str(acc))
