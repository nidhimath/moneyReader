import time
import numpy
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras.utils import np_utils
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import pickle

image_size = 256

# Loading all of the images into CIFAR-10 format
def load_cfar10_batch(fileName):
    with open(fileName, mode='rb') as file:
        # note the encoding type is 'latin1'
        batch = pickle.load(file, encoding='latin1')

    input_features = batch['data']
    print(input_features)

    numImages = (int(len(input_features)/(3*image_size*image_size)))
    print(numImages)

    features = input_features.reshape(numImages, 3, image_size, image_size).transpose(0, 2, 3, 1)

    labels = batch['labels']
    return features, labels



# Loading in the data
(X_train, y_train) = load_cfar10_batch('../cifar10-converter_WithDataAugmentation/cifar10-ready.bin')

# Need to give test data as well
(X_test, y_test) = load_cfar10_batch('../cifar10-converter_WithDataAugmentation/cifar10-ready-testData.bin')


plt.imshow(X_train[1])
print("Class : ",y_train[1])




# Normalizing the inputs from 0-255 to between 0 and 1 by dividing by 255
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0


# Designing the model
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
class_num = y_test.shape[1]


# Creating the Model
model = Sequential()


# Adding the first layer
model.add(Conv2D(32, (3, 3),
                 input_shape=X_train.shape[1:],
                 padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(BatchNormalization())


# Second layer
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())


# 3rd layer
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(BatchNormalization())



# 4th layer
# model.add(Conv2D(256, (3, 3), padding='same'))
# model.add(Activation('relu'))
# model.add(Dropout(0.2))
# model.add(BatchNormalization())


# Flattening the data
model.add(Flatten())


# Max Norm
# model.add(Dense(512, kernel_constraint=maxnorm(3)))
# model.add(Activation('relu'))
# model.add(Dropout(0.2))
# model.add(BatchNormalization())


model.add(Dense(256, kernel_constraint=maxnorm(3)))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(BatchNormalization())


model.add(Dense(128, kernel_constraint=maxnorm(3)))
model.add(Activation('relu'))
model.add(BatchNormalization())



# Soft-Norm
model.add(Dense(class_num))
model.add(Activation('softmax'))


epochs = 20
optimizer = 'adam'
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
print(model.summary())


# Starting to train the model
seed = 21
numpy.random.seed(seed)
model.fit(X_train, y_train,
          validation_data=(X_test, y_test),
          epochs=epochs,
          batch_size=image_size)

scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

model.save('blind.h5')


