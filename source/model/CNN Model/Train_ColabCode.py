from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
import glob
import tempfile
import tensorflow as tf




# get the location of downloaded zip file and then join it with the name of extracted folder name
zip_dir = "/storage/data/datasets/us-bills-data.zip"
data_dir = "../../../Downloads/Blind Project/Old Data/data/training/*/*.jpg"
data_dir_withoutspaces="../../../Downloads/Blind Project/Old Data/data/training"
print(data_dir)




image_count = len(list(glob.glob(data_dir)))
print("Number of images = " + str(image_count))

# get the names of classes as a list
image_classes = np.array([item for item in glob.glob(data_dir)])
print("Image Classes = " + str(image_classes))





IMAGE_SIZE = (224, 224) # height, width
BATCH_SIZE = 16
datagen_args = dict(rescale=1./255,
                    validation_split=0.20)
dataflow_args = dict(target_size=IMAGE_SIZE,
                     batch_size=BATCH_SIZE,
                     interpolation='bilinear')

# Validation data generator and flow
valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_args)

valid_generator = valid_datagen.flow_from_directory(
    data_dir_withoutspaces,
    subset="validation",
    shuffle=False,
    **dataflow_args)

# Train data generator, with augmentation and train flow
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
      rotation_range=45,
      horizontal_flip=True,
      width_shift_range=0.2, height_shift_range=0.2,
      shear_range=0.2, zoom_range=0.2,
      **datagen_args)
train_generator = train_datagen.flow_from_directory(
    data_dir_withoutspaces,
    subset="training",
    shuffle=True,
    **dataflow_args)


# First let's get the indices assigned to each class labels for reference
train_generator.class_indices
visualize_train_images, visualize_train_labels  = next(train_generator)


# Inspect the shape of the batch of data
print("Shape of 1 image batch = " + str(visualize_train_images.shape))
print("Shape of 1 image batch = " + str(visualize_train_labels.shape))


# This function will plot images in the form of a grid with 1 row and 6 columns
def plotImages(images_arr,images_label):
    fig, axes = plt.subplots(1, 6, figsize=(20,20))
    axes = axes.flatten()
    for img, lbl, ax in zip(images_arr, images_label, axes):
        ax.imshow(img)
        ax.set_title(lbl)
    plt.tight_layout()
    plt.show()


plotImages(visualize_train_images[:6], visualize_train_labels[:6])






tf.keras.backend.clear_session()

simple_cnn_input = tf.keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), name='input_img')
x = tf.keras.layers.Conv2D(16, 3, activation='relu')(simple_cnn_input)
x = tf.keras.layers.MaxPooling2D(2)(x)
# x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.Conv2D(32, 3, activation='relu')(x)
x = tf.keras.layers.MaxPooling2D(2)(x)
# x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.Conv2D(64, 3, activation='relu')(x)
x = tf.keras.layers.MaxPooling2D(2)(x)
# x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.Conv2D(64, 3, activation='relu')(x)
x = tf.keras.layers.MaxPooling2D(2)(x)
# x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(256, activation='relu')(x)
simple_cnn_output = tf.keras.layers.Dense(7, activation='softmax')(x)

cnn_model = tf.keras.Model(simple_cnn_input, simple_cnn_output, name='simple_cnn_model')

cnn_model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
cnn_model.summary()








CNN_MODEL_EPOCHS = 5
CALLBACK_EARLY_STOP = tf.keras.callbacks.EarlyStopping(patience=10,
                                                       min_delta=1e-2,
                                                       verbose=1)
CALLBACK_MODEL_CHKPOINT = tf.keras.callbacks.ModelCheckpoint(filepath='/storage/models/usbills-simple-cnn-v2-model',
                                                             save_best_only=True,
                                                             verbose=1)

STEPS_PER_EPOCH = train_generator.samples // train_generator.batch_size
VALIDATION_STEPS = valid_generator.samples // valid_generator.batch_size










cnn_model_history = cnn_model.fit(train_generator, epochs=CNN_MODEL_EPOCHS,steps_per_epoch = STEPS_PER_EPOCH,
                                  validation_data=valid_generator, validation_steps=VALIDATION_STEPS,
                                  callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, min_delta=1e-2, verbose=1),
                                             tf.keras.callbacks.ModelCheckpoint(filepath='../../Downloads',
                                                                                save_best_only=True, verbose=1)])


acc = cnn_model_history.history['accuracy']
val_acc = cnn_model_history.history['val_accuracy']

loss = cnn_model_history.history['loss']
val_loss = cnn_model_history.history['val_loss']

plt.figure(figsize=(20, 8))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.show()



