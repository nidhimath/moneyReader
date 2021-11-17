from keras.preprocessing import image
import numpy as np
import wget
import keras
import shutil
import os
import glob
import gtts
from playsound import playsound


files = glob.glob('/Users/Nidhi/PycharmProjects/RPiRedo/predictionData/*')
for f in files:
    os.remove(f)


def load_image(img_path, show=False):
    img = image.load_img(img_path, target_size=(150, 150))
    img_tensor = image.img_to_array(img)  # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor,
                                axis=0)  # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.  # imshow expects values in the range [0, 1]


    return img_tensor


# Set up the image URL
image_url = "http://192.168.86.44/capture?_cb=2"

# Use wget download method to download specified image url.
image_filename = wget.download(image_url)
source = "/Users/Nidhi/PycharmProjects/RPiRedo/" + image_filename
destination = "/Users/Nidhi/PycharmProjects/RPiRedo/predictionData"

image_filename = shutil.move(source, destination)

new_image = load_image(image_filename)

model = keras.models.load_model('resnet50_weights_tf_dim_ordering_tf_kernels.h5')

pred = model.predict(new_image)

##Calculating the highest image

highestNumber = -1;
denomination = "";
if pred[0][0] > highestNumber:
    highestNumber = pred[0][0]
    denomination = "Five dollar bill"

if pred[0][1] > highestNumber:
    highestNumber = pred[0][1]
    denomination = "One dollar bill"

if pred[0][2] > highestNumber:
    highestNumber = pred[0][2]
    denomination = "Ten dollar bill"

if pred[0][3] > highestNumber:
    highestNumber = pred[0][3]
    denomination = "Twenty dollar bill"

print(pred)
print(highestNumber)
print(denomination)

print('Image Successfully Downloaded: ', image_filename)

# Text to Speech
tts = gtts.gTTS(denomination)
language = 'en'

tts.save("/Users/Nidhi/PycharmProjects/RPiRedo/predictionData/denomination.mp3")
playsound("/Users/Nidhi/PycharmProjects/RPiRedo/predictionData/denomination.mp3")

