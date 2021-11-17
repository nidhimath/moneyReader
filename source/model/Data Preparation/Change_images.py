# baseline model with dropout for the dogs vs cats dataset
# import sys
# from keras.utils import to_categorical
# from keras.optimizers import SGD
# from keras.preprocessing.image import ImageDataGenerator
#
# from keras.models import load_model
# from PIL import Image
# import numpy as np
# import sys
#
import skimage
import scipy
from scipy import signal
from skimage import io
from skimage.filters import gaussian
from skimage.transform import rescale
#
# import time
# from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
# from keras.layers.convolutional import Conv2D, MaxPooling2D
# from keras.constraints import maxnorm
# from keras.utils import np_utils
# from keras.datasets import cifar10
# import matplotlib.pyplot as plt
# import pickle

import glob
import cv2
import os
import matplotlib as plt


images = []

filepath_png = "/Users/Nidhi/PycharmProjects/PracticeAI/Altering Images/data/*/*/*.png"
filepath_jpg = "/Users/Nidhi/PycharmProjects/PracticeAI/Altering Images/data/*/*/*.jpg"
filepath_jpeg = "/Users/Nidhi/PycharmProjects/PracticeAI/Altering Images/data/*/*/*.jpeg"

images_names_png = glob.glob(filepath_png)
image_names_jpg = glob.glob(filepath_jpg)
image_names_jpeg = glob.glob(filepath_jpeg)

## LIST OF IMAGES
for i in image_names_jpeg:
    filepath = os.path.abspath(i)
    print (filepath)
    img = cv2.imread(i, cv2.IMREAD_COLOR)
    img2 = cv2.resize(img, dsize=(300, 300), interpolation=cv2.INTER_CUBIC)
    #img3 = skimage.feature.canny(img2, sigma=2)

    # Hrr = skimage.feature.hessian_matrix(img2)[0]
    # Hrc = skimage.feature.hessian_matrix(img2)[1]
    # Hcc = skimage.feature.hessian_matrix(img2)[2]
    # cv2.imwrite(filepath, Hrc)

    images.append(img2)
    # plt.imshow(img3, cmap="gray")
    # plt.show()
    cv2.imwrite(filepath, img2)

for i in image_names_jpg:
    filepath = os.path.abspath(i)
    img = cv2.imread(i, cv2.IMREAD_COLOR)
    img2 = cv2.resize(img, dsize=(300, 300), interpolation=cv2.INTER_CUBIC)
    #img3 = skimage.feature.canny(img2, sigma=2)
    images.append(img2)
    # plt.imshow(img3, cmap="gray")
    # plt.show()
    cv2.imwrite(filepath, img2)



for i in images_names_png:
    filepath = os.path.abspath(i)
    img = cv2.imread(i, cv2.IMREAD_COLOR)
    img2 = cv2.resize(img, dsize=(300, 300), interpolation=cv2.INTER_CUBIC)
    #img3 = skimage.feature.canny(img2, sigma=2)
    images.append(img2)
    # plt.imshow(img3, cmap="gray")
    # plt.show()
    cv2.imwrite(filepath, img2)



for i in images:
    cv2.imwrite(str('Matches/matches' + str(i) + '.jpg'), i)


