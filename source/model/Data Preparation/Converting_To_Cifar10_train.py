# python script for converting 32x32 pngs to format
from PIL import Image
import os
from array import *
import pickle
import numpy as np

data = array('B')

batch = {'data': [], 'labels': []}

image_size = 300


for dirname, dirnames, filenames in os.walk('/Users/Nidhi/PycharmProjects/PracticeAI/Altering Images/data/train'):
    for filename in filenames:
        if (filename.endswith('.png')):

            ################
            # grab the image#
            ################

            im = Image.open(os.path.join(dirname, filename))
            pix = im.load()

            # store the class name from look at path
            class_name = int(os.path.join(dirname).split('/')[-1])
            print(class_name)
            print(os.path.join(dirname, filename))

            ###########################
            # get image into byte array#
            ###########################

            # create array of bytes to hold stuff

            # first append the class_name byte
            batch['labels'].append(class_name)

            # then write the rows
            # Extract RGB from pixels and append
            # note: first we get red channel, then green then blue
            # note: no delimeters, just append for all images in the set

            for color in range(0, 3):
                for x in range(0, image_size):
                    for y in range(0, image_size):
                        batch['data'].append(pix[x, y][color])


        if (filename.endswith('.jpg')):

            ################
            # grab the image#
            ################

            im = Image.open(os.path.join(dirname, filename))
            pix = im.load()

            # store the class name from look at path
            class_name = int(os.path.join(dirname).split('/')[-1])
            print(class_name)
            print(os.path.join(dirname, filename))

            ###########################
            # get image into byte array#
            ###########################

            # create array of bytes to hold stuff

            # first append the class_name byte
            batch['labels'].append(class_name)

            # then write the rows
            # Extract RGB from pixels and append
            # note: first we get red channel, then green then blue
            # note: no delimeters, just append for all images in the set

            for color in range(0, 3):
                for x in range(0, image_size):
                    for y in range(0, image_size):
                        batch['data'].append(pix[x, y][color])


        if (filename.endswith('.jpeg')):

            ################
            # grab the image#
            ################

            im = Image.open(os.path.join(dirname, filename))
            pix = im.load()

            # store the class name from look at path
            class_name = int(os.path.join(dirname).split('/')[-1])
            print(class_name)
            print(os.path.join(dirname, filename))

            ###########################
            # get image into byte array#
            ###########################

            # create array of bytes to hold stuff

            # first append the class_name byte
            batch['labels'].append(class_name)

            # then write the rows
            # Extract RGB from pixels and append
            # note: first we get red channel, then green then blue
            # note: no delimeters, just append for all images in the set

            for color in range(0, 3):
                for x in range(0, image_size):
                    for y in range(0, image_size):
                        batch['data'].append(pix[x, y][color])

############################################
# write all to binary, all set for cifar10!!#
############################################

batch['data'] = np.array(batch['data'])

output_file = open('cifar10-ready.bin', 'wb')
pickle.dump(batch, output_file)
#data.tofile(output_file)
output_file.close()
