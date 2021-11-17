from keras.models import load_model
from PIL import Image
import numpy as np
import sys

batch_size = 256

model = load_model('blind.h5')

data = []

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

im = Image.open(sys.argv[1])
img = im.load()

for color in range(0, 3):
    for x in range(0, batch_size):
        for y in range(0, batch_size):
            data.append(img[x, y][color])

img = np.reshape(data,[1, 3, batch_size,
                 batch_size]).transpose(0, 2, 3, 1)

classes = model.predict_classes(img)

print(classes)
