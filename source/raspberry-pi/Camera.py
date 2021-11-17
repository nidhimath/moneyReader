import os, picamera, time, datetime
import sys, cv2
import RPi.GPIO as GPIO

from keras.models import load_model
from PIL import Image
import numpy as np

todayDate = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
originalImageFilename = "/home/pi/BlindPeople/images/" + todayDate + ".png"

def take_picture():
    # Captures the photo of the bill
    camera = picamera.PiCamera()
    GPIO.setmode (GPIO.BCM)
    GPIO.setup(22,GPIO.OUT)
    GPIO.output(22,GPIO.LOW)
    time.sleep(1)
    camera.capture(originalImageFilename)
    GPIO.output(22,GPIO.HIGH)
    GPIO.cleanup()

def get_prediction(originalImageFilename):
    batch_size = 512
    model = load_model('blind.h5')
    data = []
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    im = Image.open(originalImageFilename)
    img = im.load()

    for color in range(0, 3):
        for x in range(0, batch_size):
            for y in range(0, batch_size):
                data.append(img[x, y][color])

    img = np.reshape(data, [1, 3, batch_size, batch_size]).transpose(0, 2, 3, 1)
    classes = model.predict_classes(img)
    return classes

if __name__ == '__main__':

    take_picture()

    print(get_prediction(originalImageFilename))

    # Deletes the temporary files
    #os.system("rm " + originalImageFilename)
