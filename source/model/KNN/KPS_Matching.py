import matplotlib.pyplot as plt
import numpy as np
import scipy
import skimage
from skimage import feature
from scipy import signal
from skimage import io
from skimage.filters import gaussian
from skimage.transform import rescale
from skimage.feature import corner_harris
import sklearn
from sklearn import metrics
import glob
import cv2



images = []

filepath = "/Users/Nidhi/PycharmProjects/PracticeAI/KNN_Machine/data/*/*/*.png"

images_names = glob.glob(filepath)

## LIST OF IMAGES
for i in images_names:
    img = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.resize(img, dsize=(300, 300), interpolation=cv2.INTER_CUBIC)
    #img3 = skimage.feature.canny(img2, sigma=2)
    images.append(img2)
    # plt.imshow(img3, cmap="gray")
    # plt.show()

print(images_names)

# TEST IMAGE ONLY

test_image = "/Users/Nidhi/Downloads/Blind Project/KPS Data/Ten-4.jpg"

test_image = cv2.imread(test_image, cv2.IMREAD_GRAYSCALE)
test_image = cv2.resize(test_image, dsize=(300, 300), interpolation=cv2.INTER_CUBIC)


# # For all images: distance method
# for i in images:
#     for j in images:
#         i = i.flatten().reshape(1, -1)
#         j = j.flatten().reshape(1, -1)
#         dist = sklearn.metrics.pairwise.euclidean_distances(i,j)
#         print(dist)


# # For single image: distance method
# test_image = images[0]
# for j in range(len(images)):
#     test_image = test_image.flatten().reshape(1, -1)
#     images[j] = images[j].flatten().reshape(1, -1)
#     dist = sklearn.metrics.pairwise.euclidean_distances(test_image,images[j])
#     print("Here is the distance between test images and image " + images_names[j] + ":" )
#     print(dist)



# Matching Method

#num_test_image = 11
max = 0
type = 0
#test_image = images[num_test_image]

for i in range(len(images)):
    # Getting image
    image = test_image
    image2 = images[i]

    # using kaze to find key points and descriptions
    kaze = cv2.AKAZE_create()
    kps, descs = kaze.detectAndCompute(image, None)
    kps2, descs2 = kaze.detectAndCompute(image2, None)

    # Using a brute force matcher
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descs, descs2, k=2)

    # Finding the good matches using distances
    good_matches = []
    for m, n in matches:
        if m.distance < 0.9 * n.distance:
            good_matches.append([m])

    #print("Number of good matches for " + images_names[i] + " is: " + str(len(good_matches)))

    img3 = cv2.drawMatchesKnn(image, kps, image2, kps2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(str('Matches/matches' + str(i) + '.jpg'), img3)

    if(len(good_matches)>max): #and i!=num_test_image):
        max = len(good_matches)
        type = i


print(max)
print(images_names[type])

# if (str(images_names[type]) == "/Users/Nidhi/PycharmProjects/PracticeAI/KNN_Machine/data/*/0/*.png"):
#     print("This is a one dollar bill")
#
# if (str(images_names[type]) == "/Users/Nidhi/PycharmProjects/PracticeAI/KNN_Machine/data/*/1/*.png"):
#     print("This is a five dollar bill")
#
# if (str(images_names[type]) == "/Users/Nidhi/PycharmProjects/PracticeAI/KNN_Machine/data/*/2/*.png"):
#     print("This is a ten dollar bill")
#
# if (str(images_names[type]) == "/Users/Nidhi/PycharmProjects/PracticeAI/KNN_Machine/data/test/3/IMG_2488.png"):
#     print("This is a twenty dollar bill")
#
# else:
#     print("Error")
#     print(str(images_names[type]))
#
#
# if ("data/test/3/IMG_2488.png" == "data/*/3/*.png"):
#     print("Works")
#
# else:
#     print("Doesn't work")
