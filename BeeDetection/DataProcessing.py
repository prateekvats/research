
# grab the image paths and randomly shuffle them
import cv2
import glob
import numpy as np
import tflearn
from sklearn.externals.joblib import delayed
from sklearn.utils import shuffle
from tflearn import data_utils

width = 64
height = 64
channel = 3

# grab the image paths and randomly shuffle them


# labels = []

def getRawImagePaths(imageFolder):
    nonBeesImages = glob.glob(imageFolder+'non-bees/*/*.png')
    beeImages = glob.glob(imageFolder+'bees/*/*.png')
    shuffle(nonBeesImages)
    imagePaths = []
    # for f in nonBeesImages[:len(beeImages)]:
    #     imagePaths.append((f,0))
    for f in nonBeesImages:
        imagePaths.append((f,0))
    for f in beeImages:
        imagePaths.append((f,1))
    return imagePaths



def dataPreprocessing(data,startIndex = None,endIndex = None):
    batchX = []
    batchY = []

    if startIndex is None:
        startIndex = 0
    if endIndex is None:
        endIndex = len(data)

    for i in range(startIndex,endIndex):
        image = cv2.imread(data[i][0])
        if image.shape[0] > 0 and image.shape[1] > 0:
            image = cv2.resize(image, (width, height))
            image = np.array(image)[:, :, 0:3]
            batchX.append(image)
            batchY.append(data[i][1])

    batchX = np.array(batchX, dtype="float") / 255.0
    batchY = np.array(batchY)
    batchY = data_utils.to_categorical(batchY,nb_classes=2)

    return batchX,batchY