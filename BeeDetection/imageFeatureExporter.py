import csv
import glob
import os
import skimage

import cv2
import numpy as np
import skimage
from skimage.io import imread
from skimage.feature import greycomatrix
from skimage.feature import greycoprops
from sklearn.utils import shuffle

folderPath = "/home/prateek/Desktop/BeeCountCoRelationDataCombined/TwoBox/training/bees/"


def collectAndMoveTestingData(trainingFolder,testingFolder,split=0.20):
    trainingBeeImages = glob.glob(trainingFolder+'bees/*/*.png')
    trainingBeeImages = shuffle(trainingBeeImages)
    trainingBeeImages = trainingBeeImages[: (int)(split * len(trainingBeeImages))]

    trainingNonBeeImages = glob.glob(trainingFolder+'non-bees/*/*.png')
    trainingNonBeeImages = shuffle(trainingNonBeeImages)
    trainingNonBeeImages = trainingNonBeeImages[: (int)(split * len(trainingNonBeeImages))]
    for f in trainingBeeImages:
        os.rename(f , testingFolder+'bees/'+os.path.basename(f))

    for f in trainingNonBeeImages:
        os.rename(f , testingFolder+'non-bees/'+os.path.basename(f))


def createSpectogramFeaturesCSV():
    global folderPath
    groups = glob.glob(folderPath+"*")
    folderPath = "/home/prateek/Desktop/BeeCountCoRelationDataCombined/TwoBox/validation/bees/"
    groups = groups + glob.glob(folderPath+"*")
    content = ""

    with open('/home/prateek/Desktop/anova_analysis/twobox_trainingvalidation_comparison.csv', 'wb') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(["group"]+["F"+str(f) for f in range(1,11)])

        # for g in groups:
            # groupName = os.path.basename(g)
        groupName = "training"
        for image in shuffle(glob.glob(g + "/*.png"))[:5000]:
            img = cv2.imread(image)
            try:
                values, bins = np.histogram(img)
                # content += groupName+','+(','.join([str(x) for x in values]))+'\n'
                wr.writerow([groupName] + [str(x) for x in values])
            except:
                print ("error in image:", image)




def createSingularFeaturesCSV(folderPath):
    images = glob.glob(folderPath+"*.png")
    content = ""

    with open('/home/prateek/Desktop/anova_analysis/twobox_anova_factors.csv', 'a') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        # wr.writerow(["group","contrast","energy","homogeneity"])

        # groupName = os.path.basename(g)
        groupName = "testing"
        for image in shuffle(images)[:1000]:
            img = imread(image, as_gray=True)
            x = []
            im = skimage.img_as_ubyte(img)
            im = im // 32
            g = skimage.feature.greycomatrix(im, [1], [0], levels=8, symmetric=False, normed=True)
            contrast = skimage.feature.greycoprops(g, 'contrast')[0][0]
            energy = skimage.feature.greycoprops(g, 'energy')[0][0]
            homogeneity = skimage.feature.greycoprops(g, 'homogeneity')[0][0]
            # content += groupName+','+(','.join([str(x) for x in values]))+'\n'
            wr.writerow([groupName] + [contrast, energy, homogeneity])

# createSpectogramFeaturesCSV()
createSingularFeaturesCSV('/home/prateek/Desktop/BeeCountCoRelationDataCombined/TwoBox/testing/bees/')

# collectAndMoveTestingData('/home/prateek/Desktop/BeeCountCoRelationDataCombined/TwoBox/training/','/home/prateek/Desktop/BeeCountCoRelationDataCombined/TwoBox/testing/',0.25)



