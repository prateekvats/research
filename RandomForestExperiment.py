import cPickle
import glob

from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.utils import shuffle

import DataProcessing as dp
import cv2
import numpy as np
from scipy.misc import imread
width = 64
height = 64
# clf = RandomForestClassifier()
# imageFolder_onebox = '/home/kulyukin-lab1/prateeks_workspace/BeeCountCoRelationFullData/OneBoxData/training/'
# imageFolder_twobox = '/home/kulyukin-lab1/prateeks_workspace/BeeCountCoRelationFullData/TwoBoxData/training/'

imageFolder_onebox = '/home/kulyukin-lab1/prateeks_workspace/BeeCountCoRelationDataCombined/OneBoxData/training/'
imageFolder_twobox = '/home/kulyukin-lab1/prateeks_workspace/BeeCountCoRelationDataCombined/TwoBoxData/training/'

def loadValidationImages(imageFolder):
    # grab the image paths and randomly shuffle them
    imagePaths = []

    nonBeesImages = glob.glob(imageFolder + 'non-bees/*/*.png') + glob.glob(imageFolder + 'non-bees/*.png')
    # beeImages = glob.glob(imageFolder + 'bees/*/*.png') + glob.glob(imageFolder + 'bees/*.png')
    # shuffle(nonBeesImages)
    #
    #
    for f in nonBeesImages:
        imagePaths.append((f, 0))
    # for f in beeImages:
    #     imagePaths.append((f, 1))

    return imagePaths

def dataPreprocessing(data,startIndex = None,endIndex = None):
    batchX = []
    batchY = []

    if startIndex is None:
        startIndex = 0
    if endIndex is None:
        endIndex = len(data)

    for i in range(startIndex,endIndex):
        imageFeature = cv2.imread(data[i][0])
        imageFeature = cv2.resize(imageFeature, (width, height))
        nsamples, nx, ny = imageFeature.shape
        imageFeature = imageFeature.reshape((nsamples* nx * ny))
        batchX.append(imageFeature)
        batchY.append(data[i][1])

    batchY = np.array(batchY)

    return batchX,batchY


def trainCombinedModel(model,modelType,type,trees=100):
    trainData = None
    if type=="onebox":
        trainData = dp.getRawImagePaths(imageFolder_onebox)
    elif type == "twobox":
        trainData = dp.getRawImagePaths(imageFolder_twobox)
    elif type == "all":
        trainData = dp.getRawImagePaths(imageFolder_onebox)+dp.getRawImagePaths(imageFolder_twobox)

    assert trainData is not None

    trainX, trainY = dataPreprocessing(trainData)

    print "RF Training started."
    model.fit(trainX,trainY)

    with open('SavedModels/'+modelType+'/'+modelType+"_"+str(trees)+"_"+type+'.pkl', 'wb') as fid:
        cPickle.dump(model, fid)
    #
    #     # load it again
    # with open('my_dumped_classifier.pkl', 'rb') as fid:
    #     gnb_loaded = cPickle.load(fid)
    # preds = clf.predict(tX)
    # print("Accuracy:", accuracy_score(tY, preds))
def trainSVM(model,type):
    trainData = None
    if type=="onebox":
        trainData = dp.getRawImagePaths(imageFolder_onebox)
    elif type == "twobox":
        trainData = dp.getRawImagePaths(imageFolder_twobox)
    elif type == "all":
        trainData = dp.getRawImagePaths(imageFolder_onebox)+dp.getRawImagePaths(imageFolder_twobox)

    assert trainData is not None

    trainX, trainY = dataPreprocessing(trainData)

    print "SVM Training started."
    model.fit(trainX,trainY)

    with open('SavedModels/svm/bee_svm_'+type+'_classifier.pkl', 'wb') as fid:
        cPickle.dump(model, fid)


def kmeansTrain(model,type):
    trainData = None
    if type=="onebox":
        trainData = dp.getRawImagePaths(imageFolder_onebox)
    elif type == "twobox":
        trainData = dp.getRawImagePaths(imageFolder_twobox)
    elif type == "all":
        trainData = dp.getRawImagePaths(imageFolder_onebox)+dp.getRawImagePaths(imageFolder_twobox)

    assert trainData is not None

    trainX, trainY = dataPreprocessing(trainData)

    print "Kmeans Training started."
    model.fit(trainX,trainY)

    with open('SavedModels/kmeans/bee_kmeans_'+type+'_classifier.pkl', 'wb') as fid:
        cPickle.dump(model, fid)

def evaluateModelOnGroups(type="all"):
    #These are the paths i used for training my actualModels
    # validation_onebox = '/home/kulyukin-lab1/prateeks_workspace/BeeCountCoRelationFullData/OneBoxData/validation/'
    # validation_twobox = '/home/kulyukin-lab1/prateeks_workspace/BeeCountCoRelationFullData/TwoBoxData/validation/'
    validation_onebox = '/home/kulyukin-lab1/prateeks_workspace/BeeCountCoRelationCombinedData/OneBoxData/validation/'
    validation_twobox = '/home/kulyukin-lab1/prateeks_workspace/BeeCountCoRelationCombinedData/TwoBoxData/validation/'
    # validation_onebox = 'dummyData/'
    # validation_twobox = 'dummyData/'

    rawImageData_onebox = loadValidationImages(validation_onebox)
    rawImageData_twobox = loadValidationImages(validation_twobox)
    rawImageData_combined = rawImageData_onebox + rawImageData_twobox
    shuffle(rawImageData_combined)
    groups = 50
    perSample = 100
    np.save("anovaImages_groups"+str(groups)+"_sample"+str(perSample),rawImageData_combined)

    with open('SavedModels/randomForest/bee_rf_classifier.pkl', 'rb') as fid:
        clf = cPickle.load(fid)

    for g in range(0,groups):
        x_test, y_test = dataPreprocessing(rawImageData_combined[g*perSample:(g*perSample+perSample)])
        preds = clf.predict(x_test)
        print accuracy_score(y_test, preds)




def evaluateModel(modelObject,model,modelType,modelName,DataType="all"):
    assert model is not None
    assert modelName is not None
    assert modelType is not None
    #These are the paths i used for training my actualModels
    validation_onebox = '/home/kulyukin-lab1/prateeks_workspace/BeeCountCoRelationFullData/OneBoxData/validation/'
    validation_twobox = '/home/kulyukin-lab1/prateeks_workspace/BeeCountCoRelationFullData/TwoBoxData/validation/'
    # validation_onebox = '/home/kulyukin-lab1/prateeks_workspace/BeeCountCoRelationDataCombined/OneBoxData/validation/'
    # validation_twobox = '/home/kulyukin-lab1/prateeks_workspace/BeeCountCoRelationDataCombined/TwoBoxData/validation/'
    # validation_onebox = 'dummyData/'
    # validation_twobox = 'dummyData/'

    rawImageData_onebox = loadValidationImages(validation_onebox)
    rawImageData_twobox = loadValidationImages(validation_twobox)
    rawImageData_combined = rawImageData_onebox + rawImageData_twobox

    if DataType=='onebox':
        x_test, y_test = dataPreprocessing(rawImageData_onebox)
    elif DataType=='twobox':
        x_test, y_test = dataPreprocessing(rawImageData_twobox)
    elif DataType=='all':
        x_test, y_test = dataPreprocessing(rawImageData_combined)
    else:
        x_test, y_test = dataPreprocessing(rawImageData_combined)

        # load it again
    with open('SavedModels/'+modelType+'/'+modelName+'.pkl', 'rb') as fid:
        modelObject = cPickle.load(fid)

    preds = modelObject.predict(x_test)
    print("Accuracy:", accuracy_score(y_test, preds))

if __name__=='__main__':
    #Call this function to train on only onebox data
    # for i in range(20,105,20):

        # clf = RandomForestClassifier(n_estimators=i)
        # trainCombinedModel(clf,"rf_general","onebox",trees=i)
        #
        # clf = RandomForestClassifier(n_estimators=i)
        # trainCombinedModel(clf,"rf_general","twobox",trees=i)
        #
        # clf = RandomForestClassifier(n_estimators=i)
        # trainCombinedModel(clf,"rf_general","all",trees=i)

        # evaluateModel(clf,model="onebox",modelType="rf",modelName="rf_"+str(i)+"_onebox",DataType="onebox")
        # evaluateModel(clf,model="onebox",modelType="rf_general",modelName="rf_general_"+str(i)+"_onebox",DataType="twobox")
        # evaluateModel(clf,model="onebox",modelType="rf_general",modelName="rf_general_"+str(i)+"_onebox",DataType="all")

        # evaluateModel(clf,model="twobox",modelType="rf_general",modelName="rf_general_"+str(i)+"_twobox",DataType="onebox")
        # evaluateModel(clf,model="twobox",modelType="rf",modelName="rf_"+str(i)+"_twobox",DataType="twobox")
        # evaluateModel(clf,model="twobox",modelType="rf_general",modelName="rf_general_"+str(i)+"_twobox",DataType="all")

        # evaluateModel(clf,model="all",modelType="rf_general",modelName="rf_general_"+str(i)+"_all",DataType="onebox")
        # evaluateModel(clf,model="all",modelType="rf_general",modelName="rf_general_"+str(i)+"_all",DataType="twobox")
        # evaluateModel(clf,model="all",modelType="rf_general",modelName="rf_general_"+str(i)+"_all",DataType="all")

    svm = LinearSVC()
    # trainSVM(svm, "onebox")
    # evaluateModel(svm,model="onebox",modelType="svm/non-generalized",modelName="bee_svm_onebox_classifier",DataType="onebox")
    evaluateModel(svm,model="onebox",modelType="svm/non-generalized",modelName="bee_svm_onebox_classifier",DataType="onebox")
    # evaluateModel(svm,model="onebox",modelType="svm/non-generalized",modelName="bee_svm_onebox_classifier",DataType="all")

    svm = LinearSVC()
    # trainSVM(svm, "twobox")
    # evaluateModel(svm,model="twobox",modelType="svm",modelName="bee_svm_twobox_classifier",DataType="onebox")
    evaluateModel(svm,model="twobox",modelType="svm/non-generalized",modelName="bee_svm_twobox_classifier",DataType="twobox")
    # evaluateModel(svm,model="twobox",modelType="svm",modelName="bee_svm_twobox_classifier",DataType="all")

    # svm = LinearSVC()
    # trainSVM(svm,"all")
    # evaluateModel(svm,model="all",modelType="svm",modelName="svm",DataType="onebox")
    # evaluateModel(svm,model="all",modelType="svm",modelName="svm",DataType="twobox")
    # evaluateModel(svm,model="all",modelType="svm",modelName="svm",DataType="all")
    # kmeans = MiniBatchKMeans(batch_size=128)
    # # kmeans = KNeighborsClassifier()
    # kmeansTrain(kmeans, "onebox")
    # evaluateModel(kmeans,"onebox",DataType="onebox")
    # evaluateModel(kmeans,"onebox",DataType="twobox")
    # evaluateModel(kmeans,"onebox",DataType="all")
    #
    # kmeans = KNeighborsClassifier()
    # kmeansTrain(kmeans, "twobox")
    # evaluateModel(kmeans,"twobox",DataType="onebox")
    # evaluateModel(kmeans,"twobox",DataType="twobox")
    # evaluateModel(kmeans,"twobox",DataType="all")
    #
    #
    # kmeans = KNeighborsClassifier()
    # kmeansTrain(kmeans, "all")
    # evaluateModel(kmeans,"all",DataType="onebox")
    # evaluateModel(kmeans,"all",DataType="twobox")
    # evaluateModel(kmeans,"all",DataType="all")

    # evaluateModelOnGroups()
