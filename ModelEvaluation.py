import glob
import os

from BeeClassifierNet import BeeClassifierNet
from DataProcessing import dataPreprocessing

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

width = 64
height = 64
channel = 3


#This function returns a list of tuples that contain the filePath of the image and its associated label
#imageFolder takes in the path of directory which contains folder names "non-bee" and "bees"
#imageType is by default None, if imageType is 1, then only a list of non-bee images will be returned and if its 0,
# then only a list of bee images will be returned
def loadValidationImages(imageFolder,imageType=None):
    imagePaths = []

    nonBeesImages = glob.glob(imageFolder + 'non-bees/*/*.png') + glob.glob(imageFolder + 'non-bees/*.png')
    beeImages = glob.glob(imageFolder + 'bees/*/*.png') + glob.glob(imageFolder + 'bees/*.png')

    if(imageType is None or imageType is 1):
        for f in nonBeesImages:
            imagePaths.append((f, 0))
    if(imageType is None or imageType is 0):
        for f in beeImages:
            imagePaths.append((f, 1))

    return imagePaths


#This function evaluates a tf learn model.
#First parameter takes in the modelFolder inside the savedFolder folder
#Second Parameter takes in the modelName inside the modelFolder
#modelObject takes in the DNN tflearn object from BeeClassifierNet.py
#type is onebox and twobox These are the options for this parameter.
def evaluateModel(modelFolder,modelName,modelObject,type):

    validation_onebox = 'BEE2/OneBox/validation/'
    validation_twobox = 'BEE2/TwoBox/validation/'


    rawImageData_onebox_bee = loadValidationImages(validation_onebox,0)
    rawImageData_onebox_nobee = loadValidationImages(validation_onebox,1)

    rawImageData_twobox_bee = loadValidationImages(validation_twobox,0)
    rawImageData_twobox_nobee = loadValidationImages(validation_twobox,1)



    if type=='onebox':
        x_test_bee, y_test_bee = dataPreprocessing(rawImageData_onebox_bee)
        x_test_nobee, y_test_nobee = dataPreprocessing(rawImageData_onebox_nobee)
    elif type=='twobox':
        x_test_bee, y_test_bee = dataPreprocessing(rawImageData_twobox_bee)
        x_test_nobee, y_test_nobee = dataPreprocessing(rawImageData_twobox_nobee)
    else:
        x_test_bee = None
        y_test_bee = None
        x_test_nobee = None
        y_test_nobee = None

    assert x_test_bee is not None and x_test_nobee is not None
    modelPath = 'SavedModels/'+modelFolder+'/'
    modelObject.load(modelPath+modelName+'.tfl')

    print (modelName,' - ',type,' bee accuracy:',modelObject.evaluate(x_test_bee,y_test_bee)[0])
    print(modelName, ' - ', type, ' no bee accuracy:', modelObject.evaluate(x_test_nobee, y_test_nobee)[0])

#This function is used to evaluate multiple models and prints their bee and no-bee accuracy on both onesuper and twosuper data together
#modelObjects is a list of tflearns DNN objects
#modelNames is a list of string. These are the names of the models that were persisted.
#commonModelFolder is the folder in which all of these models are persisted
def evaluateModelsTogether(modelObjects,modelNames,commonModelFolder='intelligentSystemModels'):
    modelCount = len(modelObjects)
    assert modelCount is len(modelNames)
    modelPath = commonModelFolder+'/'

    for i in range(modelCount):
        evaluateModel(modelPath+'onebox_'+modelNames[i], modelNames[i], modelObjects[i], type="onebox")
        evaluateModel(modelPath+'twobox_'+modelNames[i], modelNames[i], modelObjects[i], type="twobox")


#This function is used to run prediction using a model on a single image
#modelFolder takes in the path of the directory where the modelFolder resides
#modelName takes in the name of the persisted model file in the modelFolder
#imagePath is the path of the image, that needs to be predicted by the model
def evaluateImage(modelFolder,modelName,modelObject,imagePath):
    label = 1
    if 'non-bee' in imagePath:
        label = 0
    else:
        label = 1

    x_test, y_test = dataPreprocessing([imagePath], [label])
    modelPath = 'SavedModels/'+modelFolder+'/'
    modelObject.load(modelPath+modelName+'.tfl')
    y_pred_raw = modelObject.predict(x_test)
    print (y_pred_raw)

if __name__=='__main__':
    # b = BeeClassifierNet()
    b =  BeeClassifierNet()
    # model = b.Model4(64,64,3)
    # model = b.otherModel2(64,64,3)
    #Dont run them together. Please run just one at a time.
    #This piece is to evaluate onebox model.
    # evaluateModel('onebox_COMBINED_ONEBOX','COMBINED_ONEBOX',model,'onebox')
    # evaluateModel('onebox_COMBINED_ONEBOX_OTHER2','COMBINED_ONEBOX_OTHER2',model,'twobox')
    # evaluateModel('onebox_COMBINED_ONEBOX_OTHER2','COMBINED_ONEBOX_OTHER2',model,'all')
    #
    # # This piece is to evaluate twobox model.
    # evaluateModel('twobox_COMBINED', 'COMBINED_TWOBOX', model, 'onebox')
    # evaluateModel('twobox_COMBINED_TWOBOX','COMBINED_TWOBOX',model,'twobox')
    # evaluateModel('twobox_COMBINED_TWOBOX_OTHER2', 'COMBINED_TWOBOX_OTHER2', model, 'all')
    #
    # evaluateModel('combinedbox_ALL_DATA_OTHER2', 'ALL_DATA_OTHER2', model, 'onebox')
    # evaluateModel('combinedbox_ALL_DATA_OTHER2','ALL_DATA_OTHER2',model,'twobox')
    # evaluateModel('combinedbox_ALL_DATA_OTHER2','ALL_DATA_OTHER2', model, 'all')

    #This piece is to evaluate all data model.
    # evaluateModel('combinedbox_ALL_DATA','ALL_DATA',model,'alldata')

    # evaluateModelOnGroups('combinedbox_ALL_DATA','ALL_DATA',model,'alldata')
    # evaluateModelOnGroups('onebox_COMBINED_ONEBOX','COMBINED_ONEBOX',model,'alldata')
    # evaluateModelOnGroups('twobox_COMBINED_TWOBOX','COMBINED_TWOBOX',model,'alldata')
    # evaluateModelOnGroups('combinedbox_ALL_DATA','ALL_DATA',model,'alldata')

    models = [
        b.vggNet(width,height,channel),
        # b.resNet(width,height,channel)
        # b.AdamKing(width,height,channel),b.ArihantJain(width,height,channel),
        #       b.AsherGunsay(width,height,channel),b.ChrisKinsey(width,height,channel),
        #       b.DavidSpencer(width, height, channel),
        # b.ManishMeshram(width,height,channel),
        # b.MichaelGeigl(width, height, channel), b.RyanWilliams(width, height, channel),
        # b.VishalSharma(width, height, channel)
        ]
    #
    modelNames = [
        "vggNet",
        # "resNet"
        # "AdamKing","ArihantJain",
        # "AsherGunsay","ChrisKinsey",
        # "DavidSpencer","ManishMeshram",
        # "MichaelGeigl", "RyanWilliams",
        # "VishalSharma",
        # "Ex7"
    ]
    #Ex4_30
    evaluateModelsTogether(models,modelNames,commonModelFolder='ex_new')
    # evaluateModel2('onebox_GENERALIZED', 'model4', model, type="onebox")