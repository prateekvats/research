import os

import DataProcessing as dp
from BeeClassifierNet import BeeClassifierNet


#These variables take in the path of the training data split
training_imageFolder_onebox = 'BEE2/OneBox/training/'
training_imageFolder_twobox = 'BEE2/TwoBox/training/'

#These variables take in the path of the testing data split
testing_imageFolder_onebox = 'BEE2/OneBox/testing/'
testing_imageFolder_twobox = 'BEE2/TwoBox/testing/'

#Using this for code testing
# imageFolder_onebox = 'dummyData/'
# imageFolder_twobox = 'dummyData/'


training_onebox = dp.getRawImagePaths(training_imageFolder_onebox)
training_twobox = dp.getRawImagePaths(training_imageFolder_twobox)

testing_onebox = dp.getRawImagePaths(testing_imageFolder_onebox)
testing_twobox = dp.getRawImagePaths(testing_imageFolder_twobox)



width = 64
height = 64
channel = 3
NUM_EPOCHS = 30
BATCH_SIZE = 64




#The function is used to train models on onebox dataset takes in three arguments
#modelName takes in a string, this determines the file name with which the model will be saved
#model is the tflearn model object
#epochs determines the number of epochs the training needs to be done for
def trainOneboxModel(modelName,model,epochs,modelPath = "SavedModels"):
    trainX, trainY = dp.dataPreprocessing(training_onebox)
    testX, testY = dp.dataPreprocessing(testing_onebox)
    modelPath += "onebox_" + modelName
    if not os.path.exists(modelPath):
        os.mkdir(modelPath)

    MODEL = model

    MODEL.fit(trainX, trainY, n_epoch=epochs,
                  shuffle=True,
                  validation_set=(testX,testY),
                  show_metric=True,
                  batch_size=BATCH_SIZE,
                  run_id=modelName+"_onebox")


    MODEL.save(modelPath+'/'+modelName+'.tfl')



#The function is used to train models on twobox dataset and takes in three arguments
#modelName takes in a string, this determines the file name with which the model will be saved
#model is the tflearn model object
#epochs determines the number of epochs the training needs to be done for
def trainTwoboxModel(modelName,model,epochs,modelPath = "SavedModels"):
    trainX, trainY = dp.dataPreprocessing(training_twobox)
    testX, testY = dp.dataPreprocessing(testing_twobox)

    modelPath += "twobox_" + modelName
    if not os.path.exists(modelPath):
        os.mkdir(modelPath)

    model.fit(trainX, trainY, n_epoch=epochs,
                  shuffle=True,
                  validation_set=(testX, testY),
                  show_metric=True,
                  batch_size=BATCH_SIZE,
                  run_id=modelName+"_twobox")

    model.save(modelPath+'/'+modelName+'.tfl')



#This function trains both onebox and twobox together and used when you want to train multiple models on both onesuper and twosuper dataset
#modelObjects takes in a list tflearn DNN objects
#modelNames takes in a list of names you want to give the models being trained
#commonModelFolder is the folder where you want all these models to be trained and persisted
def trainModelsTogether(modelObjects,modelNames,commonModelFolder='',NUM_EPOCHS=30):
    modelCount = len(modelObjects)
    assert modelCount is len(modelNames)
    modelPath = 'SavedModels/'+commonModelFolder+'/'

    for i in range(modelCount):
        # Call this function to train on only onebox data
        trainOneboxModel(modelNames[i], modelObjects[i], NUM_EPOCHS,modelPath=modelPath)
        # Call this function to train on only twobox data
        # trainTwoboxModel(modelNames[i], modelObjects[i], NUM_EPOCHS,modelPath=modelPath)



if __name__=='__main__':

    b =  BeeClassifierNet()
    width = 64
    height = 64
    channel = 3

    models = [
        b.vggNet(width,height,channel)
        # b.resNet(width,height,channel)
               ]


    modelNames = [
         "vggNet",
        #    "resNet",
                   ]


    trainModelsTogether(models,modelNames,commonModelFolder='ex_new',NUM_EPOCHS=30)