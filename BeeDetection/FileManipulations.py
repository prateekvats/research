import glob
import os
import shutil

from sklearn.utils import shuffle

def createFolder(folderPath):
    if not os.path.exists(folderPath):
        os. mkdir(folderPath)


def combineFilesInFoldersAndShuffle(folderPathList,fileExtension):
    files = []
    for f in folderPathList:
        files.extend(glob.glob(f+'/*.'+fileExtension))
    return shuffle(files)

def moveFiles(files,outputFolder):
    if not os.path.exists(outputFolder):
        os.mkdir(outputFolder)
    for f in files:
        os.rename(f,os.path.join(outputFolder,os.path.basename(f)))

def createBatches(files,batchSize,batchFolderPrefix,outputFolder):
    l=len(files)
    createFolder(outputFolder)
    batchCount=0
    for i in range(0,l,batchSize):
        currentDestinationFolder = os.path.join(outputFolder,batchFolderPrefix+str(batchCount))
        createFolder(currentDestinationFolder)
        for f in files[i:min(i+batchSize,l)]:
            shutil.copy(f,os.path.join(currentDestinationFolder,os.path.basename(f)))
            # moveFiles(f,os.path.join(currentDestinationFolder,os.path.basename(f)))
        print 'Batch ',batchCount,' created.'
        batchCount+=1


if __name__=='__main__':
    # files = combineFilesInFoldersAndShuffle(['/home/prateek/Desktop/BeeCountCoRelationData/R_4_5/rois_onebox_1',
    #                                          '/home/prateek/Desktop/BeeCountCoRelationData/R_4_5/rois_onebox_2'],'png')
    # moveFiles(files[:int(len(files)*0.4)],'/home/prateek/Desktop/BeeCountCoRelationData/R_4_5/rois_onebox_combined')

    # files = glob.glob('/home/prateek/Desktop/BeeCountCoRelationData/R_4_5/rois_onebox_combined/*.png')
    # createBatches(files,batchSize=2000,batchFolderPrefix='oneboxrois',outputFolder='/home/prateek/Desktop/BeeCountCoRelationData/R_4_5/oneboxrois_batches')
    #
    files = glob.glob('/home/prateek/Desktop/BeeCountCoRelationData/R_4_5/rois_twoboxes/*.png')
    files = shuffle(files)
    # files = files[:int(len(files)*0.4)]
    createBatches(files,batchSize=2000,batchFolderPrefix='twoboxesrois',outputFolder='/home/prateek/Desktop/BeeCountCoRelationData/R_4_5/twoboxesrois_batches')