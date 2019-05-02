import glob

import BeeDetectionInVideos as b

#This unit test runs the data collection function and creates a csv file and verifies if the csv is created or not
#Please note that this will take a couple of minutes to execute
def unit_test1():
    beeCountDict = b.processVideosForRoiMotion('dummyVideo/')
    b.createCSVOfBeeCounts(beeCountDict, outputFile='dummyBeeCount.csv')
    f = glob.glob('dummyBeeCount.csv')
    if len(f)<=0:
        assert False,'Unit test failed.'

if __name__=='__main__':
    unit_test1()