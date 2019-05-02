import glob
import os

import BeeClassifierNetTraining as bct
from BeeClassifierNet import BeeClassifierNet
import ModelEvaluation as me

width = 64
height = 64
channels = 3
b = BeeClassifierNet().Model4(width,height,channels)


#This unit test trains onebox dummy model for 1 epoch
def unitTest1():

    bct.trainOneboxModel("dummy_onebox",b,1)
    f = glob.glob('SavedModels/*/dummy_onebox*')
    r = len(f)>0
    if r:
        assert True,'Pass'
    else:
        assert False,'Unit Test 1 failed.'

#This unit test trains twobox dummy model for 1 epoch
def unitTest2():
    bct.trainTwoboxModel("dummy_twobox",b,1)
    f = glob.glob('SavedModels/*/dummy_twobox*')
    r = len(f)>0
    if r:
        assert True,'Pass'
    else:
        assert False,'Unit Test 2 failed.'

#This unit test trains combined dummy model for 1 epoch
def unitTest3():

    bct.trainCombinedModel("dummy_combinedbox",b,1)
    f = glob.glob('SavedModels/*/dummy_combinedbox*')
    r = len(f)>0
    if r:
        assert True,'Pass'
    else:
        assert False,'Unit Test 3 failed.'

#This unit test verifies model evaluation and shows you that the saved model can be successfully loaded and verified using dummy data
def unitTest4():
    try:
        me.evaluateModel('combinedbox_ALL_DATA','ALL_DATA',b,'all')
        print 'Unit test passed.'
    except Exception as e:
        assert False,'Unit test 4 failed.'+e.message


if __name__=='__main__':
    unitTest1()
    unitTest2()
    unitTest3()
    unitTest4()
