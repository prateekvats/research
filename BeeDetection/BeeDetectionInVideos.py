#!/usr/bin/python


#############################################
# module: vid_to_frames.py
# description: takes a video and generates
# a directory of frames (e.g., 32x32 frames)
# cropped around centers of detected motion.
#############################################

'''
Sample call:
-v is a path to a video file
-d is a path where the frames are stored
-bckgrnd is a background subtraction method (mog, mog2, knn) 
-fs is the frame size (e.g., 32)
$ python vid_to_frames.py
        -v nn/nn_beetracking/vids/192_168_4_6-2017-05-12_18-02-12.mp4
        -d nn/nn_beetracking/vids/frames/
        -bckgrnd mog
        -fs 32
        -srd nn/nn_beetracking/roi/
'''

# import the necessary packages
import csv
import datetime
import glob
import shutil
from collections import defaultdict

import numpy as np
import argparse
import cv2
import os

from sklearn.utils import random, shuffle

from Background import subtractBackground
from BeeClassifierNet import BeeClassifierNet
import time


args = {}
#The hive we are trying to assess
args['hive'] = '4.10'
timeZone = '0'
#The video type we are trying to assess
args['hiveType'] = 'twobox'
#The video directory that contains the videos we are trying to assess
# VIDEO_DIR = '/home/prateek/Desktop/BeeDetectionTestVideos_'+args['hiveType']+'_' + args['hive'] + '/'
VIDEO_DIR = '/home/prateek/Desktop/anova_analysis/videos/'+args['hive']+'/'+timeZone+'/'
#The directory of the model we are using to detec the bees
MODEL_DIR = 'models/combinedbox_ALL_DATA/ALL_DATA.tfl'


# args['roi_dir'] = '/home/prateek/Desktop/BeeDetectionTestVideos_'+args['hiveType']+'/rois/'
encoding = '0'

if args['hive']   =='4.5':
    encoding='0'
elif args['hive'] =='4.7':
    encoding='1'
elif args['hive'] =='4.8':
    encoding='2'
elif args['hive'] =='4.10':
    encoding='3'

args['roi_dir'] = '/home/prateek/Desktop/anova_analysis/images/'+encoding+timeZone+'/'

#Background subtraction method
args['bckgrnd'] = 'mog'
args['frame_size'] = 150 if args['hiveType']=='onebox' else 90
args['video_directory'] = VIDEO_DIR
args['model_dir'] = MODEL_DIR

#The directory that consists of the temperature files
args['temperature_directory'] = '/home/prateek/Desktop/BeeCountCoRelationData/R_4_7/'+args['hiveType']+'/*/temperatures.txt'

#This is the part that loads the model
print 'Loading model...'
beeClassifier = BeeClassifierNet()
model = beeClassifier.Model4(64, 64, 3)
model.load(args['model_dir'])
print 'Model Loaded.'

# if not os.path.exists(ROI_DIR):
#       os.makedirs(ROI_DIR)

ORIG_FRAME = None
SAVE_EACH_FRAME_FLAG = False
LOWER_CNT_RADIUS = 3
UPPER_CNT_RADIUS = 15
BCKGRND = 'KNN'  # values are 'MOG', 'MOG2', 'KNN'.

if args['bckgrnd'] == 'knn':
    BCKGRND = 'KNN'
elif args['bckgrnd'] == 'mog':
    BCKGRND = 'MOG'
elif args['bckgrnd'] == 'mog2':
    BCKGRND = 'MOG2'
else:
    raise Exception('Unknown Background Method ' + args['bckgrnd'])

FRAME_SIZE = int(args['frame_size'])


def createFrameDirAndFilename(vid_path, frame_dir):
    h, t = os.path.split(vid_path)
    dir_name = t.split('.')[0]
    if frame_dir[-1] == '/':
        return frame_dir + dir_name + '/', dir_name
    else:
        return frame_dir + '/' + dir_name + '/', dir_name


def cropROI(frame, x, y, frame_size):
    # print 'cropROI', x, y, frame_size
    # print frame.shape
    nrows, ncols, nc = frame.shape
    sc = int(x - frame_size / 2)
    sr = int(y - frame_size / 2)
    er, ec = 0, 0
    if sc < 0:
        sc = 0
    if sr < 0:
        sr = 0
    er = int(sr + frame_size)
    if er >= nrows:
        sr = nrows - frame_size
        er = nrows
    ec = int(sc + frame_size)
    if ec >= ncols:
        sc = ncols - frame_size
        ec = ncols
    # print sr, er, sc, ec
    roi = frame[sr:er, sc:ec]
    # roi.reshape(32, 32)
    # print 'roi.shape', roi.shape
    assert (roi.shape[0] == frame_size)
    return roi


def dataPreprocessing(image, width, height):
    x = []
    if image.shape[0] > 0 and image.shape[1] > 0:
        image = cv2.resize(image, (width, height))
        x.append(image)
        x = np.array(x, dtype="float") / 255.0

        return x
    else:
        return None


def isBee(model, img):
    processedImg = dataPreprocessing(img, 64, 64)
    if processedImg is not None:
        prediction = model.predict_label(processedImg)
        prediction = np.argmax(prediction[0])
        return prediction
    else:
        return None


def cropRoiMotion(frame, model, orig_frame, tick, frame_size,
                  lower_cnt_radius=3, upper_cnt_radius=15,
                  save_roi_flag=True):
    cnts = cv2.findContours(frame.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None
    beeCount = 0
    if len(cnts) > 0:
        for cn, c in enumerate(cnts):
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            if lower_cnt_radius <= radius <= upper_cnt_radius:
                # crop a 32x32 from an image
                roi = cropROI(orig_frame, x, y, frame_size)
                ## Cesar, you just need to save roi. That's it.                 
                # predictions, rslt = classifyImage(conv_nn, roi)
                # print predictions, rslt

                if save_roi_flag == True:

                    if isBee(model, roi) == 0:
                        beeCount += 1
                        cv2.circle(orig_frame, (int(x), int(y)), int(radius),
                                   (0, 255, 255), 2)
                        # cv2.imwrite(args['roi_dir'] + '' + str(tick) + '.png', roi)
                    else:
                        pass
                        # cv2.imwrite(args['roi_dir'] + 'non-bees/' + str(tick) + '.png', roi)

    return orig_frame, beeCount


def processFrameForRoiMotion(frame, orig_frame,
                             tick, frame_size, roi_dir,
                             bckgrnd='MOG',
                             lower_cnt_radius=3,
                             upper_cnt_radius=15,
                             draw_flag=True):
    sb = subtractBackground(frame, bckgrnd=bckgrnd)
    return cropRoiMotion(sb, model, orig_frame, tick, frame_size,
                         lower_cnt_radius=lower_cnt_radius,
                         upper_cnt_radius=upper_cnt_radius)

#This function saves temperature in a dictionary from one temperature file
def getTemperature(temperatureFile, hive):
    temperatureDictionary = {}
    with open(temperatureFile) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')
        for row in csv_reader:
            time = datetime.datetime.strptime(row[0], "%Y-%m-%d_%H-%M-%S")
            temperature = float(row[1])
            temperatureKey = hive + '_' + time.strftime('%m-%d-%y') + '_' + str(time.hour)
            if temperatureKey not in temperatureDictionary:
                temperatureDictionary[temperatureKey] = temperature
            else:
                temperatureDictionary[temperatureKey] += temperature
                temperatureDictionary[temperatureKey] /= 2

    return temperatureDictionary

#This function saves temperature from all the temperature files into a dictionary
def getAllTemperatures(hiveFolder='/home/prateek/Desktop/BeeCountCoRelationData/R_4_7/one_box/*/temperatures.txt'):
    temperatureDictionary = {}
    for folders in glob.glob(hiveFolder):
        temperatureDictionary.update(getTemperature(folders, args['hive']))
    return temperatureDictionary

#This function takes in a dictionary object and converts it into a csv file
def createCSVOfBeeCounts(dataDictionary, outputFile='beeCountData.csv'):
    fieldnames = ['Key', 'Bee_Detections', 'Temperature']
    tempDict = getAllTemperatures(args['temperature_directory'])
    with open(outputFile, 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for k in dataDictionary:
            if k in tempDict:
                writer.writerow([k, dataDictionary[k], tempDict[k]])
            else:
                writer.writerow([k, dataDictionary[k], '-'])


#I created this function to pick and copy videos from 8 time zones from a day to rename them to the format of 01-01-18_18:59
def getVideos(destinationFolder,hive, folder='/home/prateek/Desktop/BeeCountCoRelationData/R_4_7/02jun2018/'):
    videos = glob.glob(folder + '*/*.mp4')
    videoDictionary = defaultdict(list)
    for video in videos:
        videoCreationTimeString = time.ctime(os.path.getmtime(video))
        videoCreationTime = datetime.datetime.strptime(videoCreationTimeString, "%a %b %d %H:%M:%S %Y")
        videoCreationDate = videoCreationTime.date()

        key = videoCreationDate.strftime('%m-%d-%y_%H:%M')

        if key not in videoDictionary:
            videoDictionary[key] = [video]
        else:
            videoDictionary[key].append(video)

    dates = videoDictionary.keys()
    dates = shuffle(dates)[:5]
    selectedVideosDictionary = {}
    for d in dates:
        videos = videoDictionary[d]
        shuffle(videos)
        key1 = d + '_8-9.59'
        key2 = d + '_10-10.59'
        key3 = d + '_11-12.59'
        key4 = d + '_12-12.59'
        key5 = d + '_13-14.59'
        key6 = d + '_15-15.59'
        key7 = d + '_16-17.59'
        key8 = d + '_18-19.59'
        for v in videos:
            videoCreationTimeString = time.ctime(os.path.getmtime(v))
            videoCreationTime = datetime.datetime.strptime(videoCreationTimeString, "%a %b %d %H:%M:%S %Y")
            hour = videoCreationTime.hour
            if 8 <= hour < 10:
                selectedVideosDictionary[key1] = v
            elif 10 <= hour < 11:
                selectedVideosDictionary[key2] = v
            elif 11 <= hour < 13:
                selectedVideosDictionary[key3] = v
            elif 13 <= hour < 14:
                selectedVideosDictionary[key4] = v
            if 14 <= hour < 15:
                selectedVideosDictionary[key5] = v
            elif 15 <= hour < 17:
                selectedVideosDictionary[key6] = v
            elif 17 <= hour < 18:
                selectedVideosDictionary[key7] = v
            elif 18 <= hour < 20:
                selectedVideosDictionary[key8] = v

    for k in selectedVideosDictionary:
        videoCreationTimeString = time.ctime(os.path.getmtime(selectedVideosDictionary[k]))
        videoCreationTime = datetime.datetime.strptime(videoCreationTimeString, "%a %b %d %H:%M:%S %Y")
        videoName = hive+'-'+videoCreationTime.strftime('%m-%d-%y_%H:%M')
        shutil.copy(selectedVideosDictionary[k], destinationFolder + videoName + '.mp4')

def getVideosOfHiveByTimeZones(hive=5):
    for files in glob.glob('/home/prateek/Desktop/BeeCountCoRelationData/R_4_'+str(hive)+'/onebox/*/'):
        getVideos('/home/prateek/Desktop/BeeMonitoringVideos/',"4_"+str(hive),files)


#This function takes in a directory that contains videos with their names in the following format 01-01-18_13:59
#This format is essential because it tells you when the video was recorded.
def processVideosForRoiMotion(videoDir,hiveCode=0,timeZonecode=0,showFrames=False):
    global args
    videos = glob.glob(videoDir + '*.mp4')
    FRAME_COUNTER = 0
    videoCounter = 0
    beeCountDictionary = {}

    # args['roi_dir'] = '/home/prateek/Desktop/anova_analysis/images/twobox/' + hiveCode + timeZonecode + '/'

    for v in videos:
        videoCreationTimeString = os.path.basename(v).replace('.mp4', '')
        # videoCreationTimeString = time.ctime(os.path.getmtime(v))
        videoCreationTime = datetime.datetime.strptime(videoCreationTimeString, '%m-%d-%y_%H:%M')
        # videoCreationTime = datetime.datetime.strptime(videoCreationTimeString,'%a %b %d %H:%M:%S %Y')

        videoCreationDate = videoCreationTime.date()
        videoCreationHour = videoCreationTime.hour

        hive = args['hive']
        beeCountKey = hive + '_' + videoCreationDate.strftime('%m-%d-%y') + '_' + str(videoCreationHour)
        if beeCountKey not in beeCountDictionary:
            beeCountDictionary[beeCountKey] = 0

        camera = cv2.VideoCapture(v)
        FRAME_DIR, FRAME_FILENAME = \
            createFrameDirAndFilename(v,
                                      os.path.join(os.path.dirname(v), 'frames/'))

        if FRAME_DIR[-1] != '/':
            print 'ERROR'

        # print 'FRAME_DIR=', FRAME_DIR
        # print 'FRAME_FILENAME=', FRAME_FILENAME

        # print FRAME_DIR
        # print FRAME_FILENAME

        if not os.path.exists(FRAME_DIR):
            os.makedirs(FRAME_DIR)

        # print 'Processing video from Date:', videoCreationDate, ' and time:', videoCreationTime
        # keep looping
        while True:
            # grab the current frame
            (grabbed, frame) = camera.read()

            # if we are viewing a video and we did not grab a frame,
            # then we have reached the end of the video
            if v and not grabbed:
                break

            frc = frame.copy()
            # cv2.imwrite(FRAME_DIR + FRAME_FILENAME + '_' + str(FRAME_COUNTER) + '.png', frc)
            ROI_DIR = ''
            roi_dir_folder = os.path.join(ROI_DIR, str(videoCounter))
            if not os.path.exists(roi_dir_folder):
                os.mkdir(roi_dir_folder)
            pfr, count = processFrameForRoiMotion(frame, frc, FRAME_COUNTER, FRAME_SIZE, roi_dir_folder + '/',
                                                  bckgrnd=BCKGRND)
            # cv2.imwrite(FRAME_DIR + 'pfr_' + str(FRAME_COUNTER) + '.png', pfr)
            if showFrames:
                cv2.imshow('Current Frame', frc)
            FRAME_COUNTER += 1
            beeCountDictionary[beeCountKey] += count
            key = cv2.waitKey(1) & 0xFF
            # if the 'q' key is pressed, stop the loop
            if key == ord('q'):
                break

        print 'Total Bee count:', beeCountDictionary[beeCountKey]
        print videoCounter + 1, '/', len(videos), ' done.'
        # cleanup the camera and close any open windows
        camera.release()
        cv2.destroyAllWindows()

        videoCounter += 1

    return beeCountDictionary


#When given a video directory, it returns a bee count dictionary. The video names need to be in the format %m-%d-%y_%H:%M
if __name__=='__main__':

    hives = ['4.7','4.8','4.10']
    for hive in hives:

        # hiveCode = str(i)
        # timeCode = str(j)
        #
        VIDEO_DIR = '/home/prateek/Desktop/BeeDetectionTestVideos_twobox_'+hive+'/'
        #
        print "--------------------------Processing Hive:",hive," -------------------------------------"

        beeCountDict = processVideosForRoiMotion(VIDEO_DIR)
        createCSVOfBeeCounts(beeCountDict, outputFile='beeCountData_' + hive + '_'+args['hiveType']+'.csv')
    # getVideosOfHiveByTimeZones(hive=5)

