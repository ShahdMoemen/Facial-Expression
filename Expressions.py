import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob
import os
import imghdr
import collections
from pathlib import Path
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles

type = 0 # select type : 0 -> training 1 -> testing

face_cascade = cv2.CascadeClassifier("C:\\Users\\laptop\\Anaconda3\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml")

path = "E:\Gender\extended-cohn-kanade-images\cohn-kanade-images"
labelsPath = "E:\Gender\Emotion_labels\Emotion"

svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)
hog=cv2.HOGDescriptor()


if os.path.isfile("E:\Gender\HOG_FacialExpressions_modelusingHarr.dat")== True:
    svm = cv2.ml.SVM_load("E:\Gender\HOG_FacialExpressions_modelusingHarr.dat")
    print("Done loading svm expressions")



def testSVM(image):
    img = cv2.resize(image, (64, 128))
    f = hog.compute(img)
    f = np.array(f, np.float32)
    X,Y = np.shape(f)
    f=np.reshape(f,(Y,X))
    response = svm.predict(f)
    fin_res = response[1]
    if fin_res == 0.:
        return "Neutral"
    elif fin_res == 1.:
        return "Angry"
    elif fin_res == 2.:
        return "Contempt"
    elif fin_res == 3.:
        return "Disgusted"
    elif fin_res == 4.:
        return "Afraid"
    elif fin_res == 5.:
        return "Happy"
    elif fin_res == 6.:
        return "Sad"
    elif fin_res == 7.:
        return "Surprised"
    # size = len(labelsList)
    # countTrue = 0
    # for x in range(2000):
    #     label = labelList[x]
    #     res = int(fin_res[x])
    #     if label == res:
    #         countTrue = countTrue + 1
    #
    # score = countTrue / size
    # return score


def testingusingSVM(modelName, featuresList):
    features = np.array(featuresList, np.float32)
    svm = cv2.ml.SVM_load(modelName)
    response = svm.predict(features)
    fin_res = response[1]
    return fin_res

def getaccuracyformodels(testingLabel):
    countOnes = 0
    for x in range(len(testingLabel)):
        if testingLabel[x] == 1:
            countOnes = countOnes+1

    return (countOnes/float(len(testingLabel)))

def computeHogForImage(image):
    hog = cv2.HOGDescriptor()
    resized_image = cv2.resize(image, (64, 128))
    hist = hog.compute(resized_image)
    hist1D = np.ravel(hist)
    return hist1D

def trainusingSVM(featuresList, labelsList, modelName):
    a = np.array(featuresList, np.float32)
    b = np.asarray(labelsList)
    svm.train(a, cv2.ml.ROW_SAMPLE, b)
    svm.save(modelName)

def getfeaturesopticalflow(firstframe, secondframe):
    # print("start get features optical flow")
    if len(firstframe.shape) > 2 and firstframe.shape[2] > 1:
        prvs = cv2.cvtColor(firstframe, cv2.COLOR_BGR2GRAY)
    else:
        prvs = firstframe

    if len(secondframe.shape) > 2 and secondframe.shape[2] > 1:
        next = cv2.cvtColor(secondframe, cv2.COLOR_BGR2GRAY)
    else:
        next = secondframe

    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    h = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    v = cv2.normalize(ang, None, 0, 255, cv2.NORM_MINMAX)
    histH = np.histogram(h, bins=10)
    histV = np.histogram(v, bins=10)
    firstH = histH[0]
    firstV = histV[0]
    return firstH, firstV

def most_common(lst):
    newList = lst.tolist()
    listFreq = []
    for x in range(len(newList)):
        listFreq.append(int(newList[x][0]))
    freq = []
    freq.append(listFreq.count(0))
    freq.append(listFreq.count(1))
    freq.append(listFreq.count(2))
    freq.append(listFreq.count(3))
    freq.append(listFreq.count(4))
    freq.append(listFreq.count(5))
    freq.append(listFreq.count(6))
    freq.append(listFreq.count(7))

    maxNumber=0
    numberAcc= 0

    for idx in range(len(freq)):
        if freq[idx] > maxNumber:
            numberAcc = idx
            maxNumber = freq[idx]
    return numberAcc

def extractfeatures(image_list, currentemotion):
    # print("start extract features")
    featuresList = []
    labelsList = []
    hogfeaturesList = []
    testingLabel = []
    size = len(image_list)
    for z in range(int(size) - 1):
        final_out = []
        histVtemp = []
        ### get HOG for image ###
        faceImage = face_cascade.detectMultiScale(image_list[z],1.3,5)
        # print("here")
        if len(faceImage) > 0:
            print("ok")
            i, j , w, h = faceImage[0]
        else:
            continue
        # cv2.imshow('here', image_list[z][j:j+h, i:i+w])
        # cv2.waitKey()
        hogFeature = computeHogForImage(image_list[z][j:j+h, i:i+w])
        hogfeaturesList.append(hogFeature)

        ### get HOF between video frames ###
        for y in range(8):
            for x in range(8):
                y1 = y *8
                x1 = x * 8
                y2 = (y * 8) + 8
                x2 = (x * 8) + 8
                firstframe = image_list[z][y1:y2, x1:x2]
                secondframe = image_list[z+1][y1:y2, x1:x2]
                histH, histV = getfeaturesopticalflow(firstframe, secondframe)
                for itr in range(10):
                    final_out.append(histH[itr])
                for itr in range(10):
                    histVtemp.append(histV[itr])

        for itr in range(len(histVtemp)):
            final_out.append(histVtemp[itr])

        featuresList.append(final_out)
        labelsList.append(currentemotion)

    if type == '1': # test current image using HOF
        print("start calculating features .. ")
        outarrHOF = testingusingSVM("HOF_FacialExpressions_model.dat", featuresList)
        outarrHOG = testingusingSVM("HOG_FacialExpressions_model.dat", hogfeaturesList)
        allArrays = np.concatenate([outarrHOF, outarrHOG])
        common = most_common(allArrays)
        if common == currentemotion:
            testingLabel = 1
        else:
            testingLabel = 0

    if type == '0':
        return featuresList, labelsList, hogfeaturesList
    else:
        return testingLabel, featuresList, hogFeaturesList

featuresList = []
labelsList = []
hogFeaturesList = []
testingLabelsAll = []

type = input("Select (0) for training (1) for testing: \n")

for filename in os.listdir(path):
    personpath = path + "\\" + str(filename)
    personlabelpath = labelsPath + "\\" + str(filename)
    for person in os.listdir(personpath):
        emotion = personpath + "\\" + person
        emotionlabel = personlabelpath + "\\" + person
        image_list = []
        currentemotion = ""
        if "DS_Store" in emotion:
            continue
        for personemotion in os.listdir(emotion):
            if "DS_Store" in personemotion:
                continue
            imagepath = emotion + "\\" + personemotion
            my_file = Path(emotionlabel)
            if not my_file.exists():
                continue
            all_files = os.listdir(emotionlabel)
            if len(all_files) > 0:
                imagelabelpath = emotionlabel + "\\" + all_files[0]
                my_file = Path(imagelabelpath)
                if my_file.exists():
                    f = open(imagelabelpath, 'r')
                    x = f.readlines()
                    currentLabel = int(x[0].strip()[0])
                    f.close()
                    image = cv2.imread(imagepath, 0)
                    resized_image = cv2.resize(image, (64, 64))
                    image_list.append(resized_image)

        if type == '0' and len(image_list)>0:
            featuresListOut, labelsListOut, hogFeatures = extractfeatures(image_list, currentLabel)
            for itr in range(len(featuresListOut)):
                featuresList.append(featuresListOut[itr])
                labelsList.append(labelsListOut[itr])
                hogFeaturesList.append(hogFeatures[itr])
        elif type == '1' and len(image_list)>0:
            testingLabel, featuresList, hogFeaturesList = extractfeatures(image_list, currentLabel)
            testingLabelsAll.append(testingLabel)

print(len(labelsList))

if type == '0': #training
    print("start training ... ")
    # print("start HOF")
    # trainusingSVM(featuresList, labelsList, "HOF_FacialExpressions_model.dat")
    print("start HOG")
    print(hogFeaturesList)
    trainusingSVM(hogFeaturesList, labelsList, "HOG_FacialExpressions_modelusingHarr.dat")
    print("Finish training successfully ...")
else: #testing
    print("start testing ... ")
    #test hog ang hof togrther
    score = getaccuracyformodels(testingLabelsAll)
    print("Final score (hog+hof) is: ")
    print(score * 100);
    test hof
    score = testSVM(featuresList, ,"HOF_FacialExpressions_model.dat")
    print("Final score (hof) is: ")
    print(score * 100)
    test hog
    score = testSVM(hogFeaturesList, , "HOG_FacialExpressions_model.dat")
    print("Final score (hog) is: ")
    print(score * 100)
    print("Finish testing successfully ...")

