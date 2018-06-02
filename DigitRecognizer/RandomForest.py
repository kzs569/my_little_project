import csv
import numpy as np
import pandas
import theano
import theano.tensor as T
from sklearn.ensemble import RandomForestClassifier
from pandas import DataFrame,Series
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


def readCSVFile(file):
    rawData=[]
    trainFile = open(file,'rt',encoding='utf-8')
    reader = csv.reader(trainFile)

    for line in reader:
        rawData.append(line)
    rawData.pop(0)
    intData = np.array(rawData).astype(np.int32)
    trainFile.close()
    return intData

def loadTrainingData():
    intData = readCSVFile('train.csv')
    label = intData[:,0]
    data = intData[:,1:]
    data = np.where(data>0,1,0)
    return data,label

def loadTestData():
    intData = readCSVFile("test.csv")
    data = np.where(intData>0,1,0)
    return data

def loadTestResult():
    intData=readCSVFile("knn_benchmark.csv")
    data=np.mat(intData)
    return data[:,1]


def saveResult(result):
    myFile = open("result.csv", 'wb')
    myWriter = csv.writer(myFile)
    myWriter.writerow(['ImageId', 'Label'])
    ind = range(len(result))
    for i, val in zip(ind, result):
        line = []
        line.append(i + 1)
        for v in val:
            line.append(v)
        myWriter.writerow(line)

def handwritingClassTest():
    #load data and normalization
    trainData,trainLabel=loadTrainingData()
    testData=loadTestData()
    testLabel=loadTestResult()
    # train the rf classifier
    clf=RandomForestClassifier(n_estimators=1000,min_samples_split=5)
    clf=clf.fit(trainData,trainLabel)#train 20 objects
    m,n=np.shape(testData)
    errorCount=0
    resultList=[]
    for i in range(m):#test 5 objects
         classifierResult = clf.predict(testData[i].reshape(-1,1))
         resultList.append(classifierResult)
         print ("the classifier came back with: %d, the real answer is: %d" % (classifierResult, testLabel[i]))
         if (classifierResult != testLabel[i].reshape(-1,1)): errorCount += 1.0
    print( "\nthe total number of errors is: %d" % errorCount)
    print( "\nthe total error rate is: %f" % (errorCount/float(m)))
    saveResult(resultList)


handwritingClassTest()

def shared_dataset(data_xy,borrow=True):
    data_x,data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x,dtype = theano.config.floatX),borrow)
    shared_y = theano.shared(np.asarray(data_y,dtype = theano.config.floatX),borrow)
    return shared_x,T.cast(shared_y,'int32')

def load_data(path):
