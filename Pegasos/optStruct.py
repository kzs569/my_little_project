import pandas as pd
import numpy as np
import random
import time

class optStruct:

    def __init__(self,dataMatIn,classLabels,C,toler,kTup):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = dataMatIn.shape[0]
        self.alphas = np.mat(np.zeros((self.m,1)))
        self.b = 0.0
        self.eCache = np.mat(np.zeros((self.m,2)))
        #第一列是有效与否的标记
        self.K = np.mat(np.zeros((self.m,self.m)))
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X, self.X[i,:], kTup)    #软cache

    def selectJrand(self, i, m):
        j = i  # we want to select any J not equal to i
        while (j == i):
            j = int(random.uniform(0, m))
        return j



    def calcEk(self, oS, k):
        fXk = float(np.multiply(oS.alphas, oS.labelMat).T * oS.K[:, k]) + oS.b
        Ek = fXk - float(oS.labelMat[k])
        return Ek

    def updateEk(self,oS,k):
        Ek = self.calcEk(oS,k)
        oS.eCache[k] = [1,Ek]

    def innerL(self,i, oS):
        Ei = self.calcEk(oS, i)
        if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or (
                (oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
            j, Ej = self.selectJ(i, oS, Ei)  # this has been changed from selectJrand
            alphaIold = oS.alphas[i].copy()
            alphaJold = oS.alphas[j].copy()
            if (oS.labelMat[i] != oS.labelMat[j]):
                L = max(0, oS.alphas[j] - oS.alphas[i])
                H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
            else:
                L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
                H = min(oS.C, oS.alphas[j] + oS.alphas[i])
            if L == H:
                print("L==H")
                return 0
            eta = 2.0 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]  # changed for kernel
            if eta >= 0:
                print("eta>=0")
                return 0
            oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
            oS.alphas[j] = self.clipAlpha(oS.alphas[j], H, L)
            self.updateEk(oS, j)  # added this for the Ecache
            if (abs(oS.alphas[j] - alphaJold) < 0.00001):
                print("j not moving enough")
                return 0
            oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (
                    alphaJold - oS.alphas[j])  # update i by the same amount as j
            self.updateEk(oS, i)  # added this for the Ecache                    #the update is in the oppostie direction
            b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, i] - oS.labelMat[j] * (
                    oS.alphas[j] - alphaJold) * oS.K[i, j]
            b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, j] - oS.labelMat[j] * (
                    oS.alphas[j] - alphaJold) * oS.K[j, j]
            if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
                oS.b = b1
            elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
                oS.b = b2
            else:
                oS.b = (b1 + b2) / 2.0
            return 1
        else:
            return 0

    def clipAlpha(self,aj, H, L):
        if aj > H:
            aj = H
        if L > aj:
            aj = L
        return aj

    def updateEk(self,oS, k):  # after any alpha has changed update the new value in the cache
        Ek = self.calcEk(oS, k)
        oS.eCache[k] = [1, Ek]

    def selectJ(self, i, oS, Ei):  # this is the second choice -heurstic, and calcs Ej
        maxK = -1;
        maxDeltaE = 0;
        Ej = 0
        oS.eCache[i] = [1, Ei]  # set valid #choose the alpha that gives the maximum delta E
        validEcacheList = np.nonzero(oS.eCache[:, 0].A)[0]
        if (len(validEcacheList)) > 1:
            for k in validEcacheList:  # loop through valid Ecache values and find the one that maximizes delta E
                if k == i: continue  # don't calc for i, waste of time
                Ek = self.calcEk(oS, k)
                deltaE = abs(Ei - Ek)
                if (deltaE > maxDeltaE):
                    maxK = k
                    maxDeltaE = deltaE
                    Ej = Ek
            return maxK, Ej
        else:  # in this case (first time around) we don't have any valid eCache values
            j = self.selectJrand(i, oS.m)
            Ej = self.calcEk(oS, j)
        return j, Ej

    def selectJrand(self,i, m):
        """
        随机从0到m挑选一个不等于i的数
        :param i:
        :param m:
        :return:
        """
        j = i  # 排除i
        while (j == i):
            j = int(random.uniform(0, m))
        return j

    def clipAlpha(self,aj, H, L):
        """
        将aj剪裁到L(ow)和H(igh)之间
        :param aj:
        :param H:
        :param L:
        :return:
        """
        if aj > H:
            aj = H
        if L > aj:
            aj = L
        return aj

def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):  # full Platt SMO
    oS = optStruct(dataMatIn, classLabels, C, toler, kTup)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:  # go over all
            for i in range(oS.m):
                alphaPairsChanged += oS.innerL(i, oS)
                print("fullSet, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1
        else:  # go over non-bound (railed) alphas
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += oS.innerL(i, oS)
                print("non-bound, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1
        if entireSet:
            entireSet = False  # toggle entire set loop
        elif (alphaPairsChanged == 0):
            entireSet = True
        print("iteration number: %d" % iter)
    return oS.b, oS.alphas

def kernelTrans(X, A, kTup):
    """
    核函数
    :param X: 全部向量
    :param A: 某个向量
    :param kTup: 核函数的名称
    :return: rasin NameError
    """
    m, n = np.shape(X)
    K = np.zeros((m, 1))
    if kTup[0] == 'lin':
        K = X * A.T
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = np.dot(deltaRow,deltaRow)
        K = np.exp(K / (-1 * kTup[1] ** 2))
    else:
        raise NameError('Houston We Have a Problem -- That Kernel is not recognized')

    return K

if __name__ == '__main__':
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')

    kTup = ('rbf', 10)

    labelArr = train.iloc[:,-1]
    labelArr = labelArr.values.reshape(train.shape[0],1)
    dataArr = train.iloc[:,:-1]
    dataArr = dataArr.values.reshape(train.shape[0],train.shape[1] - 1)
    start = time.time()
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, kTup)
    end = time.time()
    print("smo time:", end - start)
    datMat = dataArr
    labelMat = labelArr
    svInd = np.nonzero(alphas.A > 0)[0]
    print(alphas)
    sVs = datMat[svInd]
    labelSV = labelMat[svInd]
    print("there are %d Support Vectors" % np.shape(sVs)[0])
    m, n = datMat.shape
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], kTup)
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]): errorCount += 1
    print("the training error rate is: %f" % (float(errorCount) / m))

    labelArr = test.iloc[:,-1]
    labelArr = labelArr.values.reshape(test.shape[0],1)
    dataArr = test.iloc[:,:-1]
    dataArr = dataArr.values.reshape(test.shape[0],test.shape[1] - 1)

    errorCount = 0
    datMat = dataArr
    labelMat = labelArr
    m, n = datMat.shape
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], kTup)
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]):
            errorCount += 1
    print("the test error rate is: %f" % (float(errorCount) / m))


