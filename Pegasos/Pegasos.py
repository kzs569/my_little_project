import os
import numpy as np
import math
import pandas as pd


class Pegasos:
    def __init__(self):
        self.G_WEIGHT = np.zeros((10, 32 * 32), dtype=float)
        self.train = pd.read_csv('train.csv')
        self.test = pd.read_csv('test.csv')

    def train_one_sample(self, data, num, sampleNum):
        for modelNum in range(10):
            label = -1
            if num == modelNum:
                label = 1
            self.train_one_model(data, label, sampleNum, modelNum)

    def train_one_model(self, data, label, sampleNum, modelNum):
        pvalue = self.predict(self.G_WEIGHT[modelNum], data)
        # the hinge loss
        if pvalue * label >= 1: return

        # update model
        lambd = 0.5
        new_weight = self.G_WEIGHT[modelNum] * (1 - 1.0 / sampleNum) + (1.0 / (lambd * sampleNum)) * label * data

        # projection
        norm2 = np.linalg.norm(new_weight)
        if norm2 > (1 / math.sqrt(lambd)):
            self.G_WEIGHT[modelNum] = new_weight / (norm2 * math.sqrt(lambd))
        else:
            self.G_WEIGHT[modelNum] = new_weight

    def predict(self, model, data):
        return np.inner(model, data)



    def run(self):
        for _ in range(50):
            for i in range(len(self.train)):
                data = self.train.iloc[i][:-1]
                label = self.train.iloc[i][-1]
                self.train_one_sample(data, label, i + 1)  # data,label,Num

        right = 0
        wrong = 0
        can_not_classify = 0
        total = 0
        for i in range(len(self.test)):
            total += 1
            data = self.test.iloc[i][:-1]
            label = self.test.iloc[i][-1]
            classify_failed = True
            for i in range(10):
                pvalue = self.predict(self.G_WEIGHT[i], data)
                if pvalue > 0:
                    classify_failed = False
                    if i == label:
                        right += 1
                    else:
                        wrong += 1
            if classify_failed:
                can_not_classify += 1
        print("right=", right)
        print("wrong=", wrong)
        print("can_not_classify=", can_not_classify)
        print("total=", total)



