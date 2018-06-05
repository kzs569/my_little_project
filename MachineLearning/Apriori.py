

class Apriori:
    def __init__(self):
        self.dataSet = [[1,3,4],[2,3,5],[1,2,3,5],[2,5]]

    def createC1(self):
        C1 = []
        for transaction in self.dataSet:
            for item in transaction:
                if not [item] in C1:
                    C1.append([item])
        C1.sort()
        map(frozenset,C1)

    def scanD(self, Ck,miniSupport = 0.5):
        D = map(set,self.dataSet)
        ssCnt = {}
        for tid in D:
            for can in Ck:
                if can.issubset(tid):
                    if not ssCnt.has_key(can):
                        ssCnt[can] = 1
                    else:
                        ssCnt[can] += 1
        numItems = float(len(D))
        retList = []
        supportData = {}
        for key in ssCnt:
            support = ssCnt[key]/numItems
            if support >= miniSupport:
                retList.insert(0,key)
            supportData[key] = support
        return retList,supportData

    def aprioriGen(self,Lk,k):
        retList = []
        lenLK = len(Lk)
        for i in range(lenLK):
            for j in range(i+1,lenLK):
                L1 = list(Lk[i])[:k-2]
                L2 = list(Lk[j])[:k-2]
                L1.sort()
                L2.sort()
                if L1 == L2:
                    retList.append(Lk[i]|Lk[j])
        return retList

    def apriori(self,minSupport = 0.5):
        C1 =  self.createC1()
        L1 , supportData = self.scanD(C1, miniSupport=0.5)
        L = [L1]
        k = 2
        while(len(L[k-2]) > 0):
            Ck  = self.aprioriGen(L[k-2],k)
            Lk,supK = self.scanD(Ck,minSupport = 0.5)
            supportData.update(supK)
            L.append(Lk)
            k += 1
        return L,supportData