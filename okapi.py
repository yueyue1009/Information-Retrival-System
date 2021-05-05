import numpy as np
from sklearn import preprocessing

def okapiDoc(TF, IDF, k ,b):
    docLen = TF.sum(1)
    avgLen = docLen.mean()
    
    TF = TF.tocoo()
    molecular = TF * (k + 1)
    denominator = k * (1 - b + b * docLen / avgLen)
    TF.data += np.array(denominator[TF.row]).reshape(len(TF.data),)
    TF.data = molecular.data / TF.data
    TF.data *= IDF[TF.col]
    TF = TF.tocsr()
    return TF

def okapiQuery(TF, IDF, k):
    IDF = IDF.tocoo()
    molecular = IDF * (k + 1)
    denominator = IDF.data + k
    IDF.data = molecular.data / IDF.data
    IDF = IDF.tocsr()
    return IDF