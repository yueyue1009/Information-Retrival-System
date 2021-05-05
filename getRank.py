import numpy as np
from sklearn import preprocessing

def getRank(docVec, queryVec, qs, idx2File):
    
    docVec = preprocessing.normalize(docVec, norm='l2', axis=1)
    queryVec = preprocessing.normalize(queryVec, norm='l2', axis=1)

     # Accelerating multiplication
    nonzero = np.unique(queryVec.indices)
    docVec = docVec[:, nonzero]
    queryVec = queryVec[:, nonzero]
    similarity = docVec * (queryVec.transpose())

    
    prediction = []
    for e, Q in enumerate(qs):
        simList = []
        for i in range(len(idx2File)):
            simList.append((i, similarity[i, e]))
        simList.sort(key = lambda x: x[1], reverse = True)
        prediction.append(simList)
        
    return prediction