import os
import argparse
import numpy as np
import scipy

from getTFIDF import parseDoc, parseQuery
from preprocess import preprocess
from okapi import okapiDoc, okapiQuery
from query import Query
from getRank import getRank

parser = argparse.ArgumentParser(description='vsmodel')
parser.add_argument('-r', action='store_true', dest="feedback")
parser.add_argument('-i', type=str, dest="queryFile")
parser.add_argument('-o', type=str, dest="rankList")
parser.add_argument('-m', type=str, dest="modelPath")
parser.add_argument('-d', type=str, dest="unuse")
args = parser.parse_args()

# param
alpha = 1
beta = 0.8
gamma = 0.2
relCount = 10
nrelCount = 50 
iters = 2
#okapi
k = 1.5
k3 = 2
b = 0.6

voc2Idx, idx2File, ngram2Idx = preprocess(args.modelPath)

docTF, docIDF = parseDoc(args.modelPath, len(ngram2Idx), len(idx2File))
Qs, queryTF, queryIDF = parseQuery(args.queryFile, voc2Idx, ngram2Idx) 

docV = okapiDoc(docTF, docIDF, k, b)
queryV = okapiQuery(queryTF, queryIDF, k)

print("Start Predicting!")
preds = getRank(docV, queryV, Qs, idx2File)

if args.feedback:
    for i in range(iters):
        print(str(i + 1) + "th feedback\n")
        # get relevent document and irrelevent document of each queries
        relPreds = [pred[:relCount] for pred in preds]
        nrelPreds = [pred[-nrelCount:] for pred in preds]

        for j in range(len(relPreds)):
            relV = scipy.sparse.csr_matrix(docV[[ele[0] for ele in relPreds[j]]].mean(axis=0))
            nrelV = scipy.sparse.csr_matrix(docV[[ele[0] for ele in nrelPreds[j]]].mean(axis=0))
            # print(queryV.shape)
            # print(queryV[j].shape)
            # print((nrelV.shape), flush=True)
            # print((relV.shape), flush=True)
            # print(type(queryV))
            # print(type(queryV[j]))
            # print(type(nrelV))
            # print(type(relV))
            queryV[j] = (alpha * queryV[j] + beta * relV - gamma * nrelV).tocsr()

        preds = getRank(docV, queryV, Qs, idx2File)
    print("Complete feedback")

answer = []
for A in preds:
    answer.append([idx2File[a[0]] for a in A[:100]])

with open(args.rankList, 'w') as f:
    print("query_id,retrieved_docs", file=f)
    for e, Q in enumerate(Qs):
        print('{},{}'.format(Q.num, " ".join(answer[e])), file=f)
