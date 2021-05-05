import numpy as np
import xml.etree.ElementTree as ET
from scipy.sparse import csr_matrix
from tqdm import tqdm
from query import Query




def parseDoc(path, ngramLen, fileLen):
    print('Start parseDoc!\n')
    count = []
    r = []
    c = []
    IDF = []


    f = open(path + '/inverted-file', 'r')
    for i, line in enumerate(tqdm(f, total=ngramLen)):
        trash1, trash2, n = line.split(' ')
        for j in range(int(n)):
            file, cnt = f.readline().strip().split(' ')
            r.append(int(file))
            c.append(i)
            count.append(int(cnt))

        IDF.append(np.log((fileLen - int(n) + 0.5) / (int(n) + 0.5)))
    
    DF = csr_matrix((count, (r, c)), shape=(fileLen, ngramLen), dtype='float')
    print('Complete parseDoc!\n')
    return DF, np.array(IDF)
    


def cleanQuery(q):
    return q.replace('相關', '').replace('文件', '').replace('查詢', '').replace('內容', '').replace('應', '').replace('包括', '').replace('主要', '').replace('說明', '')

def calDFIDF(Q, voc2Idx, ngram2Idx):
    weight1 = 5
    weight2 = 9
    weight3 = 3
    weight4 = 3
    weight5 = 6
    weight6 = 6

    title = Q.title
    query = Q.query
    concept = Q.concepts
    narrative = Q.narrative

    DF = np.zeros(len(ngram2Idx))
    IDF = np.zeros(len(ngram2Idx))

    for i in title:
        if i in voc2Idx:
            bigram = (voc2Idx[i], -1)
            if bigram in ngram2Idx:
                DF[ngram2Idx[bigram]] += weight1
                IDF[ngram2Idx[bigram]] += 1

    for i in range(len(title) - 1):
        if (title[i] in voc2Idx) and (title[i + 1] in voc2Idx):
            bigram = (voc2Idx[title[i]], voc2Idx[title[i + 1]])

            if bigram in ngram2Idx:
                DF[ngram2Idx[bigram]] += weight2
                IDF[ngram2Idx[bigram]] += 1

    for i in query:
        if i in voc2Idx:
            bigram = (voc2Idx[i], -1)

            if bigram in ngram2Idx:
                DF[ngram2Idx[bigram]] += weight3
                IDF[ngram2Idx[bigram]] += 1

    for i in range(len(query) - 1):
        if (query[i] in voc2Idx) and (query[i + 1] in voc2Idx):
            bigram = (voc2Idx[query[i]], voc2Idx[query[i + 1]])

            if bigram in ngram2Idx:
                DF[ngram2Idx[bigram]] += weight4
                IDF[ngram2Idx[bigram]] += 1

    for i in concept:
        if i in voc2Idx:
            bigram = (voc2Idx[i], -1)
            if bigram in ngram2Idx:
                DF[ngram2Idx[bigram]] += weight3
                IDF[ngram2Idx[bigram]] += 1

    
    for i in range(len(concept) - 1):
        if (concept[i] in voc2Idx) and (concept[i + 1] in voc2Idx):
            bigram = (voc2Idx[concept[i]], voc2Idx[concept[i + 1]])

            if bigram in ngram2Idx:
                DF[ngram2Idx[bigram]] += weight4
                IDF[ngram2Idx[bigram]] += 1

    return DF, IDF

def parseQuery(path, voc2Idx, ngram2Idx):
    print('Start parseQuery!\n')
    f = open(path, "r", encoding='utf-8')
    root = ET.parse(f).getroot()
    topics = root.findall("topic")

    DF = list()
    IDF = list()
    Qs = list()

    for topic in topics:
        num = topic.find("number").text.strip().split("ZH")[1]
        title = topic.find("title").text.strip()
        question = topic.find("question").text.strip()
        narrative = topic.find("narrative").text.split('。')[0]
        concepts = topic.find("concepts").text.strip()
        query = " ".join([question, concepts])
        query =  cleanQuery(query)

        Q = Query(num, title, question, narrative, concepts, query)
        Qs.append(Q)

        df, idf = calDFIDF(Q, voc2Idx, ngram2Idx)
        DF.append(df)
        IDF.append(idf)

    IDF = np.stack(IDF)
    IDF = csr_matrix(IDF)
    DF = np.stack(DF)
    DF = csr_matrix(DF)

    print('Complete parseQuery!\n')
    return Qs, DF, IDF
