def preprocess(path):
    print("Start preprocess!\n")

    voc2Idx = dict()
    for i, line in enumerate(open(path + 'vocab.all', 'r')):
        voc2Idx[line.strip()] = i 


    idx2File = list()
    for i, line in enumerate(open(path + 'file-list', 'r')):
        idx2File.append(line.split('/')[-1].strip().lower())

    ngram2Idx = dict()
    f = open(path + 'inverted-file', 'r')
    for i, line in enumerate(f):
        first, second, n = line.split(' ')
        ngram2Idx[(int(first), int(second))] = i
        for j in range(int(n)):
            f.readline()
            
    print('Complete prepocess!\n')
    return voc2Idx, idx2File, ngram2Idx