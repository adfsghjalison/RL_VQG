from importlib import import_module
import sys, random
import numpy as np

data_limit = 5
cap_len=16
ques_len=26
type_size=38

typel = range(type_size)

modeltype = 'qg_model'
mod = import_module('model.'+modeltype+'.model_arc')

def vectorize(fea_f = open('train/feature'), cap_f = open('train/cap_index', 'r'), word_f = open('train/word_cap', 'r')):
    vocab = []
    for l in word_f:
        vocab.append(l.strip('\n'))
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
    
    X = []
    C = []
    T = []
    unique = []
    cnt = 0

    for f, c in zip(fea_f, cap_f):
        if f in unique:
            continue
        unique.append(f)
        f = f.strip('\n')
        c = c.strip('\n')
        x1 = [ float(i) for i in f.split() ]
        X.append(x1)
        
        caps = np.zeros((cap_len, len(vocab)+5))
        for i, j in enumerate(c.split()):
            if j in vocab:
                caps[i][word_idx[j]] = 1
            else:
                caps[i][0] = 1    
        C.append(caps)
        
        random.shuffle(typel)
        q = [0]*type_size; q[typel[0]] = 1;
        T.append(q)
        q = [0]*type_size; q[typel[1]] = 1;
        T.append(q)
        
        cnt = cnt + 1
        if cnt >= data_limit:
            break
        
    X = np.asarray(X)
    T = np.asarray(T)
    C = np.asarray(C)
    
    return X, T, C, len(vocab)+5, vocab

if len(sys.argv)<2:
    print 'python qg_test.py [load_weights]'
    sys.exit()

X, T, C, vocab_size, vocab = vectorize()
model = mod.qg_model(path='model/'+modeltype+'/'+sys.argv[1]+'.h5', vocab_size=vocab_size)
Y = model.predict([X, T, C])

for y in Y:
    for w in y:
        word = np.argmax(w)-1
        print vocab[word] if word>=0 else '[und]',
        if word>=0 and vocab[word]=='?':
            print ''
            break

