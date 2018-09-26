from importlib import import_module
import sys, random, os, numpy as np

BATCH_SIZE = 32
modeltype = 'type'
modeltype2 = 'qg'



sys.path.append(os.getcwd()+'/model/qg')
sys.path.append(os.getcwd()+'/model/type')
from model_arc1 import type_model
from model_arc2 import qg_model

fea_f = open('data/feature_qg', 'r')
cap_f = open('data/cap_uni', 'r')
word_f = open('model/qg/word_cap', 'r')
output = open(sys.argv[3], 'w')

data_start = int(sys.argv[4])
data_limit = int(sys.argv[5])
fre_cnt = int(sys.argv[6])

type_size = 38
cap_len=16
ques_len=26

if len(sys.argv)<7:
    print 'python qg_gen.py [load_type] [load_qg] [output-f] [start] [num] [rank]'
    sys.exit()

def vectorize():
    
    vocab = []
    for l in word_f:
        vocab.append(l.strip('\n'))
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
        
    X = []
    C = []
    
    for i in range(data_start):
        fea_f.readline()
        cap_f.readline()

    for k in range(data_limit):
        f = fea_f.readline()
        x = [ float(i) for i in f.strip('\n').split() ]
        X.append(x)
            
        cap = cap_f.readline().strip()
        caps = np.zeros((cap_len, len(vocab)+5))
        for i, j in enumerate(cap.split()):
            if j in vocab:
                caps[i][word_idx[j]] = 1
            else:
                caps[i][0] = 1    
        for i in range(fre_cnt):
            C.append(caps)
    
    X = np.asarray(X)
    C = np.asarray(C)

    return X, C, len(vocab)+5, vocab


def vectorize2(X1, T):

    F = []
    T2 = []

    for x, t in zip(X1, T):
        x = x.tolist()
        for i in range(fre_cnt):
            q = [0]*type_size
            q[t[i]] = 1
            F.append(x)
            T2.append(q)
	
    F = np.asarray(F)
    T2 = np.asarray(T2)
    return F, T2


X, C, vocab_size, vocab = vectorize()

model = type_model(path='model/'+modeltype+'/'+sys.argv[1]+'.h5', type_size=type_size)
print "\n\ntype_mode bulit successfully\n"

Y = model.predict(X)
T = []

for y in Y:
    ty = np.argsort(y)[::-1]
    T.append(ty[0:fre_cnt])

F, T2 = vectorize2(X, T)
model = qg_model(path='model/'+modeltype2+'/'+sys.argv[2]+'.h5', vocab_size=vocab_size)
print "\n\nqg_model bulit successfully\n"

Y = model.predict([F, T2, C])
T = [j for i in T for j in i]

for t, y in zip(T, Y):
    print >>output, t, '|', 	
    en = 0
    for w in y:
        word = np.argmax(w)-1
        print >>output, vocab[word] if word>=0 else '[und]',
        if word>=0 and vocab[word]=='?':
            print >>output, ''
            en = 1
            break
    if en==0:
        print >>output, ''

output.close()


