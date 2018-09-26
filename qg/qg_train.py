from importlib import import_module
import sys, random, numpy as np
import tensorflow as tf

BATCH_SIZE = 32
data_block = 10000

modeltype = 'qg_model'
mod = import_module('model.'+modeltype+'.model_arc')
#fea_f = open('train/try_f')
#ques_f = open('train/try_q')
fea_f = open('train/feature', 'r')
ques_f = open('train/question', 'r')
typef = open('train/type', 'r')
cap_f = open('train/cap_index', 'r')

cap_len=16
ques_len=26
type_size=38

def read_dict(word_f = open('train/word_cap', 'r')):
    vocab = []
    for l in word_f:
        vocab.append(l.strip('\n'))
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
    return vocab, word_idx

def vectorize(vocab, word_idx):
    
    F = []
    T = []
    C = []
    Q = []
    
    cnt = 0
    for k in range(data_block):
        f = fea_f.readline().strip()
        ty = typef.readline().strip()
        cap = cap_f.readline().strip()
        l = ques_f.readline().strip()
        
        if f == "":
            break
        
        fs = [ float(i) for i in f.split() ]
        tys = [0]*type_size
        tys[int(ty)] = 1

        #x.extend(type)
        #X.append([x])
        F.append(fs)
        T.append(tys)
        
        
        caps = np.zeros((cap_len, len(vocab)+5))
        for i, j in enumerate(cap.split()):
            if j in vocab:
                caps[i][word_idx[j]] = 1
            else:
                caps[i][0] = 1    
        C.append(caps)

        y = np.zeros((ques_len, len(vocab)+5))
        for i, j in enumerate(l.split()):
            if j in vocab:
                y[i][word_idx[j]] = 1
            else:
                y[i][0] = 1    
        Q.append(y)

        cnt = cnt + 1
    Z = zip(F, T, C, Q)
    random.shuffle(Z)
    F, T, C, Q = zip(*Z)

    F = np.asarray(F)
    T = np.asarray(T)
    C = np.asarray(C)
    Q = np.asarray(Q)
    return F, T, C, Q

if len(sys.argv)<2:
    print 'python qg_train.py [EPOCH] ([load_weights])'
    sys.exit()
EPOCH = int(sys.argv[1])
NEXT = EPOCH if len(sys.argv)==2 else str(EPOCH+int(sys.argv[2]))

vocab, word_idx = read_dict()

"""
model = mod.qg_model(path='model/'+modeltype+'/'+sys.argv[2]+'.h5' if len(sys.argv)>2 else None, vocab_size=len(vocab)+5)
print "\n\nmodel bulit successfully\n"

for i in range(1):
    print "\n---------------", i*data_block, "-----------------\n"
    F, T, C, Q = vectorize(vocab, word_idx)
    model.fit([F, T, C], Q, batch_size=BATCH_SIZE, nb_epoch=EPOCH, validation_split=0.05)
    #model.save_weights('model/'+modeltype+'/temp'+str(i)+'.h5')

model.save_weights('model/'+modeltype+'/'+str(NEXT)+'.h5')
"""


model = mod.qg_model(path='model/'+modeltype+'/100.h5', vocab_size=len(vocab)+5)
print "\n\nmodel bulit successfully\n"

F, T, C, Q = vectorize(vocab, word_idx)

for i in [150, 200, 250, 300]:
    model.fit([F, T, C], Q, batch_size=BATCH_SIZE, nb_epoch=50, validation_split=0.05)
    model.save_weights('model/'+modeltype+'/'+str(i)+'.h5')

