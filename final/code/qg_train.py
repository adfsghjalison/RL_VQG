import os, sys, random, numpy as np

BATCH_SIZE = 32
#data_block = 10000

sys.path.append(os.getcwd()+'/model/qg')
from model_arc2 import qg_model

fea_f = open('data/feature_qg', 'r')
cap_f = open('data/cap_uni', 'r')
ques_f = open(sys.argv[1], 'r')

cap_len=16
ques_len=26
type_size=38


if len(sys.argv)<5:
    print 'python qg_train.py [input] [EPOCH] [load_weights] [write_weights]'
    sys.exit()


def read_dict(word_f = open('model/qg/word_cap', 'r')):
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
 

    f_ind = -1
    for k in ques_f:
        k = k.split(' | ')
        [index, ty, l] = [int(k[0]), int(k[1]), k[2]]
        while index > f_ind:
            f = fea_f.readline().strip('\n')
            cap = cap_f.readline().strip('\n')
            f_ind = f_ind + 1
        assert index == f_ind, "must equal"
        
        fs = [ float(i) for i in f.split() ]
        tys = [0]*type_size
        tys[ty] = 1

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
        for i, j in enumerate(l.strip('\n').split()):
            if j in vocab:
                y[i][word_idx[j]] = 1
            else:
                y[i][0] = 1    
        Q.append(y)

    Z = zip(F, T, C, Q)
    random.shuffle(Z)
    F, T, C, Q = zip(*Z)

    F = np.asarray(F)
    T = np.asarray(T)
    C = np.asarray(C)
    Q = np.asarray(Q)
    return F, T, C, Q


EPOCH = int(sys.argv[2])

vocab, word_idx = read_dict()

model = qg_model(path='model/qg/'+sys.argv[3]+'.h5', vocab_size=len(vocab)+5)
print "\n\nmodel bulit successfully\n"

F, T, C, Q = vectorize(vocab, word_idx)
model.fit([F, T, C], Q, batch_size=BATCH_SIZE, nb_epoch=EPOCH, validation_split=0.05)

model.save_weights('model/qg/'+sys.argv[4]+'.h5')

