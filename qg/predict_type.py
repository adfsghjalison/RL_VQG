from importlib import import_module
import sys, random, numpy as np

BATCH_SIZE = 32
data_block = 10000

modeltype = 'type'
modeltype2 = '2_1_lstm'
mod = import_module('model.'+modeltype+'.model_arc')
mod2 = import_module('model.'+modeltype2+'.model_arc')
#fea_f = open('train/try_f')
fea_f = open('val/feature', 'r')
typef = open('data/type')
word_f = open('train/word', 'r')
data_limit = 200
type_size = 38
fre_cnt = 3

type_list = []

def read_type():
    for l in typef:
        type_list.append(l.strip('\n'))
    type_list.append("others")

def vectorize():
    
    X = []
    unique = []
    
    cnt = 0
    for k in range(data_block):
        f = fea_f.readline()
        if f in unique:
            continue
        unique.append(f)
        x = [ float(i) for i in f.strip('\n').split() ]
        X.append(x)
        cnt = cnt + 1
        if cnt >= data_limit:
            break
    X = np.asarray(X)
    return X

def vectorize2(X1, T):
    
    max_ques = int(word_f.readline().strip('\n'))
    vocab = []
    for l in word_f:
        vocab.append(l.strip('\n'))
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))

    X = []
    for x, t in zip(X1, T):
        x = x.tolist()
        for i in range(fre_cnt):
            x2 = x[:]
            q = [0]*type_size
            q[t[i]] = 1
            x2.extend(q)
            X.append([x2])
    X = np.asarray(X)
    return X, len(vocab)+5, max_ques, vocab


if len(sys.argv)<3:
    print 'python predict_type.py [load_weights_1] [load_weights_2]'
    sys.exit()

read_type()
X = vectorize()

model = mod.qg_model(path='model/'+modeltype+'/'+sys.argv[1]+'.h5', type_size=type_size)
#print "\n\nmodel bulit successfully\n"

Y = model.predict(X)
T = []

for y in Y:
    ty = np.argsort(y)[::-1]
    T.append(ty[0:fre_cnt])
    #for i in range(fre_cnt):
    #    print ty[i], type_list[ty[i]]
    #print ""

X, vocab_size, max_ques, vocab = vectorize2(X, T)
model = mod2.qg_model(path='model/'+modeltype2+'/'+sys.argv[2]+'.h5', vocab_size=vocab_size, ques_len=max_ques, type_size=type_size)
Y = model.predict(X)

cnt = 0
T = [j for i in T for j in i]
for t, y in zip(T, Y):
    print ty[t], type_list[ty[t]], ' | ', 
    for w in y:
        word = np.argmax(w)-1
        print vocab[word] if word>=0 else '[und]',
        if word>=0 and vocab[word]=='?':
            print ''
            break
    cnt = cnt+1
    if cnt%3==0:
        print ''
