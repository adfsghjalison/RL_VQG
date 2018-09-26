from importlib import import_module
import os, sys, random, numpy as np

BATCH_SIZE = 32
data_block = 30000

modeltype = 'type'
mod = import_module('model.'+modeltype+'.model_arc')
#fea_f = open('train/try_f')
fea_f = open('train/feature', 'r')
typef = open('train/type', 'r')
type_size = 38

def vectorize():
    
    X = []
    Y = []
    
    cnt = 0
    for k in range(data_block):
        f = fea_f.readline()
        ty = typef.readline().strip('\n')

        if f == "":
            break
        
        x = [ float(i) for i in f.strip('\n').split() ]
        type = [0]*type_size
        type[int(ty)] = 1
        X.append(x)
        Y.append(type)
        cnt = cnt + 1
    Z = zip(X, Y)
    random.shuffle(Z)
    X, Y = zip(*Z)

    X = np.asarray(X)
    Y = np.asarray(Y)
    return X, Y, cnt%data_block

if len(sys.argv)<2:
    print 'python qg_train.py [EPOCH] ([load_weights])'
    sys.exit()
EPOCH = int(sys.argv[1])
NEXT = EPOCH if len(sys.argv)==2 else str(EPOCH+int(sys.argv[2]))

model = mod.qg_model(path='model/'+modeltype+'/'+sys.argv[2]+'.h5' if len(sys.argv)>2 else None, type_size=type_size)
print "\n\nmodel bulit successfully\n"

cnt = 1

while(True):
    X, Y, en = vectorize()
    if en:
        print "\n---------------", (cnt-1)*data_block+en , "----------------\n"
    else:
        print "\n---------------", cnt*data_block, "-----------------\n"
    model.fit(X, Y, batch_size=BATCH_SIZE, nb_epoch=EPOCH, validation_split=0.05)
    #model.save_weights('model/'+modeltype+'/'+str(NEXT)+'-'+str(cnt)+'.h5')
    cnt = cnt+1
    if en:
        break
#os.system('mv model/'+modeltype+'/'+str(NEXT)+'-'+str(cnt-1)+'.h5 model/'+modeltype+'/'+str(NEXT)+'.h5')
model.save_weights('model/'+modeltype+'/'+str(NEXT)+'.h5')

