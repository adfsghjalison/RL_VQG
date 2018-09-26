import os, argparse, sys
import cv2, spacy, numpy as np
from keras.models import model_from_json
from keras.optimizers import SGD
from sklearn.externals import joblib
    
BATCH_SIZE = 32

fea_f = open("data/feature_qg","r")
qus_f = open(sys.argv[1],"r")
data_num = int(sys.argv[2])
plan = sys.argv[3]
word_embeddings = spacy.load('en')


if len(sys.argv)!=4:
	print 'python final_train.py [input] [data_num] [plan]'
	sys.exit()

def get_VQA_model(ans_size):
        sys.path.append(os.getcwd()+'/model/final')
        from VQA import VQA_MODEL
	vqa_model = VQA_MODEL(output_num = ans_size, path = None)
	vqa_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
	return vqa_model

def read_word(word_f = open('model/final/'+plan+'/ans.txt', 'r')):
	word = {}
	ans_size = 0
	for l in word_f:
		l = l.rstrip('\n')
		word[l] = ans_size
		ans_size = ans_size + 1
	ans_size = ans_size + 1
	return word, ans_size

def vectorize(word, ans_size):
	#X1 = []	
	X1 = np.zeros((data_num, 30, 300))
	X2 = []
	Y = []

	f_ind = -1
	#for k, limit in zip(qus_f, range(1000)):
	for cnt, k in enumerate(qus_f):
		k = k.strip('\n').split(' | ')
		[index, q, a] = [int(k[0]), k[1], k[2]]
		while index > f_ind:
			f = fea_f.readline().strip('\n')
			f_ind = f_ind + 1
		assert index == f_ind, "must equal"
		x1 = word_embeddings(unicode(q, 'utf-8'))
		for j in xrange(len(x1)):
			X1[cnt, j, :] = x1[j].vector
		x2 = [ float(i) for i in f.split() ]
		X2.append(x2)
		y = [0]*ans_size
		y[word[a]] = 1
		Y.append(y)
	#X1 = np.asarray(X1)
	X2 = np.asarray(X2)
	Y = np.asarray(Y)
	return X1, X2, Y

print "loading data..."
word, ans_size = read_word()
X1, X2, Y = vectorize(word, ans_size)

print X1.shape
print X2.shape
print Y.shape

print "loading model..."
vqa_model = get_VQA_model(ans_size)
print "load model successfully..."


print("\ntraining ...")

vqa_model.fit([X1, X2], Y, batch_size=BATCH_SIZE, nb_epoch=40, validation_split=0.05)
vqa_model.save_weights('model/final/'+plan+'/40.h5')

vqa_model.fit([X1, X2], Y, batch_size=BATCH_SIZE, nb_epoch=10, validation_split=0.05)
vqa_model.save_weights('model/final/'+plan+'/50.h5')

vqa_model.fit([X1, X2], Y, batch_size=BATCH_SIZE, nb_epoch=10, validation_split=0.05)
vqa_model.save_weights('model/final/'+plan+'/60.h5')

vqa_model.fit([X1, X2], Y, batch_size=BATCH_SIZE, nb_epoch=20, validation_split=0.05)
vqa_model.save_weights('model/final/'+plan+'/80.h5')

vqa_model.fit([X1, X2], Y, batch_size=BATCH_SIZE, nb_epoch=20, validation_split=0.05)
vqa_model.save_weights('model/final/'+plan+'/100.h5')

