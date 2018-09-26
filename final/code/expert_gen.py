import os, argparse, sys, math
import cv2, spacy, numpy as np
from keras.models import model_from_json
from keras.optimizers import SGD
from sklearn.externals import joblib
    
# File paths for the model, all of these except the CNN Weights are 
# provided in the repo, See the models/CNN/README.md to download VGG weights
fea_f = open("data/feature_expert","r")
qus_f = open(sys.argv[1],"r")
VQA_weights_file_name   = 'model/expert/models/VQA/VQA_MODEL_WEIGHTS.hdf5'
label_encoder_file_name = 'model/expert/models/VQA/FULL_labelencoder_trainval.pkl'
UNIT = 0.3

data_start = int(sys.argv[4])
data_limit = int(sys.argv[5])
rank_cnt = int(sys.argv[6])
word_embeddings = spacy.load('en')


if len(sys.argv)<7:
	print 'python expert_gen.py [input-q] [output-a] [output-up] [start] [num] [rank]'
	sys.exit()

def get_VQA_model(VQA_weights_file_name):
	''' Given the VQA model and its weights, compiles and returns the model '''
        sys.path.append(os.getcwd()+'/model/expert/models/VQA')
	from VQA import VQA_MODEL
	vqa_model = VQA_MODEL()
	vqa_model.load_weights(VQA_weights_file_name)
	vqa_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
	return vqa_model


X1 = np.zeros((data_limit*rank_cnt, 30, 300))
X2 = []

vqa_model = get_VQA_model(VQA_weights_file_name)
labelencoder = joblib.load(label_encoder_file_name)
print "load model successfully..."

for i in range(data_start):
	fea_f.readline()

index = []
cnt = 0

for k in range(data_limit):
	f = fea_f.readline()
	x2 = [ float(i) for i in f.strip('\n').split() ]
	for r in range(rank_cnt):
		x1 = word_embeddings(unicode(qus_f.readline().split(' | ')[1], 'utf-8'))
		for j in xrange(len(x1)):
			X1[cnt, j, :] = x1[j].vector
		cnt = cnt + 1
		X2.append(x2)
		index.append(data_start+k)

X2 = np.asarray(X2)

print X1.shape
print X2.shape

print("\n\n\nPredicting result ...") 
Y = vqa_model.predict([X1, X2])

qus_f.close()
qus_f = open(sys.argv[1],"r")
f2 = open(sys.argv[2], 'w')
f3 = open(sys.argv[3], 'w')

for i, y in zip(range(data_limit*rank_cnt), Y):
	label = np.argmax(y)
	#for label in reversed(y_sort_index[-5:]):
	l = qus_f.readline().rstrip('\n').split(' | ')
	print >> f2, index[i], '|', l[1], '|', labelencoder.inverse_transform(label)
	
	times = 0
	if(math.isnan(y[label])==False):
		times = int(y[label]/UNIT)
	for j in range(times):
		print >> f3, index[i], '|', l[0], '|', l[1]

f2.close()
f3.close()


