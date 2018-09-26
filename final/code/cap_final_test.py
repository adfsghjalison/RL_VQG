import os, argparse, sys
import cv2, spacy, numpy as np
#from keras.models import model_from_json
#from keras.optimizers import SGD
from sklearn.externals import joblib
from fuzzywuzzy import fuzz
    
BATCH_SIZE = 32

fea_f = open("../val/feature_val","r")
cap_f = open("../val/val_cap_index", 'r')
qus_f = open("../val/question", "r")
cho = open("../val/choices.test", "r")
ans = open("../val/answer.test", "r")
plan = sys.argv[1]
word_embeddings = spacy.load('en')
data_num = 72801
#data_num = 10

cap_len = 20
ques_len = 30

if len(sys.argv)<3:
	print 'python final_test.py [plan] [load_weights]'
	sys.exit()

def get_VQA_model(ans_size):
        sys.path.append(os.getcwd()+'/model/final')
	from cap_VQA import VQA_MODEL
	vqa_model = VQA_MODEL(output_num = ans_size, path ='model/final/'+plan+'/'+sys.argv[2]+'.h5')
	vqa_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
	return vqa_model
	
def read_word(word_f = open('model/final/'+plan+'/ans.txt', 'r')):
	word = []
	ans_size = 0
	for l in word_f:
		l = l.rstrip('\n')
		word.append(l)
		ans_size = ans_size + 1
	word.append('[unk]')
	ans_size = ans_size + 1
	return word, ans_size

def vectorize():
        F = []
        C = np.zeros((data_num, cap_len, 300))
	Q = np.zeros((data_num, ques_len, 300))

	for index, f, cap, q in zip(range(data_num), fea_f, cap_f, qus_f):
            cap = cap[:cap_len] if len(cap)>cap_len else cap
            caps = word_embeddings(unicode(cap, 'utf-8'))
            for j in xrange(len(caps)):
                C[index, j, :] = caps[j].vector

            q = q[:ques_len] if len(q)>ques_len else q
            qs = word_embeddings(unicode(q, 'utf-8'))
            for j in xrange(len(qs)):
                Q[index, j, :] = qs[j].vector
            
            fs = [ float(i) for i in f.split() ]
            F.append(fs)
            if index%1000==0:
		print index
	F = np.asarray(F)
	return F, C, Q

print 'Load testing data ...'
word, ans_size = read_word()
F, C, Q = vectorize()
print F.shape
print C.shape
print Q.shape

print "loading model..."
vqa_model = get_VQA_model(ans_size)
print "load model successfully..."

print("\npredicting ...")
Y = vqa_model.predict([F, C, Q])

sim_out = open('output/'+plan+'/out-'+sys.argv[2], 'w')
out_f = open('output/'+plan+'/out_cls-'+sys.argv[2], 'w')
acc_f = open('output/'+plan+'/acc-'+sys.argv[2], 'w')
cho.readline()
ans.readline()

cls = 0
cls_80 = 0
eq = 0
total = 0

#for i in range(72081):
for y in Y:
	label = np.argmax(y)
	output = word[label]
	sp = cho.readline().rstrip('\n').split('\t')[2].split('(')[1:]	

	choices = [c[2:-2] if c[2]!='"' else c[3:-3] for c in sp]
	max_i = 0
	max_p = 0
	for i in range(5):
		cm = fuzz.token_sort_ratio(output, choices[i])
		if cm > max_p:
			max_p = cm
			max_i = i
	answer = ans.readline().rstrip('\n').split('\t')[2][1:-1]
	print >> sim_out, output
	print >> out_f, '{0:<20}{1:<5}{2:<20}{3:<10}'.format(output, '->', choices[max_i], '('+str(max_p)+'%)'),
	if choices[max_i] == answer:
		cls = cls + 1
		if max_p >= 80:
			cls_80 = cls_80 + 1
	if output == answer:
		eq = eq + 1
		print >> out_f, 'V',
	print >> out_f, ''
	total = total + 1
print >> acc_f, float(cls)/total*100, '%'
print >> acc_f, float(cls_80)/total*100, '%'
print >> acc_f, float(eq)/total*100, '%'                                                                
print >> acc_f, 'closes                         : ', cls, '/', total
print >> acc_f, 'closest && similarity >= 80%   : ', cls_80, '/', total
print >> acc_f, 'equal                          : ', eq, '/', total

sim_out.close()
out_f.close()
acc_f.close()

