import os, argparse, sys
import cv2, spacy, numpy as np
#from keras.models import model_from_json
#from keras.optimizers import SGD
from sklearn.externals import joblib
from fuzzywuzzy import fuzz
    
BATCH_SIZE = 32

fea_f = open("../val/feature_val","r")
qus_f = open("../val/question", "r")
modellist = open('code/models_testing')
word_embeddings = spacy.load('en')
data_num = 72801

def get_VQA_model(ans_size, plan, w):
        sys.path.append(os.getcwd()+'/model/final')
        from VQA import VQA_MODEL
	vqa_model = VQA_MODEL(output_num = ans_size, path ='model/final/'+plan+'/'+w+'.h5')
	vqa_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
	return vqa_model
	
def read_word(plan):
	word_f = open('model/final/'+plan+'/ans.txt', 'r')
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
	X1 = np.zeros((data_num, 30, 300))
	X2 = []
	cnt = 0
	for index, f, q in zip(range(data_num), fea_f, qus_f):
		x1 = word_embeddings(unicode(q, 'utf-8'))
		for j in xrange(len(x1)):
			X1[index, j, :] = x1[j].vector
		x2 = [ float(i) for i in f.split() ]
		X2.append(x2)
		cnt = cnt+1
		if cnt%1000==0:
			print cnt
	X2 = np.asarray(X2)
	return X1, X2

print 'Load testing data ...'
X1, X2 = vectorize()
print X1.shape
print X2.shape

for l in modellist:
	l = l.strip('\n').split()
	plan = l[0]
	word, ans_size = read_word(plan)
	for w in l[1:]:
		print '\nTesting for ( plan , weights ) = (', plan, ',', w, ')\n'
		print "loading model..."
		vqa_model = get_VQA_model(ans_size, plan, w)
		print "load model successfully..."

		print("\npredicting ...")
		Y = vqa_model.predict([X1, X2])

		sim_out = open('output/'+plan+'/out-'+w, 'w')
		out_f = open('output/'+plan+'/out_cls-'+w, 'w')
		acc_f = open('output/'+plan+'/acc-'+w, 'w')
		
		cho = open("../val/choices.test", "r")
		ans = open("../val/answer.test", "r")
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

