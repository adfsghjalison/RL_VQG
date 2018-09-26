
for i in [0, 2, 4, 6]:
	for j in range(1, 6):
		f = open('output/'+str(i)+'_'+str(j)+'/acc-50')
		f.readline()
		f.readline()
		print  f.readline()[:4]+'\t',
	print ''

