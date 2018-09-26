import sys

if len(sys.argv)<3:
	print "python gen_word.py [in] [out]"
	sys.exit()

ans = set()
f = open(sys.argv[1], 'r')

for l in f:
	a = l.rstrip('\n').split(' | ')[2]
	ans.add(a)

ans = sorted(ans)
print >> open(sys.argv[2], 'w'), '\n'.join(ans)
