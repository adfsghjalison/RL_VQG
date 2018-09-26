import re


#create word table
s = set()
f = open('train/question', 'r')
f2 = open('train/cap_uni', 'r')
word = open('train/word_cap', 'w')

max_que = 0

for _f in [f, f2]:
    for q in _f:
        q = q.strip('\n').split()
        max_que = max(max_que, len(q))
        for j in q:
            if any(char.isdigit() for char in j) or '_' in j:
                continue
            s.add(j)
    print 'max :', max_que
    max_que = 0

print >> word, '\n'.join(sorted(s))
word.close()

