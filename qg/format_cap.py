
f = open('train/cap_index')

cnt = 0
for l in f:
    l = l.strip()
    if '.' not in l:
        print l, '.'
    else:
        print l

