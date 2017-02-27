d = dict()
for w in words:
    n = len(w)
    if n > 2:
        if n not in d:
            d[n] = []
        d[n].append(w)
        
for i, l in d.items():
    print(i, l)