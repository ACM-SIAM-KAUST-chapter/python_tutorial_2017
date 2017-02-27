l = list(range(5, 17, 2))
print(l)
l[-1] = list(range(4, 10, 2))
print(l)
l[-1][-2] = -1
print(l)
del l[1:3]
print(l)
l.insert(1, "Hello")
print(l)