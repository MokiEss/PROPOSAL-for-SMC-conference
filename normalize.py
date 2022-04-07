

a = [10,-5,6,4,-7,6,3]
normalize_a = a
for i in range(len(a)) :
    normalize_a[i] = (a[i] - min(a))/(max(a)-min(a))
print((normalize_a))
