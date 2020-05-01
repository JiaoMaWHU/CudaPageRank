import random
file = {}
row = 800
for i in range(row):
    file[i] = []
    for j in range(row):
        if random.randint(0, 1) == 0:
            file[i].append(j)

f = open("./%d.txt"%(row), "w")
lines = []
for k in file:
    for i in file[k]:
        lines.append("%d %d\n"%(k, i))
f.writelines(lines)
f.close()