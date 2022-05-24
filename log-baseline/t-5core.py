import numpy as np

user = np.random.randint(10, 12, (10)).tolist()
item = np.random.randint(1000, 1003, (10)).tolist()
print(user)
print(item)
for i in range(len(user)):
    user[i] = str(user[i])+'as3d'
    item[i] = str(item[i])+'as3d'

d = dict()
for i in range(len(user)):
    if (user[i], item[i]) not in d.keys():
        #print('none')
        d[(user[i], item[i])] = 1
    else:
        d[(user[i], item[i])] += 1

print(d)
print(len(d))
