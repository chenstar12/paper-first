import time

index = range(10)

import numpy as np

d = np.random.normal(size=(20))
print(d)
print(d[index])


s= 'trans--'
print(s[:5]=='trans')

print(1e4)

for i in range(10):
    time.sleep(1)
    print(i)