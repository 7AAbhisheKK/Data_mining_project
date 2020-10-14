import numpy as np
from util import *
a=np.array([[1,4,3],[4,2,6]])
b=np.array([[1],[2]])
# print(softmax(a).shape)
# print(a+b)
# print(a.T)
p=np.argmax(a,axis=0)
print(p)