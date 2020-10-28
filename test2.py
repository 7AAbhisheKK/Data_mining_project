import numpy as np
from util import *
# a=np.array([[1,4,3],[4,2,6]])
# b=np.array([[1],[2]])
# # print(softmax(a).shape)
# # print(a+b)
# # print(a.T)
# p=np.argmax(a,axis=0)
# print(p)
def iterate_minibatches(inputs, targets, batchsize):
    for start_idx in range(0, inputs.shape[1] - batchsize + 1, batchsize):
        yield inputs[:,start_idx:start_idx+batchsize], targets[:,start_idx:start_idx+batchsize]
a=np.arange(50).reshape(5,-1)
b=np.arange(10).reshape(1,-1)
print(a,b)
for x_train,y_train in iterate_minibatches(a,b,3):
    print(x_train,y_train)

