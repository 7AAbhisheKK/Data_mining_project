import numpy as np
def sigmoid(x):
    activation= 1/(1+np.exp(-x))
    cache=x
    return activation,cache 

def relu(x):
    activation = np.maximum(x,0)
    cache=x
    return activation,cache

def softmax(X):
    return np.exp(X) / np.sum(np.exp(X), axis=0,keepdims=True)


def softmax_loss(X,Y):
    m = Y.shape[1]
    # print(X.shape,"#")
    correct = X[Y,np.arange(m)]
    # print(correct.shape,correct)
    loss = -correct+np.log(np.sum(np.exp(X),axis=0).reshape(1,-1))
    return loss
def softmax_grad(X,Y):
    correct=np.zeros_like(X)
    m=Y.shape[1]
    correct[Y,np.arange(m)]=1
    x=(-correct+softmax(X))
    return x    
def relu_backward(dA, cache):
    
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def sigmoid_backward(dA, cache):

    
    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ    
# n=np.array([1,2,3,5,9,7,4])
# np.random.seed(3)
# n1=np.random.rand(7,10)
# print(n.shape,n1.shape)
# print(softmax_loss(n1,n))

