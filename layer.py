from util import *
import numpy as np
def initialize_layer(dims): ###initialize the weight and bias matrix of every layer 
    np.random.seed(3)
    parameters={}
    L=len(dims)
    for i in range(1,L):
        parameters['W'+str(i)]=np.random.rand(dims[i],dims[i-1])*0.01
        # print("--", parameters['W'+str(i)].shape)
        parameters['b'+str(i)]=np.zeros((dims[i],1))
    return parameters


def linear_forward(A,W,b): ##calculate the z
    z=np.dot(W,A)+b
    cache=(A,W,b)
    return z,cache


def linear_forward_pass(A_prev,W,b,activation): ## calculate activation of layer
    if activation=="sigmoid":
        Z, linear_cache = linear_forward(A_prev,W,b)
        A, activation_cache = sigmoid(Z)
    elif activation=="Relu":
        Z, linear_cache = linear_forward(A_prev,W,b)
        A, activation_cache = relu(Z)
    cache = (linear_cache, activation_cache)
    return A,cache

def L_layer_forward_pass(X,parameters):
    caches=[]
    A=X
    L=len(parameters)//2
    for l in range(1,L):
        A_prev=A
        A,cache=linear_forward_pass(A_prev,parameters['W'+str(l)],parameters['b'+str(l)],"Relu")
        # print("caches for ",l,"layers ",cache)
        caches.append(cache)
    AL,cache=linear_forward_pass(A,parameters['W'+str(L)],parameters['b'+str(L)],"Relu")
    # print("caches for ",L,"layers ",cache)
    caches.append(cache)

    return AL,caches
def linear_backward(dZ,cache):
    A_prev, W, b = cache
    # print(A_prev.shape,dZ.shape)
    m = A_prev.shape[1]
    dW = (1/m)*np.dot(dZ,A_prev.T)
    db = 1/m*(np.sum(dZ,axis=1,keepdims=True))
    dA_prev = np.dot(W.T,dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db
def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA,activation_cache)
        dA_prev, dW, db = linear_backward(dZ, cache[0])
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA,activation_cache)
        dA_prev, dW, db = linear_backward(dZ, cache[0])
    
    return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
   
    grads = {}
    L = len(caches) 
    dAL = softmax_grad(AL,Y)
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache,"relu")
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA"+str(l+1)], current_cache,"relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads
def update_parameters(parameters, grads, learning_rate):
    
    L = len(parameters) // 2 
    for l in range(L):
        parameters["W" + str(l+1)] =parameters["W" + str(l+1)]-learning_rate*grads["dW"+str(l+1)]
        parameters["b" + str(l+1)] =parameters["b" + str(l+1)]-learning_rate*grads["db"+str(l+1)]
    return parameters            




