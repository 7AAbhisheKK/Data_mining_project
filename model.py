import tqdm
from layer import *
from data_visulisation import *
X_train,Y_train,X_test,Y_test=dataset()
Y_train=Y_train.reshape((1,-1))
Y_test
dims=[]
dims.insert(0,X_train.shape[0])
dims.append(128)
# dims.append(64)
print(X_test.shape,Y_test.shape)
dims.append(10)
parameters=initialize_layer(dims)
AL,caches=L_layer_forward_pass(X_train,parameters)
c=softmax_grad(AL,Y_train)
# for i in tqdm.tqdm(range(600)):
#     AL,caches=L_layer_forward_pass(X_train,parameters)
#     loss=softmax_loss(AL,Y_train)
#     if i%50==0:
#         print(loss.mean())
#     grads=L_model_backward(AL, Y_train, caches)
#     parameters=update_parameters(parameters, grads, 0.01)
# predict(X_test,Y_test,parameters)
