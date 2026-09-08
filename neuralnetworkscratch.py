import numpy as np

X = np.array([[0,0,1,1],
     [0,1,0,1]])

#y = np.array([0,1,1,0])#this gives shape (4,)as shape which is hard for backpropogation 
y = np.array([[0,1,1,0]])#this gives shape (1,4)as shape which is better for backpropogation 


learning_rate = 0.1

def init_parameters():
    W1 = np.random.randn(4,2)
    b1 = np.random.randn(4,1)
    W2 = np.random.randn(1,4)
    b2 = np.random.randn(1,1)

    return W1,b1,W2,b2


def ReLU(Z):
    return np.maximum(0,Z)

def ReLU_derivative(Z):
    return (Z>0).astype(float)

def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def forward_propogation(W1,b1,W2,b2,X):
    Z1 = np.dot(W1,X) + b1
    A1 = ReLU(Z1)
    Z2 = np.dot(W2,A1)+b2
    #A2 = ReLU(Z2) for output we need to 0-1 so we have to use sigmoid
    A2 = sigmoid(Z2)

    return A2,A1,Z1,Z2

def cost(A2,y,m):
    value = (1/m)*(np.sum(-((y*np.log(A2))+(1-y)*np.log(1-A2))))
    return value

def backward_propogation(W1,b1,W2,b2,X,A2,y,m,A1,Z1):
    dZ2 = A2 - y
    dW2 = (1/m)*np.dot(dZ2,A1.T)#here we need the transpose cause it is in the formula of dW2
    db2 = (1/m)*np.sum(dZ2,axis=1,keepdims=True)
    dZ1 = (np.dot(W2.T,dZ2))*ReLU_derivative(Z1)
    dW1 = (1/m)*np.dot(dZ1,X.T)
    db1 = (1/m)*np.sum(dZ1,axis=1,keepdims=True)

    return dW1,dW2,db1,db2

def learning(dW1,dW2,db1,db2,W1,W2,b1,b2,learning_rate):
    W1 = W1 - learning_rate*(dW1)
    W2 = W2 - learning_rate*(dW2)
    b1 = b1 - learning_rate*(db1)
    b2 = b2 - learning_rate*(db2)

    return W1,W2,b1,b2

"""
# this code only runs this once now we have to run it for multiple iterations
W1,b1,W2,b2 = init_parameters()

A2,A1,Z1,Z2 =forward_propogation(W1,b1,W2,b2,X)

m = X.shape[1]

J = cost(A2,y,m)
print(J)

dW1,dW2,db1,db2 = backward_propogation(W1,b1,W2,b2,X,A2,y,m,A1,Z1)

W1,W2,b1,b2 = learning(dW1,dW2,db1,db2,W1,W2,b1,b2,learning_rate)

"""

'''
Just to check if everything is correct or not

print("W1:", W1.shape)
print("b1:", b1.shape)

print("Z1:", Z1.shape)
print("A1:", A1.shape)

print("Z2:", Z2.shape)
print("A2:", A2.shape)

print("dW1:", dW1.shape)
print("db1:", db1.shape)

print("dW2:", dW2.shape)
print("db2:", db2.shape)

'''

W1,b1,W2,b2 = init_parameters()

for i in range(10000):
    
    A2,A1,Z1,Z2 =forward_propogation(W1,b1,W2,b2,X)

    m = X.shape[1]

    J = cost(A2,y,m)

    dW1,dW2,db1,db2 = backward_propogation(W1,b1,W2,b2,X,A2,y,m,A1,Z1)

    W1,W2,b1,b2 = learning(dW1,dW2,db1,db2,W1,W2,b1,b2,learning_rate)

    if i%1000 == 0:
        print(i,J)

