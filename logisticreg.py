import numpy as np

def sigmoid(z):
     return 1/(1+np.exp(-z))

def logistic_regression(X,y,learning_rate = 0.01,no_iterations=1000):

    m = X.shape[1]
    n = X.shape[0]

    w = np.zeros((n,1))
    b = 0

    for i in range(no_iterations):
        z = np.dot(w.T,X) + b
        a = sigmoid(z)

        cost = (1/m)*np.sum(-((y*np.log(a)) + ((1-y)*(np.log(1-a)))))

        dz = a-y
        db = (1/m)*np.sum(dz)
        dw = (1/m)*np.dot(X,dz.T)

        w = w - learning_rate*(dw)
        b = b - learning_rate*(db)

    return w,b

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

data = load_breast_cancer()
X = data.data
y = data.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

X = X.T
y = y.reshape(1,-1)
w,b = logistic_regression(X,y,learning_rate = 0.001,no_iterations=1000)
z = np.dot(w.T,X) + b
a = sigmoid(z)

predictions =(a >= 0.5).astype(int)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y,predictions)
print(accuracy)

