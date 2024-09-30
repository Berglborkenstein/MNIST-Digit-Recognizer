import pandas
import numpy as np

data = pandas.read_csv('train.csv')

data = np.array(data)
m, n = data.shape
np.random.shuffle(data) # shuffle before splitting into dev and training sets

data_dev = data[0:1000].T
Y_test = data_dev[0]
X_test = data_dev[1:n]
X_test = X_test / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape

def init_params(hidden_size):
    # Assuming the 0th index of hidden_size is 784, and the final index is 10
    W = [np.random.rand(hidden_size[i],hidden_size[i-1]) - 0.5 for i in range(1,len(hidden_size))]
    b = [np.random.rand(hidden_size[i],1) - 0.5 for i in range(1,len(hidden_size))]

    return W,b

def ReLU(Zi):
    return np.maximum(Zi,0)

def softmax(Zi):
    return np.exp(Zi)/sum(np.exp(Zi))

def deriv_ReLU(Zi):
    return Zi > 0

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def Forward_Prop(X,W,b):
    Z = [X]
    A = [X]

    for i in range(len(W)):
        Z.append(W[i].dot(A[i]) + b[i])
        A.append(ReLU(Z[i+1]))
    
    A[-1] = softmax(Z[-1])
    return Z,A

def Back_Prop(X,Y,A,Z,W,b):
    one_hot_y = one_hot(Y)
    q = Y.size

    dZ = [np.zeros(z.shape) for z in Z]
    dW = [np.zeros(w.shape) for w in W]
    db = np.zeros(len(b))

    dZ[-1] = A[-1] - one_hot_y
    dZ[0] = X

    for i in range(len(dZ) - 2, 0, -1):
        dZ[i] = W[i].T.dot(dZ[i+1]) * deriv_ReLU(Z[i])

    for i in range(len(W)):
        dW[i] = 1 / q * dZ[i+1].dot(A[i].T)
        db[i] = 1 / q * np.sum(dZ[i+1])

    return dW,db

def Update(W,b,dW,db,alpha):
    for i in range(len(W)):
        W[i] -= alpha * dW[i]
    for i in range(len(b)):
        b[i] -= alpha * db[i]

    return W,b

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def grad_descent(X,Y,iterations,alpha,hidden_size):
    W,b = init_params(hidden_size)

    for i in range(iterations+1):
        Z,A = Forward_Prop(X,W,b)
        dW,db = Back_Prop(X,Y,A,Z,W,b)
        W,b = Update(W,b,dW,db,alpha)
        if i % (10) == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A[-1])
            print(f'Accuracy: {get_accuracy(predictions, Y)}')

    return W,b

def ai_test(X,Y,W,b):
    _,A = Forward_Prop(X,W,b)
    predictions = get_predictions(A[-1])
    print(get_accuracy(predictions,Y))

grad_descent(X_train,Y_train,1000,0.1,[784,10,10])