import pandas
import numpy as np
import matplotlib.pyplot as plt
from Graph import graph

data = pandas.read_csv('train.csv')

data = np.array(data)
m, n = data.shape
np.random.shuffle(data) # shuffle before splitting into dev and training sets

data_dev = data[0:100].T
Y_test = data_dev[0]
X_test = data_dev[1:n]
X_test = X_test / 255.

data_train = data[100:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape

def batches(batch_size):
    full_size = m_train  # Total number of training examples
    num_batches = full_size // batch_size  # Number of full batches
    
    # Shuffle the data
    permutation = np.random.permutation(full_size)
    shuffled_X = X_train[:, permutation]
    shuffled_Y = Y_train[permutation]
    
    # Generate batches
    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size
        
        # Get the batch of X and Y
        X_batch = shuffled_X[:, start:end]
        Y_batch = shuffled_Y[start:end]
        
        yield X_batch, Y_batch  # Return the batch
    
    # Handle the last batch if the total number of samples isn't divisible by batch_size
    if full_size % batch_size != 0:
        X_batch = shuffled_X[:, num_batches * batch_size:]
        Y_batch = shuffled_Y[num_batches * batch_size:]
        yield X_batch, Y_batch

def init_params(hidden_size): # Base
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




def Forward_Prop(X,W,b,dropout = 0):
    Z = [X]
    A = [X]
    dropout_masks = []

    for i in range(len(W)):
        Z.append(W[i].dot(A[i]) + b[i])
        A_current = ReLU(Z[i+1])

        mask = [1 if np.random.random() > dropout else 0 for item in range(len(A_current))]
        A_new = [(A_current[i] * mask[i])/(1 - dropout) for i in range(len(A_current))]
        
        A.append(A_new)
        dropout_masks.append(mask)

    A[-1] = softmax(Z[-1])
    return Z,A,dropout_masks

def Back_Prop(X,Y,A,Z,W,b,dropout,dropout_masks):
    one_hot_y = one_hot(Y)
    q = Y.size

    dZ = [np.zeros(z.shape) for z in Z]
    dW = [np.zeros(w.shape) for w in W]
    db = np.zeros(len(b))

    dZ[-1] = A[-1] - one_hot_y
    dZ[0] = X

    for i in range(len(dZ) - 2, 0, -1):
        dZ_current = W[i].T.dot(dZ[i+1]) * deriv_ReLU(Z[i])

        if dropout_masks:
            mask  = dropout_masks[i-1]
            dZ_current = [(dZ_current[j] * mask[j])/(1 - dropout) for j in range(len(dZ_current))]

            dZ[i] = np.array(dZ_current)

    for i in range(len(W)):
        dW[i] = 1 / q * dZ[i+1].dot(np.array(A[i]).T)
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


def calc_momentum(momentum, dk, v_k):

    a = 1 - momentum
    dk_new = [a * item for item in dk]
    v_k_prev = [momentum * item for item in v_k]
    v_k = [v_k_prev[i] + dk_new[i] for i in range(len(dk_new))]

    return v_k





def grad_descent(X,Y,iterations,alpha,hidden_size, momentum = 0, dropout = 0):
    W,b = init_params(hidden_size)
    num_iter = 0.1 * iterations

    if momentum:
        v_w = [np.zeros_like(w) for w in W]
        v_b = [np.zeros_like(bi) for bi in b]

    # Sets up values for the graph
    fig,ax = plt.subplots()
    train_line, = ax.plot([], [], 'r-', label='Train Accuracy')
    test_line, = ax.plot([], [], 'b-', label='Test Accuracy')
    ax.legend()

    ax.set_ylim(0,1)
    ax.set_xlim(0,iterations)
    
    plt.ion()

    # Starts the loop
    for i in range(iterations+1):
        Z,A,dropout_masks = Forward_Prop(X,W,b,dropout)
        dW,db = Back_Prop(X,Y,A,Z,W,b,dropout,dropout_masks)

        if momentum: 
            v_w = calc_momentum(momentum, dW, v_w)
            v_b = calc_momentum(momentum, db, v_b)
            dW, db = v_w, v_b

        W,b = Update(W,b,dW,db,alpha)

        if i % (num_iter) == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A[-1])
            train_accuracy = get_accuracy(predictions, Y)
            test_accuracy = ai_test(X_test,Y_test,W,b,dropout)
            print(f'Accuracy: {train_accuracy}')  

            graph(train_accuracy,test_accuracy,i,train_line,test_line,ax)  
    
    for i in range(len(W)):
        W_current = W[i]
        b_current = b[i]

        np.save(f'Weight{i}.npy',W_current)
        np.save(f'Bias{i}.npy',b_current)

    plt.pause(3)

    return W,b

def ai_test(X,Y,W,b,dropout):
    _,A,_ = Forward_Prop(X,W,b,dropout)
    predictions = get_predictions(A[-1])
    return get_accuracy(predictions,Y)

    #return [item[0] for item in A[-1]]





def display_test_images(X_test):
    # Check the number of samples in X_test
    num_samples = X_test.shape[1] if X_test.ndim > 1 else 1  # If X_test has 2 dimensions
    image_size = 28

    # Create a figure for displaying the images
    if num_samples == 1:
        fig, ax = plt.subplots(figsize=(5, 5))
        image_data = X_test.flatten()  # Flatten to 1D if it’s a single image
        ax.imshow(image_data.reshape(image_size, image_size), cmap='gray')
        ax.axis('off')  # Hide axis labels
        ax.set_title('Test Image', fontsize=16)
    else:
        # Create a grid for multiple images
        num_cols = 10
        num_rows = (num_samples + num_cols - 1) // num_cols  # Calculate number of rows needed
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 1.5 * num_rows))

        # If we have only one row, axes is not a list of lists, so we convert it
        if num_rows == 1:
            axes = [axes]

        fig.suptitle('Test Images', fontsize=16)

        for i in range(num_samples):
            # Get the image data and reshape it to 28x28
            image_data = X_test[:, i].reshape(image_size, image_size)

            # Find the appropriate subplot
            row, col = divmod(i, num_cols)
            ax = axes[row][col] if num_rows > 1 else axes[col]

            # Display the image
            ax.imshow(image_data, cmap='gray')
            ax.axis('off')  # Hide axis labels

        # Hide any remaining unused subplots
        for j in range(num_samples, num_rows * num_cols):
            row, col = divmod(j, num_cols)
            ax = axes[row][col] if num_rows > 1 else axes[col]
            ax.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Adjust to make room for the title
    plt.show()

W,b = grad_descent(X_train,Y_train,1000,0.01,[784,128,10],momentum = 0.7, dropout = 0.3)