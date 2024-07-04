import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt 

data = pd.read_csv('./digit-recognizer/train.csv')

data = np.array(data)
m, n = data.shape
print(m,n)
np.random.shuffle(data)

# This is the data that will be used for testing the NN after it has been trained
# We transpose the matrix so that the rows of data become the columns of data
data_dev = data[:1000].T
# These are all the answers
Y_dev = data_dev[0]
# These are all the individual pixels that will be inputs to the NN
X_dev = data_dev[1:n]
X_dev = X_dev / 255.
# This is the data that will be used for training
data_train = data[1000:m].T
# Again answers for the data
Y_train = data_train[0]
# Input data for the NN
X_train = data_train[1:n]
# We want to match the range of inputs with the range of the activation function
# In this case we use ReLU and Softmax which both are between [0,1]
# So I think thats why each pixel is scaled between [0,1] in the training data
# The NN does not work without this change
# The input range does impact the output range and vice versa
# When i was training with pixel values ranging [0,255], the gradient 
# descent was super small and fell towards 0 rather than 1000
X_train = X_train / 255.

# checking the shape
print(X_train[:, 0].shape)

# Initializing weights and biases for hidden and output layers of the NN
def init_params():
    W1 = np.random.normal(size=(10, 784)) * np.sqrt(1./(784))
    b1 = np.random.normal(size=(10, 1)) * np.sqrt(1./10)
    W2 = np.random.normal(size=(10, 10)) * np.sqrt(1./20)
    b2 = np.random.normal(size=(10, 1)) * np.sqrt(1./(784))
    return W1, b1, W2, b2
    # # W1 is the weights for the hidden layer, we make a 2d array that is 10 x 784
    # # we then subtract 0.5 from it so that it goes from -0.5 to 0.5
    # # You can also do np.random.randn() instead
    # W1 = np.random.randn(10,784)
    # # b1 are the biases for the hidden layers nodes, its shape is 10 x 1
    # # We do the same thing here and subtract 0.5 so that the values are
    # # between -0.5 and 0.5
    # b1 = np.random.randn(10,1)
    # # W2 is the weights for the ou:%y+tput layer, we make a 2d array that is 10 x 10
    # # we then subtract 0.5 from it so that it goes from -0.5 to 0.5
    # # You can also do np.random.randn() instead
    # W2 = np.random.randn(10,10)
    # # b2 are the biases for the output layer nodes, its shape is 10 x 1
    # b2 = np.random.randn(10,1)
    #
    # return W1, b1, W2, b2

# Activation function ReLU for the hidden layer
# Z is the computed values for the hidden layers after weights and biases
def ReLU(Z):
    # Going through each element in Z and if its greater than 0, return values
    # If not return 0
    return np.maximum(0, Z)

# Line of ReLU has a max slope of 1, and a minimum slope of 0 
# This just takes the highest of those
def derivative_ReLU(Z):
    return Z > 0

# Activation function Softmax for the output layer
# Each number is put to the power of exponential function to help remove negative numbers
# Amplification of the differences, and normalization
# Basically take each value and e to its power, e^value 
# And then divide each output from the total sum to find the chance that the output 
# Is correct
# Pretty simple find percent of total function
# Softmax is the vector generalization of the Sigmoid function
def softmax(Z):
    Z -= np.max(Z, axis=0)  # Subtract max value for numerical stability
    A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
    return A

# Function to return a matrix of the correct number that was supposed to be percieved
# With a 1 at that position and 0 at every other position
def one_hot(Y):
    # size of Y for the size of the matrix
    # Y.max() assumes 0-9 and we want 0-10
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    # Find the column specified by the label in Y and set it to 1
    # We're indexing through one_hot_Y by using an array of the training set, size m 
    # So we're effectively saying find the row with Y.size, then find the column with Y 
    # Then set the value to 1 and keep everything else 0
    one_hot_Y[np.arange(Y.size), Y] = 1 
    # Transposing for each column to be an example
    one_hot_Y = one_hot_Y.T
    
    return one_hot_Y
    

# Perform forward propagation in our NN
def forward_propagation(W1, b1, W2, b2, X):
    # Take the dot product of the weights from the hidden layer and the pixels
    # And add the biases with matrix addition
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)

    return Z1, A1, Z2, A2

def backward_propagation(Z1, A1, Z2, A2, W2, X, Y):
    m = Y.size
    one_hot_Y = one_hot(Y)
    # Difference of softmax and actual answer
    dZ2 = 2 * (A2 - one_hot_Y)
    # Propagating the weights with differences based on the dot product of A1 and dZ2
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    # Dot product of W2 and derivate of Z2 (weights of output layer)
    # multiplied by the derivative of the ReLU activation function
    dZ1 = W2.T.dot(dZ2) * derivative_ReLU(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)

    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    # Update all the weights and biases based on the respective derivatives
    # and on the learning rate, alpha
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2

    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X,Y, iterations, alpha):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_propagation(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_propagation(Z1, A1, Z2, A2, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            print("Accuracy: ", get_accuracy(get_predictions(A2), Y))

    return W1, b1, W2, b2


def main():
    W1, b1, W2, b2 = gradient_descent(X_train,Y_train, 500, 0.1)


if __name__ == '__main__':
    main()
