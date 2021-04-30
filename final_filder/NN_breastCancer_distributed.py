from itertools import tee
from typing import Sized
from sklearn.datasets import fetch_openml
from sklearn import preprocessing
import time
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from keras.utils.np_utils import to_categorical
from mpi4py import MPI
import matplotlib.pyplot as plt

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


class DeepNeuralNetwork():
    def __init__(self, sizes, epochs=20, l_rate=0.001):
        self.sizes = sizes
        self.epochs = epochs
        self.l_rate = l_rate

        # we save all parameters in the neural network in this dictionary
        self.params = self.initialization()
        self.accs = []

    def sigmoid(self, x, derivative=False):
        if derivative:
            return (np.exp(-x))/((np.exp(-x)+1)**2)
        return 1/(1 + np.exp(-x))

    def softmax(self, x, derivative=False):
        # Numerically stable with large exponentials
        exps = np.exp(x - x.max())
        if derivative:
            return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
        return exps / np.sum(exps, axis=0)

    def initialization(self):
        # number of nodes in each layer
        input_layer = self.sizes[0]
        hidden_1 = self.sizes[1]
        # hidden_2=self.sizes[2]
        output_layer = self.sizes[2]

        params = {
            'W1': np.random.randn(hidden_1, input_layer) * np.sqrt(1. / hidden_1),
            # 'W2':np.random.randn(hidden_2, hidden_1) * np.sqrt(1. / hidden_2),
            'W2': np.random.randn(output_layer, hidden_1) * np.sqrt(1. / output_layer),
            # 'W3':np.random.randn(output_layer, hidden_2) * np.sqrt(1. / output_layer)
        }

        return params

    def forward_pass(self, x_train):
        params = self.params

        # input layer activations becomes sample
        params['A0'] = x_train

        # input layer to hidden layer 1
        params['Z1'] = np.dot(params["W1"], params['A0'])
        params['A1'] = self.sigmoid(params['Z1'])

        # hidden layer 1 to output layer
        params['Z2'] = np.dot(params["W2"], params['A1'])
        params['A2'] = self.softmax(params['Z2'])

        # hidden layer 2 to output layer
        #params['Z3'] = np.dot(params["W3"], params['A2'])
        #params['A3'] = self.softmax(params['Z3'])

        return params['A2']

    def backward_pass(self, y_train, output):
        '''
            This is the backpropagation algorithm, for calculating the updates
            of the neural network's parameters.

            Note: There is a stability issue that causes warnings. This is 
                  caused  by the dot and multiply operations on the huge arrays.

                  RuntimeWarning: invalid value encountered in true_divide
                  RuntimeWarning: overflow encountered in exp
                  RuntimeWarning: overflow encountered in square
        '''
        params = self.params
        change_w = {}

        # Calculate W2 update
        # error = 2*(output-y_train) / \
        #    output.shape[0]*self.softmax(params['Z2'], derivative=True)
        #change_w['W2'] = np.outer(error, params['A1'])

        # Calculate W1 update
        # error = np.dot(params['W2'].T, error) * \
        #    self.sigmoid(params['Z1'], derivative=True)
        #change_w['W1'] = np.outer(error, params['A0'])

        error = grad2 = 2 * \
            (output-y_train)/output.shape[0] * \
            self.softmax(params['Z2'], derivative=True)
        grad2 = np.outer(grad2, params['A1'])
        grad2[grad2 > 0] = 1
        grad2[grad2 < 0] = -1
        grad2[grad2 == 0] = 0
        grad2 = grad2.astype(np.int8)
        comm.Send(grad2, dest=0, tag=6)
        comm.Recv(grad2, source=0, tag=9)
        change_w['W2'] = grad2

        error = grad1 = np.dot(
            params['W2'].T, error)*self.sigmoid(params['Z1'], derivative=True)
        grad1 = np.outer(grad1, params['A0'])
        grad1[grad1 > 0] = 1
        grad1[grad1 < 0] = -1
        grad1[grad1 == 0] = 0
        grad1 = grad1.astype(np.int8)
        comm.Send(grad1, dest=0, tag=4)
        comm.Recv(grad1, source=0, tag=7)
        change_w['W1'] = grad1

        return change_w

    def update_network_parameters(self, changes_to_w):
        '''
            Update network parameters according to update rule from
            Stochastic Gradient Descent.

            θ = θ - η * ∇J(x, y), 
                theta θ:            a network parameter (e.g. a weight w)
                eta η:              the learning rate
                gradient ∇J(x, y):  the gradient of the objective function,
                                    i.e. the change for a specific theta θ
        '''

        for key, value in changes_to_w.items():
            self.params[key] -= self.l_rate * value

    def compute_accuracy(self, x_val, y_val):
        '''
            This function does a forward pass of x, then checks if the indices
            of the maximum value in the output equals the indices in the label
            y. Then it sums over each prediction and calculates the accuracy.
        '''
        predictions = []

        for x, y in zip(x_val, y_val):
            output = self.forward_pass(x)
            pred = np.argmax(output)
            predictions.append(pred == np.argmax(y))

        return np.mean(predictions)

    def train(self, x_train, y_train, x_val, y_val):
        start_time = time.time()
        for iteration in range(self.epochs):
            for x, y in zip(x_train, y_train):
                output = self.forward_pass(x)
                changes_to_w = self.backward_pass(y, output)
                self.update_network_parameters(changes_to_w)

            #accuracy = self.compute_accuracy(x_val, y_val)
            # print('Epoch: {0}, Time Spent: {1:.2f}s, Accuracy: {2:.2f}%'.format(
            #    iteration+1, time.time() - start_time, accuracy * 100
            # ))

    def plot_accs(self):
        plt.plot(list(range(1, self.epochs+1)), self.accs)
        plt.xticks(list(range(1, self.epochs+1)))
        plt.yticks(np.arange(0.6, 1.0, 0.05))
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.show()


train_x = pd.read_csv("cancer_data.csv")
train_x = (train_x-train_x.mean())/train_x.std()
train_x = train_x.to_numpy().astype('float64')

train_y = pd.read_csv("cancer_data_y.csv")
train_y = to_categorical(train_y.to_numpy())

X_test = pd.read_csv("test_cancer_data.csv")
X_test = (X_test-X_test.mean())/X_test.std()
X_test = X_test.to_numpy().astype('float64')

Y_test = pd.read_csv("test_cancer_data_y.csv")
Y_test = to_categorical(Y_test.to_numpy())
#dnn = DeepNeuralNetwork(sizes=[30, 5, 2])
#dnn.train(train_x, train_y, X_test, Y_test)

EPOCHS = 8  # global number of epochs
L_RATE = 0.001  # global learning rate
SIZES = [30, 5, 2]

if rank == 0:
    start = time.time()
    accs = []
    mainNN = DeepNeuralNetwork(sizes=SIZES, epochs=EPOCHS, l_rate=L_RATE)
    comm.Send(mainNN.params['W1'], dest=1, tag=1)
    comm.Send(mainNN.params['W2'], dest=1, tag=2)
    comm.Send(mainNN.params['W1'], dest=2, tag=1)
    comm.Send(mainNN.params['W2'], dest=2, tag=2)

    print('Starting epochs')
    for epoch in range(EPOCHS):
        for i in range(209):
            # if i % 50 == 0:
            #    print(i)
            grad12 = np.empty(mainNN.params['W2'].shape, dtype=np.int8)
            comm.Recv(grad12, source=1, tag=6)
            grad22 = np.empty(mainNN.params['W2'].shape, dtype=np.int8)
            comm.Recv(grad22, source=2, tag=6)

            vote2 = grad12+grad22
            vote2[vote2 > 0] = 1
            vote2[vote2 < 0] = -1
            vote2[vote2 == 0] = 0
            comm.Send(vote2, dest=1, tag=9)
            comm.Send(vote2, dest=2, tag=9)

            grad11 = np.empty(mainNN.params['W1'].shape, dtype=np.int8)
            comm.Recv(grad11, source=1, tag=4)
            grad21 = np.empty(mainNN.params['W1'].shape, dtype=np.int8)
            comm.Recv(grad21, source=2, tag=4)

            vote1 = grad11+grad21
            vote1[vote1 > 0] = 1
            vote1[vote1 < 0] = -1
            vote1[vote1 == 0] = 0

            comm.Send(vote1, dest=1, tag=7)
            comm.Send(vote1, dest=2, tag=7)

            mainNN.update_network_parameters({'W1': vote1, 'W2': vote2})
        #print('Epochs: {}'.format(epoch+1))
        #print(mainNN.compute_accuracy(X_test, Y_test))
        accs.append(mainNN.compute_accuracy(X_test, Y_test))
    mainNN.accs = accs
    print(time.time()-start, ' seconds')
    mainNN.plot_accs()
elif rank == 1:
    nn1 = DeepNeuralNetwork(sizes=SIZES, epochs=EPOCHS, l_rate=L_RATE)
    comm.Recv(nn1.params['W1'], source=0, tag=1)
    comm.Recv(nn1.params['W2'], source=0, tag=2)
    train_x = train_x[:209]
    train_y = train_y[:209]
    nn1.train(train_x, train_y, X_test, Y_test)
elif rank == 2:
    nn2 = DeepNeuralNetwork(sizes=SIZES, epochs=EPOCHS, l_rate=L_RATE)
    comm.Recv(nn2.params['W1'], source=0, tag=1)
    comm.Recv(nn2.params['W2'], source=0, tag=2)
    train_x = train_x[209:]
    train_y = train_y[209:]
    nn2.train(train_x, train_y, X_test, Y_test)
