from sklearn.datasets import fetch_openml
import numpy as np
import time
from mpi4py import MPI
import pandas as pd
from matplotlib import pyplot as plt

#x, y = fetch_openml('mnist_784', version=1, return_X_y=True)
#x = (x/255).astype('float32')
#y = to_categorical(y)
#
# x_train, x_val, y_train, y_val = train_test_split(
#    x, y, test_size=0.15, random_state=42)

# now loading data from the csv files
x_train = np.loadtxt('x_train.csv', delimiter=',')
x_val = np.loadtxt('x_test.csv', delimiter=',')
y_train = np.loadtxt('y_train.csv', delimiter=',')
y_val = np.loadtxt('y_test.csv', delimiter=',')
#print('read data')


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
        hidden_2 = self.sizes[2]
        output_layer = self.sizes[3]

        params = {
            'W1': np.random.randn(hidden_1, input_layer) * np.sqrt(1. / hidden_1),
            'W2': np.random.randn(hidden_2, hidden_1) * np.sqrt(1. / hidden_2),
            'W3': np.random.randn(output_layer, hidden_2) * np.sqrt(1. / output_layer)
        }

        return params

    def forward_pass(self, x_train):
        params = self.params

        # input layer activations becomes sample
        params['A0'] = x_train

        # input layer to hidden layer 1
        params['Z1'] = np.dot(params["W1"], params['A0'])
        params['A1'] = self.sigmoid(params['Z1'])

        # hidden layer 1 to hidden layer 2
        params['Z2'] = np.dot(params["W2"], params['A1'])
        params['A2'] = self.sigmoid(params['Z2'])

        # hidden layer 2 to output layer
        params['Z3'] = np.dot(params["W3"], params['A2'])
        params['A3'] = self.softmax(params['Z3'])

        return params['A3']

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

        # communicate all 3 gradients one by one
        # Calculate W3 update
        error = grad3 = 2 * (output - y_train) / \
            output.shape[0] * self.softmax(params['Z3'], derivative=True)
        grad3 = np.outer(grad3, params['A2'])
        grad3[grad3 > 0] = 1
        grad3[grad3 < 0] = -1
        grad3[grad3 == 0] = 0
        comm.Send(grad3, dest=0, tag=6)
        comm.Recv(grad3, source=0, tag=9)
        change_w['W3'] = grad3

        # Calculate W2 update
        error = grad2 = np.dot(params['W3'].T, error) * \
            self.sigmoid(params['Z2'], derivative=True)
        grad2 = np.outer(grad2, params['A1'])
        grad2[grad2 > 0] = 1
        grad2[grad2 < 0] = -1
        grad2[grad2 == 0] = 0
        comm.Send(grad2, dest=0, tag=5)
        comm.Recv(grad2, source=0, tag=8)
        change_w['W2'] = grad2

        # Calculate W1 update
        error = grad1 = np.dot(params['W2'].T, error) * \
            self.sigmoid(params['Z1'], derivative=True)
        grad1 = np.outer(grad1, params['A0'])
        grad1[grad1 > 0] = 1
        grad1[grad1 < 0] = -1
        grad1[grad1 == 0] = 0
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

    def train(self, x_train, y_train):
        start_time = time.time()
        for iteration in range(self.epochs):
            for x, y in zip(x_train, y_train):
                output = self.forward_pass(x)
                changes_to_w = self.backward_pass(y, output)
                self.update_network_parameters(changes_to_w)
            # for i in range(len(x_train)//MINI_BATCH_SIZE):
            #    output = self.forward_pass(
            #        x_train[:,i*MINI_BATCH_SIZE:(i+1)*MINI_BATCH_SIZE])
            #    changes_to_w = self.backward_pass(
            #        y_train[i*MINI_BATCH_SIZE:(i+1)*MINI_BATCH_SIZE].T, output)
            #    self.update_network_parameters(changes_to_w)
            #accuracy = self.compute_accuracy(x_val, y_val)
            # print('Epoch: {0}, Time Spent: {1:.2f}s, Accuracy: {2:.2f}%'.format(
            #    iteration+1, time.time() - start_time, accuracy * 100
            # ))

    def plot_accs(self):
        plt.plot(list(range(1, self.epochs+1)), self.accs)
        plt.xticks(list(range(1, self.epochs+1)))
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.show()


comm = MPI.COMM_WORLD  # initialize mpi
rank = comm.Get_rank()  # get rank, in other words, worker number for this process
EPOCHS = 16  # global number of epochs
L_RATE = 0.000001  # global learning rate
SIZES = [784, 128, 64, 10]
MINI_BATCH_SIZE = 100

if rank == 0:
    start = time.time()
    accs = []
    mainNN = DeepNeuralNetwork(
        sizes=SIZES, epochs=EPOCHS, l_rate=L_RATE)
    comm.Send(mainNN.params['W1'], dest=1, tag=1)
    comm.Send(mainNN.params['W2'], dest=1, tag=2)
    comm.Send(mainNN.params['W3'], dest=1, tag=3)

    comm.Send(mainNN.params['W1'], dest=2, tag=1)
    comm.Send(mainNN.params['W2'], dest=2, tag=2)
    comm.Send(mainNN.params['W3'], dest=2, tag=3)
    comm.Send(mainNN.params['W1'], dest=3, tag=1)
    comm.Send(mainNN.params['W2'], dest=3, tag=2)
    comm.Send(mainNN.params['W3'], dest=3, tag=3)

    print('Starting loops')
    for epoch in range(EPOCHS):
        for i in range(20000):
            # if i % 1000 == 0:
            #    print(i)
            grad13 = np.empty(mainNN.params['W3'].shape, dtype='float64')
            comm.Recv(grad13, source=1, tag=6)
            grad23 = np.empty(mainNN.params['W3'].shape, dtype='float64')
            comm.Recv(grad23, source=2, tag=6)
            grad33 = np.empty(mainNN.params['W3'].shape, dtype='float64')
            comm.Recv(grad33, source=3, tag=6)
            vote3 = grad13+grad23+grad33
            vote3[vote3 > 0] = 1
            vote3[vote3 < 0] = -1
            vote3[vote3 == 0] = 0
            comm.Send(vote3, dest=1, tag=9)
            comm.Send(vote3, dest=2, tag=9)
            comm.Send(vote3, dest=3, tag=9)
            grad12 = np.empty(mainNN.params['W2'].shape, dtype='float64')
            comm.Recv(grad12, source=1, tag=5)
            grad22 = np.empty(mainNN.params['W2'].shape, dtype='float64')
            comm.Recv(grad22, source=2, tag=5)
            grad32 = np.empty(mainNN.params['W2'].shape, dtype='float64')
            comm.Recv(grad32, source=3, tag=5)
            vote2 = grad12+grad22+grad32
            vote2[vote2 > 0] = 1
            vote2[vote2 < 0] = -1
            vote2[vote2 == 0] = 0
            comm.Send(vote2, dest=1, tag=8)
            comm.Send(vote2, dest=2, tag=8)
            comm.Send(vote2, dest=3, tag=8)
            grad11 = np.empty(mainNN.params['W1'].shape, dtype='float64')
            comm.Recv(grad11, source=1, tag=4)
            grad21 = np.empty(mainNN.params['W1'].shape, dtype='float64')
            comm.Recv(grad21, source=2, tag=4)
            grad31 = np.empty(mainNN.params['W1'].shape, dtype='float64')
            comm.Recv(grad31, source=3, tag=4)
            vote1 = grad11+grad21+grad31
            vote1[vote1 > 0] = 1
            vote1[vote1 < 0] = -1
            vote1[vote1 == 0] = 0
            comm.Send(vote1, dest=1, tag=7)
            comm.Send(vote1, dest=2, tag=7)
            comm.Send(vote1, dest=3, tag=7)
            mainNN.update_network_parameters(
                {'W1': vote1, 'W2': vote2, 'W3': vote3})

        #print('Epochs: {}'.format(epoch+1))
        #print(mainNN.compute_accuracy(x_val, y_val))
        accs.append(mainNN.compute_accuracy(x_val, y_val))
    mainNN.accs = accs
    print(time.time()-start, ' seconds')
    mainNN.plot_accs()

elif rank == 1:
    #print('Start rank 1')
    nn1 = DeepNeuralNetwork(sizes=SIZES, epochs=EPOCHS, l_rate=L_RATE)
    #temp = np.empty((128, 784), dtype=np.float64)
    comm.Recv(nn1.params['W1'], source=0, tag=1)
    comm.Recv(nn1.params['W2'], source=0, tag=2)
    comm.Recv(nn1.params['W3'], source=0, tag=3)
    #print('Start rank 1')
    #print('on nn1')
    # print(nn1.params['W1'])
    x_train = x_train[:20000]
    y_train = y_train[:20000]
    nn1.train(x_train, y_train)
elif rank == 2:
    #print('Start rank 2')
    nn2 = DeepNeuralNetwork(sizes=SIZES, epochs=EPOCHS, l_rate=L_RATE)
    comm.Recv(nn2.params['W1'], source=0, tag=1)
    comm.Recv(nn2.params['W2'], source=0, tag=2)
    comm.Recv(nn2.params['W3'], source=0, tag=3)
    #print('Start rank 2')
    x_train = x_train[20000:40000]
    y_train = y_train[20000:40000]
    nn2.train(x_train, y_train)
elif rank == 3:
    #print('Start rank 3')
    nn3 = DeepNeuralNetwork(sizes=SIZES, epochs=EPOCHS, l_rate=L_RATE)
    comm.Recv(nn3.params['W1'], source=0, tag=1)
    comm.Recv(nn3.params['W2'], source=0, tag=2)
    comm.Recv(nn3.params['W3'], source=0, tag=3)
    #print('Start rank 3')
    x_train = x_train[40000:]
    y_train = y_train[40000:]
    nn3.train(x_train, y_train)
