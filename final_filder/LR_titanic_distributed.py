# go to readme first
from mpi4py import MPI
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
import math
import time 

# basic idea of how mpi works
#comm = MPI.COMM_WORLD
#rank = comm.Get_rank()
#
# if rank == 0:
#    data = {'a': 7, 'b': 3.14}
#    comm.send(data, dest=1, tag=11)
# elif rank == 1:
#    data = comm.recv(source=0, tag=11)

df = pd.read_csv('titanic.csv')

# begin preprocessing
cols = ['Name', 'Ticket', 'Cabin']
df = df.drop(cols, axis=1)
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
df['Embarked'] = df['Embarked'].fillna(2)
df['Age'] = df['Age'].interpolate()
# print(df.columns)
cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = df[cols]
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()
NUM_FEATURES = len(cols)  # I will often add 1 to it for bias
# print(X_train[:5])
# exit()


class LogRegSGD:  # ignore this class
    def __init__(self, learning_rate=0.05, num_iterations=10000):
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = None
        self.num_iterations = num_iterations
        self.accs = []

    def batchgradient(self, X, y):
        num_samples, num_features = X.shape
        linear_model = np.dot(X, self.weights)
        y_pred = self.sigmoid(linear_model)
        dw = (1/num_samples)*np.dot(X.T, (y_pred-y))
        # print(dw)
        return dw

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.ones(num_features)
        self.bias = 0

        # gradient descent
        convergence = False
        X, y = shuffle(X, y)
        batches = int(math.floor((len(X)/128)))
        # while(convergence is not True):
        for i in range(self.num_iterations):
            if i % 100 == 0:
                print(i)
            # Wx+B
            linear_model = np.dot(X, self.weights)
            y_pred = self.sigmoid(linear_model)
            gradient = 0
            vote = [0, 0, 0, 0, 0, 0, 0]
            for b in range(batches):
                if (b != batches-1):
                    gradient += self.batchgradient(
                        X[b*128:((b+1)*128)], y[b*128:((b+1)*128)])
                else:
                    gradient += self.batchgradient(X[b*128:len(X)],
                                                   y[b*128:len(X)])
                # print(gradient)
                for i in range(len(vote)):
                    if(gradient[i] < 0):
                        vote[i] += -1
                    else:
                        vote[i] = +1

            dw = gradient/batches
            for i in range(len(vote)):
                if(vote[i] < 0):
                    if(dw[i] >= 0):
                        dw[i] = (dw[i]*-1)
                else:
                    if(dw[i] <= 0):
                        dw[i] = dw[i]*-1

            old_wts = self.weights
            # old_bias=self.bias

            self.weights = self.weights-self.learning_rate*dw
            # self.bias=self.bias-self.learning_rate*db

            diff_weights = np.sum(old_wts-self.weights)

            y_pred[y_pred > 0.5] = 1
            y_pred[y_pred <= 0.5] = 0
            self.accs.append(accuracy_score(y, y_pred))

            # print(diff_weights)
            # if(diff_weights<abs(0.000001)):
            #    convergence=True

    def predict(self, X):
        linear_model = np.dot(X, self.weights)+self.bias
        y_pred = self.sigmoid(linear_model)
        predictions = []
        for i in y_pred:
            if(i > 0.5):
                predictions.append(1)
            else:
                predictions.append(0)
        return predictions

    def sigmoid(self, x):
        return(1/(1+np.exp(-x)))

    def plot_accs(self, step_size=500):
        x_data = []
        y_data = []
        for i in range(1, self.num_iterations+1, step_size):
            x_data.append(i)
            y_data.append(self.accs[i-1])
        x_data.append(self.num_iterations)
        y_data.append(self.accs[-1])
        plt.plot(x_data, y_data)
        plt.xticks(x_data)
        plt.xlabel('Number of iterations')
        plt.ylabel('Accuracy')
        plt.show()


#xcla = LogRegSGD()
#xcla.fit(X_train, y_train)
#res = xcla.predict(X_test)
#print("Sign SGD", accuracy_score(res, y_test))
#print(confusion_matrix(y_test, res))
# xcla.plot_accs(1000)


class LogReg:  # Ignore this class, just for reference
    def __init__(self):
        self.W = None

    # method could be one vs one or one vs all
    def train(self, X, y, alpha, epochs=100, iterations=2000):
        N = X.shape[0]  # number of data points
        Nfs = X.shape[1]+1  # number of features with a bias

        self.W = np.random.rand(Nfs)  # random W vector with bias
        newX = np.ones((N, Nfs))  # augment X for bias
        newX[:, :-1] = X
        X = newX

        for i in range(epochs):
            # print(i)
            for j in range(iterations):
                hx = np.dot(X, self.W)
                hx = 1/(1+np.exp(-hx))
                self.W = self.W - alpha*np.dot((hx-y).T, X)/N
                hx[hx >= 0.5] = 1.0
                hx[hx <= 0.5] = 0.0
                count = np.count_nonzero(hx == y)
                # print(count)
                if count >= 0.98*N:
                    break

    def accuracy(self, testX, testy):
        augx = np.ones((testX.shape[0], testX.shape[1]+1))
        augx[:, :-1] = testX
        prediction = np.dot(augx, self.W)
        prediction = 1/(1+np.exp(-prediction))
        prediction[prediction >= 0.5] = 1.0
        prediction[prediction <= 0.5] = 0.0
        count = np.count_nonzero(prediction == testy)
        return count/testy.shape[0]

    def predict(self, X):
        augx = np.ones((X.shape[0], X.shape[1]+1))
        augx[:, :-1] = X
        prediction = np.dot(augx, self.W)
        prediction = 1/(1+np.exp(-prediction))
        prediction[prediction >= 0.5] = 1.0
        prediction[prediction < 0.5] = 0.0
        return prediction


class LogisticReg:  # the class we are using
    def __init__(self):  # weight vector belonging to this class
        self.W = None
        self.accs = []
        self.num_iterations = 0

    # get a randomly generated weight vector once for parameter server which also includes bias
    def getInitialW(self, numFeatures: int):
        return np.random.rand(numFeatures)

    def sigmoid(self, x):  # basic sigmoid function
        return(1/(1+np.exp(-x)))

    def accuracy(self, testX, testy):  # accuracy function for test data
        augx = np.ones((testX.shape[0], testX.shape[1]+1))
        augx[:, :-1] = testX
        prediction = np.dot(augx, self.W)
        prediction = self.sigmoid(prediction)
        prediction[prediction >= 0.5] = 1.0
        prediction[prediction <= 0.5] = 0.0
        count = np.count_nonzero(prediction == testy)
        return count/testy.shape[0]

    def predict(self, X):  # predict on data
        augx = np.ones((X.shape[0], X.shape[1]+1))
        augx[:, :-1] = X
        prediction = np.dot(augx, self.W)
        prediction = self.sigmoid(prediction)
        prediction[prediction >= 0.5] = 1.0
        prediction[prediction < 0.5] = 0.0
        return prediction

    def getGradient(self, X, y):  # get gradient of logistic regression
        N = y.shape[0]
        hx = np.dot(X, self.W)
        hx = self.sigmoid(hx)
        hx[hx >= 0.5] = 1
        hx[hx <= 0.5] = 0
        result = np.dot((hx-y).T, X)/N
        return result, hx

    def train(self, W, trainX, trainy, alpha, iterations=10000):  # train on training data
        self.W = W  # take weight vector from parameter server initially
        N = trainX.shape[0]  # number of data points
        # one more column of only ones attached to training data for bias
        self.num_iterations = iterations
        newX = np.ones((N, NUM_FEATURES+1))
        newX[:, :-1] = trainX

        for j in range(iterations):
            # get gradient in each iteration
            grad, hx = self.getGradient(newX, trainy)
            # for each gradient get the sign, if gradient is 0 then sign is 0 too
            for i in range(len(grad)):
                if grad[i] > 0:
                    grad[i] = 1
                elif grad[i] < 0:
                    grad[i] = -1
                else:
                    grad[i] = 0
            # send signs to parameter server, tag 2
            comm.Send(grad, dest=0, tag=2)
            # receive majority voted signs from parameter server, tag 3
            comm.Recv(grad, source=0, tag=3)
            self.W = self.W - alpha*grad  # update weight vector for this particular worker

    def plot_accs(self, step_size=500):
        x_data = []
        y_data = []
        print(len(self.accs))
        for i in range(1, self.num_iterations+1, step_size):
            x_data.append(i)
            y_data.append(self.accs[i-1])
        x_data.append(self.num_iterations)
        y_data.append(self.accs[-1])
        plt.plot(x_data, y_data)
        plt.xticks(x_data)
        plt.xlabel('Number of iterations')
        plt.ylabel('Accuracy')
        plt.show()


# once again, go to readme first
comm = MPI.COMM_WORLD  # initialize mpi
rank = comm.Get_rank()  # get rank, in other words, worker number for this process
NUM_ITERATIONS = 10000  # number of iterations on training data
ALPHA = 0.005  # learning rate

# parameter server
if rank == 0:
    start = time.time()
    accs = []
    LGobj = LogisticReg()  # get a linear regression object for parameter server
    # get a random weight vector including bias
    initialW = LGobj.getInitialW(NUM_FEATURES+1)
    LGobj.W = initialW
    comm.Send(initialW, dest=1, tag=1)  # tag is 1 for initial weight vector
    comm.Send(initialW, dest=2, tag=1)  # tag is 1 for initial weight vector
    for j in range(NUM_ITERATIONS):
        # if j % 100 == 0: # use this for debugging to print number of iterations done
        #    print(j)

        # get gradient signs of worker 1
        grad1 = np.empty(NUM_FEATURES+1, dtype=np.float64)
        comm.Recv(grad1, source=1, tag=2)
        # get gradient signs of worker 2
        grad2 = np.empty(NUM_FEATURES+1, dtype=np.float64)
        comm.Recv(grad2, source=2, tag=2)

        # vector of votes for gradient signs
        vote = np.zeros(NUM_FEATURES+1, dtype=np.float64)
        for i in range(NUM_FEATURES+1):
            vote[i] = grad1[i]+grad2[i]  # will replace this with loop
            if vote[i] > 0:
                vote[i] = 1
            elif vote[i] < 0:
                vote[i] = -1
            else:
                vote[i] = 0

        comm.Send(vote, dest=1, tag=3)  # send sign votes to all workers
        comm.Send(vote, dest=2, tag=3)
        LGobj.W = LGobj.W-ALPHA*vote  # update the weight vector of server
        if j % 1000 == 0:
            accs.append(LGobj.accuracy(X_test, y_test))
    accs.append(LGobj.accuracy(X_test, y_test))
    print(time.time()-start, ' seconds')
    #print(LGobj.accuracy(X_test, y_test))  # print accuracy with test data
    #print('Confusion Matrix: ')
    #print(confusion_matrix(y_test, LGobj.predict(X_test)))

    #x_data = []
    #y_data = []
    # for i in range(1, NUM_ITERATIONS+1, 1000):
    #    x_data.append(i)
    # x_data.append(NUM_ITERATIONS)
    #plt.plot(x_data, accs)
    # plt.xticks(x_data)
    #plt.xlabel('Number of iterations')
    # plt.ylabel('Accuracy')
    # plt.show()

# worker 1
elif rank == 1:
    initialW1 = np.empty(NUM_FEATURES+1, dtype=np.float64)
    comm.Recv(initialW1, source=0, tag=1)
    LGw1 = LogisticReg()
    LGw1.train(initialW1, X_train, y_train, ALPHA, NUM_ITERATIONS)

# worker 2
elif rank == 2:
    initialW2 = np.empty(NUM_FEATURES+1, dtype=np.float64)
    comm.Recv(initialW2, source=0, tag=1)
    LGw2 = LogisticReg()
    LGw2.train(initialW2, X_train, y_train, ALPHA, NUM_ITERATIONS)
