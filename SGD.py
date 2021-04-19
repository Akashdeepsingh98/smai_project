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
NUM_FEATURES = len(cols)
# print(X_train[:5])
# exit()


class LogRegSGD:  # SIGNSGD
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

class LogReg:
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


class LogisticReg:
    def __init__(self):
        self.W = None

    def getInitialW(self, numFeatures: int):
        return np.random.rand(numFeatures)

    def train(self, W, trainX, trainy, alpha, iterations=10000):
        pass


comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# parameter server
if rank == 0:
    LGobj = LogisticReg()
    initialW = LGobj.getInitialW(NUM_FEATURES+1)
    #print('param server', initialW)
    comm.Send(initialW, dest=1, tag=1)  # tag is 1 for initial weight vector
    #print('param server', initialW)
    comm.Send(initialW, dest=2, tag=1)  # tag is 1 for initial weight vector

# worker 1
elif rank == 1:
    initialW1 = np.empty(NUM_FEATURES+1, dtype=np.float64)
    comm.Recv(initialW1, source=0, tag=1)
    #print('worker 1', initialW1)

# worker 2
elif rank == 2:
    initialW2 = np.empty(NUM_FEATURES+1, dtype=np.float64)
    comm.Recv(initialW2, source=0, tag=1)
    #print('worker 2', initialW2)
