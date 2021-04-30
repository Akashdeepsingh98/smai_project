import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import math
import time
from sklearn.utils import shuffle


class LogisticRegression_custom:

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
            # Wx+B
            linear_model = np.dot(X, self.weights)
            y_pred = self.sigmoid(linear_model)
            gradient = 0
            vote = [0, 0, 0, 0, 0, 0, 0]
            for b in range(batches):
                if(b != batches-1):
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


if __name__ == "__main__":
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix
    import pandas as pd
    import numpy as np
    start = time.time()
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
    xcla = LogisticRegression_custom()
    xcla.fit(X_train, y_train)
    res = xcla.predict(X_test)
    #print("Normal gradient descent with Sign SGD",accuracy_score(res,y_test))
    #print(confusion_matrix(y_test, res))
    print(time.time()-start, ' seconds')
    xcla.plot_accs(1000)
