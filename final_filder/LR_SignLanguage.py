# dataset used is digit sign language.
# dataset https://www.kaggle.com/ardamavi/sign-language-digits-dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
import math
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import time
x_l = np.load("X.npy") # image
y_l = np.load("Y.npy") # label
x_l.shape
pixels = x_l.flatten().reshape(2062, 4096)
df=pd.DataFrame(pixels)
#print(pixels.shape)
print(df.shape)
x=list(np.zeros(10))


###


class LogisticRegression_custom:

    def __init__(self):
        self.num_iterations = 10000
        self.learning_rate = 0.05
        self.weights = 0
        self.bias = 0

    def batchgradient(self, X, y):
        num_samples, num_features = X.shape
        linear_model = np.dot(X, self.weights)
        y_pred = self.sigmoid(linear_model)
        dw = (1 / num_samples) * np.dot(X.T, (y_pred - y))
        # print(dw)
        return dw

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.ones(num_features)
        self.bias = 0

        # gradient descent
        convergence = False
        X, y = shuffle(X, y)
        batches = int(math.floor((len(X) / 128)))
        # while(convergence is not True):
        for i in range(self.num_iterations):
            # Wx+B
            linear_model = np.dot(X, self.weights)
            y_pred = self.sigmoid(linear_model)
            gradient = 0
            vote = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            for b in range(batches):
                if (b != batches - 1):
                    gradient += self.batchgradient(X[b * 128:((b + 1) * 128)], y[b * 128:((b + 1) * 128)])
                else:
                    gradient += self.batchgradient(X[b * 128:len(X)], y[b * 128:len(X)])
                # print(gradient)
                for i in range(len(vote)):
                    if (gradient[i] < 0):
                        vote[i] += -1
                    else:
                        vote[i] = +1

            dw = gradient / batches
            for i in range(len(vote)):
                if (vote[i] < 0):
                    if (dw[i] >= 0):
                        dw[i] = (dw[i] * -1)
                else:
                    if (dw[i] <= 0):
                        dw[i] = dw[i] * -1

            old_wts = self.weights
            # old_bias=self.bias
            self.weights = self.weights - self.learning_rate * dw
            # self.bias=self.bias-self.learning_rate*db
            diff_weights = np.sum(old_wts - self.weights)

    def one_vs_all(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_model)
        return y_pred.T

    def one_vs_one(self, X, weight):
        prob = np.dot(X, weight)
        if (prob >= 0.5):
            return 1
        else:
            return 0

    def sigmoid(self, x):
        return (1 / (1 + np.exp(-x)))
    # print(class_list[0])
    # print(res)


#####

start = time.time()
X_train, X_test, y_train, y_test = train_test_split(df, y_l, test_size=0.3)
class_list=[]
vector=np.zeros(10)
for i in range(10):
    vector[i]=1
    class_vector_test=np.array(np.dot(y_test,vector))
    class_vector_train=np.array(np.dot(y_train,vector))
    xcla=LogisticRegression_custom()
        #X_train, X_test, y_train, y_test = train_test_split(df, class_vector, test_size=0.3)
    xcla.fit(X_train,class_vector_train)
    class_list.append(xcla.one_vs_all(X_test))
    vector[i]=0
    #print(class_list)
print(time.time()-start,' seconds')


result_class=[]
for i in range(len(class_list[0])):
    max_ac=0
    maxpos=0
    for j in range(len(class_list)):
        if(class_list[j][i]>max_ac):
            max_ac=class_list[j][i]
            maxpos=j
    result_class.append(maxpos)
#print(result_class)
test_class=[]
for x in y_test:
    for j in range(len(x)):
        if(x[j]==1):
            test_class.append(j)
            break
print("accuracy one vs all",accuracy_score(test_class,result_class))
print("confusion matrix for one vs all is",confusion_matrix(test_class,result_class))
