from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape((x_train.shape[0], -1))
x_test = x_test.reshape((x_test.shape[0], -1))

X = np.concatenate((x_train, x_test), axis=0)
Y = np.concatenate((y_train, y_test), axis=0)

X = X.astype('float32')/255.0
Y = to_categorical(Y)

x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.15, random_state=42)

np.savetxt('x_train.csv', x_train, delimiter=',')
np.savetxt('x_test.csv', x_test, delimiter=',')
np.savetxt('y_train.csv', y_train, delimiter=',')
np.savetxt('y_test.csv', y_test, delimiter=',')
