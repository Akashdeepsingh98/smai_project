from mpi4py import MPI

#comm = MPI.COMM_WORLD
#rank = comm.Get_rank()
#
#if rank == 0:
#    data = {'a': 7, 'b': 3.14}
#    comm.send(data, dest=1, tag=11)
#elif rank == 1:
#    data = comm.recv(source=0, tag=11)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
df = pd.read_csv('titanic.csv')

#begin preprocessing
cols = ['Name', 'Ticket', 'Cabin']
df = df.drop(cols, axis=1)

df['Sex'] = df['Sex'].map({'male':0, 'female':1})
df['Embarked'] = df['Embarked'].map({'C':0, 'Q':1, 'S':2})
df['Embarked'] = df['Embarked'].fillna(2)
df['Age'] = df['Age'].interpolate()
print(df.columns)
cols=['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
X=df[cols]
y=df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
