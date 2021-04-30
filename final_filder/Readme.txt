Important files
- NN_breastCancer_normal.py: Neural network non-distributed for breast cancer.
- NN_breastCancer_distributed.py: Neural network distributed for breast cancer.
- NN_MNIST_normal.py: Sign SGD on MNIST Neural network non-distributed.
- NN_MNIST_distributed.py: Sign SGD on MNIST Neural network distributed.
- LR_titanic_normal.py: Titanic Logistic Regression non-distributed
- LR_titanic_distributed.py: Titanic Logistic Regression distributed
- LR_SignLanguage.py -Logistic regression on Sign Language



Datasets Links:
- MNSIT dataset: http://yann.lecun.com/exdb/mnist/
- Breast Cancer Dataset: https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/
- Titanic Dataset: https://www.kaggle.com/c/titanic/data
- Sign Language Digits Dataset: https://www.kaggle.com/ardamavi/sign-language-digits-dataset



Method to run each file:
1. NN_breastCancer_normal.py: python NN_breastCancer_normal.py 
2. NN_breastCancer_distributed.py: mpiexec -n 3 python NN_breastCancer_distributed.py
3. NN_MNIST_normal.py: python NN_MNIST_normal.py
4. NN_MNIST_distributed.py: mpiexec -n 4 NN_MNIST_distributed.py
5. LR_titanic_normal.py: python LR_titanic_normal.py 
6. LR_titanic_distributed.py: python LR_titanic_distributed.py
7. LR_SignLanguage.py: python LR_SignLanguage.py
