# Important files
- NN_breastCancer_normal.py: Neural network non-distributed for breast cancer.
- NN_breastCancer_distributed.py: Neural network distributed for breast cancer.
- NN_MNIST_normal.py: Sign SGD on MNIST Neural network non-distributed.
- NN_MNIST_distributed.py: Sign SGD on MNIST Neural network non-distributed.
- LR_titanic_normal.py: Titanic Logistic Regression non-distributed
- LR_titanic_distributed.py: Titanic Logistic Regression distributed
- LR_SignLanguage-Logistic regression on Sign Language
- prepdata.py: Prepare mnist data for neural network.

# SMAI Project - SIGNSGD
- Have to make a new class for logistic regression that will be specifically for signsgd.
- Working:
1. Parameter server will create a random weight vector and give it to all the workers.
2. Each worker will have its own subset of data.
3. Whenever each workers is done training a batch, it will share signs of gradient with parameter server.
4. The parameter server will use majority vote and get signs of each gradient.
5. The parameter server sends signs to all the workers (paper algorithm 3) as well as updates its own weight vector (paper algorithm 1).
- The Logistic Regression class being used is names 'LogisticReg'. It is the 3rd one among the 3 logistic regression classes. Other 2 are only for reference that I used.
- Rank 0 is parameter server.
- Ranks 1 and 2 are workers.
- For now both workers are using entire dataset to train, but with more work each worker will have an equal size subset of training data.
- Testing data is only fed on parameter server.
- The parameter server will take gradient signs from both workers, do majority voting and send signs back to workers, and it will do it as many times as number of iterations.