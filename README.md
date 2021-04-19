# SMAI Project - SIGNSGD
- Have to make a new class for logistic regression that will be specifically for signsgd.
- Working:
1. Parameter server will create a random weight vector and give it to all the workers.
2. Each worker will have its own subset of data.
3. Whenever each workers is done training a batch, it will share signs of gradient with parameter server.
4. The parameter server will use majority vote and get signs of each gradient.
5. The parameter server sends signs to all the workers (paper algorithm 3) as well as updates its own weight vector (paper algorithm 1).