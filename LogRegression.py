import numpy as np

class Log_Regression:

    def __init__(self, lr, eps, m):
        self.learning_rate = lr
        self.stop_cond = eps
        self.max_iterations = m
        self.w = -1

    # w Dx1
    # x DxN
    # gives decision boundary 0.5
    def logistic(self, w, X):
        return 1 / (1 + np.exp(-np.dot(X, w)))

    # added a / N to the grad definition as per lec 7 slide 5.1
    def gradient(self, X, y, w):
        N, D = X.shape
        yh = self.logistic(w, X)
        grad = (1/N) * np.dot(X.T, yh - y)
        return grad

    def gradient_descent(self, X, y, lr, eps):
        N, D = X.shape
        w = np.zeros((D, 1))
        g = np.inf
        its = 0
        while np.linalg.norm(g) > eps and its < self.max_iterations:
            g = self.gradient(X, y, w)
            # print(np.linalg.norm(g))
            w = w - lr * g
            its = its + 1
        print("Terminating gradient descent at iterations: {}".format(its))
        return w

    # for multiclass classification
    def softmax(z):
        yh = np.exp(z)
        yh /= np.sum(yh)
        return yh

    def cost(w,  # N
             X,  # N x D
             y,  # N
             ):
        z = np.dot(X, w)
        J = np.mean(y * np.log1p(np.exp(-z)) + (1 - y) * np.log1p(np.exp(z)))
        return J

    # training
    # X: NxD training data
    # y: Nx1 discrete targets (for now y in {0,1})
    def fit(self, X, y):
        self.w = self.gradient_descent(X, y, self.learning_rate, self.stop_cond)


    # predict and return y_hat
    # X: NxD test data
    def predict(self, X):
        n, d = np.shape(X)
        y_hat = np.zeros((n, 1))
        log = self.logistic(self.w, X)

        # categorical!!
        if log.shape[1] != 1:
        	y_hat = np.zeros(np.shape(log))
        	for i in range(n):
        		index = np.argmax(log[i])
        		y_hat[i,index] = 1
        	return y_hat


        # binary
        for i in range(n):
            if log[i] >= 0.5:
                y_hat[i] = 1
            else:
                y_hat[i] = 0
        return y_hat
