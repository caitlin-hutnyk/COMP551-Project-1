import numpy as np

class Log_Regression:

	def __init__(self, lr, eps):
		self.learning_rate = lr
		self.stop_cond = eps
		self.w = -1
		return self

	# w Dx1
	# x DxN
	# gives decision boundary 0.5
	def logistic(w, x):
		return 1/(1 + np.exp(-np.dot(np.transpose(w), x)))

	def gradient(X, y, w):
		N,D = X.shape
		yh = logistic(w,X)
		grad = np.dot(np.transpose(X), yh - y)
		return grad

	def gradient_descent(X, y, lr, eps):
		N,D = X.shape
		w = np.zeros(D)
		g = np.inf 
		while np.linalg.norm(g) > eps:
			g = gradient(X,y,w)
			w = w - lr*g
		return w

	def softmax(z):
		yh = np.exp(z)
		yh /= np.sum(yh)
		return yh

	def cost(w, # D
			 X, # N x D
			 y, # N
			 ):
		z = np.dot(X,w)
		J = np.mean(y * np.log1p(np.exp(-z)) + (1-y)*np.log1p(np.exp(z)))
		return J


	# training
	# X: NxD training data
	# y: Nx1 discrete targets (for now y in {0,1})
	def fit(X, y):
		self.w = gradient_descent(X,y,self.learning_rate, self.stop_cond)

	# predict and return y_hat
	# X: NxD test data
	def predict(X):
		n,d = np.shape(X)
		y_hat = np.zeros((n,1))
		log = logistic(self.w,X)
		for i in range(n):
			if log[i] >= 0.5:
				y_hat[i] = 1
			else:
				y_hat[i] = 0
		return y_hat
		
