import numpy as np

class Log_Regression:

	def __init__(self):
		pass

	# training
	# X: NxD training data
	# y: Nx1 discrete targets (for now y in {0,1})
	def fit(X, y):
		pass

	# predict and return y_hat
	# X: NxD test data
	def predict(X):
		n,d = np.shape(X)[0], np.shape(X)[1]
		y_hat = np.zeros((n,1))
		
