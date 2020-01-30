import numpy as np

# currently just evaluates binary
def evaluate_acc(y, y_hat):
	error = 0
	if np.shape(y) != np.shape(y_hat):
		raise SizeError('Size y != size yh')
	for i in range(np.shape(y)):
		if y[i] != y_hat[i]:
			error += 1
	return error