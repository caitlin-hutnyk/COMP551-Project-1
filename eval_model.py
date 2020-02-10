import numpy as np
import pandas as pd
from numpy import ma, math

from testImport import read_data
import LogRegression

def convert_y(y):
	n,c = y.shape
	print(y.shape)
	result = []
	for ar in y:
		for j in range(len(ar)):
			if ar[j]:
				result.append(j)
	print(len(result))
	return np.array(result)

# returns performance rate
def evaluate_acc(y, y_hat):
	success = 0
	if np.shape(y) != np.shape(y_hat):
		print("error: y != y_h")
		# raise SizeError('Size y != size yh')
	if y.shape[1] != 1:
		y = convert_y(y)
		y_hat = convert_y(y_hat)
	for i in range(np.shape(y)[0]):
		if y[i] == y_hat[i]:
			success += 1
	return success/(np.shape(y)[0])


def k_fold(X, k):
	# create a list of k pairs of train and validate sets
	split_list = []
	size = math.floor(X.shape[0] / k)
	# select blocks of size-sized instances to validate, and allocate
	# the rest to train
	for i in range(k - 2):
		v_data = X[i * size:(i + 1) * size]
		v_y, v_X = v_data[:, -1, np.newaxis], v_data[:, :-1]
		validate = (v_X, v_y)

		t_data = np.delete(X, np.s_[i * size: (i + 1) * size], 0)
		t_y, t_X = t_data[:, -1, np.newaxis], t_data[:, :-1]
		train = (t_X, t_y)

		assert (v_y.shape[1] + v_X.shape[1] == t_y.shape[1] + t_X.shape[1] ==
				v_data.shape[1] == t_data.shape[1])
		split_list.append((train, validate))

	# append the final block
	v_data = X[(k - 1) * size:]
	v_y, v_X = v_data[:, -1, np.newaxis], v_data[:, :-1]
	assert (v_y.shape[1] + v_X.shape[1] == v_data.shape[1])
	validate = (v_X, v_y)

	t_data = X[:(k - 1) * size]
	t_y, t_X = t_data[:, -1, np.newaxis], t_data[:, :-1]
	train = (t_X, t_y)

	split_list.append((train, validate))

	return split_list

# find best hyper_param learning rate using k-fold, trying 5 different values
def find_model(X_train, trial_val_y, X_test, test_y):
	train_validate_list = k_fold(np.append(X_train, trial_val_y, axis=1), 5)

	hyper_params = [2, 1.5, 1, 0.5, 0.1]

	performance = []

	for h in range(len(hyper_params)):
		perf = 0
		print("Testing hyperparam lr = " + str(hyper_params[h]))
		for train_validate_set in train_validate_list:

			train, validate = train_validate_set
			train_X, train_y = train
			validate_X, validate_y = validate

			# add first col of ones
			train_X = np.append(np.ones((train_X.shape[0], 1)), train_X, axis=1)
			validate_X = np.append(np.ones((validate_X.shape[0], 1)), validate_X, axis=1)

			# print("train shape {} type {}" .format(train_X.shape, train_X.dtype))
			# print("v shape {} type {}".format(validate_X.shape, validate_X.dtype))

			model = LogRegression.Log_Regression(hyper_params[h], 0.005)
			model.fit(train_X, train_y)
			y_h = model.predict(validate_X)

			# print("training set {} ".format(training_set_count))
			# print("y label: {}" .format(train_y.dtype))
			# print("y predicted: {}" .format(y_h.dtype))

			percent = evaluate_acc(validate_y, y_h)
			print("Success rate: " + str(percent))
			perf += (percent)
		perf /= len(train_validate_list)
		performance.append(perf)

	best_perf = 0
	best_index = 0
	print(hyper_params)
	print(performance)
	for i in range(len(hyper_params)):
		print("" + str(performance[i]) + ", " + str(hyper_params[i]))
		if performance[i] >= best_perf:
			best_perf = performance[i]
			best_index = i

	return hyper_params[best_index]

def main():
	X_train, trial_val_y, X_test, test_y = read_data(3,1)
	print("shapes!!! \n\n\n")
	print(X_train.shape)
	print(X_test.shape)
	print(trial_val_y.shape)
	print(test_y.shape)
	# X = np.concatenate((train_validate_categorical, train_validate_continuous), axis=1)

	# do k-fold split on the training data to get k folds of train and validate
	learning_rate = find_model(X_train, trial_val_y, X_test, test_y)
	print("best learning rate found " + str(learning_rate))

	log_r = LogRegression.Log_Regression(learning_rate, 0.005)
	log_r.fit(X_train, trial_val_y)
	y_h = log_r.predict(X_test)
	percent = evaluate_acc(test_y, y_h)

	print("Final success rate : " + str(percent))

# quick testing to make sure stuff is working
def q_test():
	X_train, trial_val_y, X_test, test_y = read_data(3,1)
	print(X_train.shape)
	print(X_test.shape)
	log_r = LogRegression.Log_Regression(0.1, 0.005)
	log_r.fit(X_train, trial_val_y)
	y_h = log_r.predict(X_test)
	error = evaluate_acc(test_y, y_h)
	print("Final successrate : " + str(error))

def test_2():
	train_validate_continuous, train_validate_categorical, train_val_y, test_continuous, test_categorical, test_y = read_data(0)
	print(train_validate_categorical)

if __name__ == "__main__":
	q_test()