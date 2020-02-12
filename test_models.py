import numpy as np
import pandas as pd

import testImport
import LogRegression
import NaiveBayes
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

def run_k_folds(params, fold_list):
	perf = 0
	for train_validate_set in fold_list:

		train, validate = train_validate_set
		train_X, train_y = train
		validate_X, validate_y = validate

		# add first col of ones

		# print("train shape {} type {}" .format(train_X.shape, train_X.dtype))
		# print("v shape {} type {}".format(validate_X.shape, validate_X.dtype))
		x,y,z = params
		model = LogRegression.Log_Regression(x,y,z)
		model.fit(train_X, train_y)
		y_h = model.predict(validate_X)

		# print("training set {} ".format(training_set_count))
		# print("y label: {}" .format(train_y.dtype))
		# print("y predicted: {}" .format(y_h.dtype))

		percent = evaluate_acc(validate_y, y_h)
		perf += percent
	perf /= len(fold_list)
	return perf

def run_k_folds_best(params, fold_list):
	best_p = 0
	best_model = -1
	for train_validate_set in fold_list:

		train, validate = train_validate_set
		train_X, train_y = train
		validate_X, validate_y = validate

		# add first col of ones

		# print("train shape {} type {}" .format(train_X.shape, train_X.dtype))
		# print("v shape {} type {}".format(validate_X.shape, validate_X.dtype))
		x,y,z = params
		model = LogRegression.Log_Regression(x,y,z)
		model.fit(train_X, train_y)
		y_h = model.predict(validate_X)

		# print("training set {} ".format(training_set_count))
		# print("y label: {}" .format(train_y.dtype))
		# print("y predicted: {}" .format(y_h.dtype))

		percent = evaluate_acc(validate_y, y_h)
		if percent > best_p:
			best_p = percent
			best_model = model
	return model

# runs tests for log_r
# takes lots of parameters, to get a reasonable amount of data keep most lists size 1
# dataset is just 1-4
# lr_list, eps_list, max_list, reg_list all lists of logreg parameters to try
# n_sizes, d_sizes is list of sizes for cases/features you want to try
# folds is how many folds to run in k-fold
def test_model_log(dataset, lr_list, eps_list, max_list, n_sizes, d_sizes, folds, reg_list = None):
	x, y, x_t, y_t = testImport.read_data(dataset, 1)

	# currently don't combine removing features and test cases
	# tests each n_size with all the other stuff
	n_performances = []
	for size in n_sizes:
		x_s, y_s = less_cases_together(x,y,size)
		'''
		print(x_s.shape)
		print(y_s.shape)
		print(x_t.shape)
		print(y_t.shape)
		'''
		fold_list = k_fold(x_s, y_s, folds)
		# performance is a list of tuples
		# (lr, eps, m, (reg if its not none), k_fold performance, test performance)
		performances = []

		# test all possible pairings of the different log_r parameters
		for lr in lr_list:
			for eps in eps_list:
				for m in max_list:
					if reg_list != None:
						for r in reg_list:
							model = LogRegression.Log_Regression(lr, eps, m, r)
							k_perf = run_k_folds(model, fold_list)
							real_perf = evaluate_acc(y_t, model.predict(x_t))
							performances.append((size, lr, eps, m, r, k_perf, real_perf))
					else: 
						model = LogRegression.Log_Regression(lr, eps, m)
						k_perf = run_k_folds(model, fold_list)
						real_perf = evaluate_acc(y_t, model.predict(x_t))
						performances.append((size, lr, eps, m, k_perf, real_perf))
		n_performances.append(performances)
					
	#tests each d_size with all the other stuff
	d_performances = []
	for size in d_sizes:
		x_s, x_t_s = less_features(x,x_t,size)
		fold_list = k_fold(x_s, y, folds)
		# performance is a list of tuples
		# (lr, eps, m, (reg if its not none), performance)
		performances = []

		# test all possible pairings of the different log_r parameters
		for lr in lr_list:
			for eps in eps_list:
				for m in max_list:
					if reg_list != None:
						for r in reg_list:
							model = LogRegression.Log_Regression(lr, eps, m, r)
							k_perf = run_k_folds(model, fold_list)
							real_perf = evaluate_acc(y_t, model.predict(x_t_s))
							performances.append((size, lr, eps, m, r, k_perf, real_perf))
					else: 
						model = LogRegression.Log_Regression(lr, eps, m)
						k_perf = run_k_folds(model, fold_list)
						real_perf = evaluate_acc(y_t, model.predict(x_t_s))
						performances.append((size, lr, eps, m, k_perf, real_perf))
		d_performances.append(performances)

	return n_performances, d_performances

def test_model_nb(dataset):
	X_con, X_cat, Y, test_con, test_cat, test_y = testImport.read_data(dataset, 0)

	model = NaiveBayes.NaiveBayes()
	model.fit(X_con, X_cat, Y)
	y_hat = model.predict(test_con, test_cat)

	ac = evaluate_acc_NB(test_y, y_hat)
	print(ac)


# runs test_model on both model types, all 4 datasets
# returns a tuple of tuples
# ((4 results in order for log_r), (4 results in order for nb))
# so result[1][3] would be n_performances, d_performances for log_r, poker hands
def iterate(lr_list, eps_list, max_list, n_sizes, d_sizes, folds, reg_list = None):
	log_r = []
	nb = []
	for i in range(1,5):
		log_r.append(test_model_log(i, lr_list, eps_list, max_list, n_sizes, d_sizes, folds, reg_list))
		nb.append(test_model_nb(i, lr_list, eps_list, max_list, n_sizes, d_sizes, folds, reg_list))
	return (tuple(log_r), tuple(nb))


# for log_reg
# shuffles order of test cases and returns 'many' many of them
# so that we can standardize how many test cases we have for each example
def less_cases_together(x_t, y_t, many):
	if x_t.shape[0] <= many:
		return (x_t, y_t)
	x_shuf, y_shuf = shuffle(x_t, y_t)
	return x_shuf[:many,:], y_shuf[:many,:]

# same but for x split up for nb
# if one of the x cases is None simply use the less_cases_together instead
def less_cases_separate(x_con, x_cat, y_t, many):
	if x_con.shape[0] <= many:
		return (x_con, x_cat, y_t)
	x_con_shuf, x_cat_shuf, y_shuf = shuffle(x_con, x_cat, y_t)
	return x_con_shuf[:many,:], x_cat_shuf[:many,:], y_shuf[:many,:]

# randomly reduce the number of features in x NxD
# for x_con and x_cat split up, should be done separately for each
# although this needs to be done for both testing and training data!!!
def less_features(x_train, x_test, many):
	if x_train.shape[1] <= many:
		return x_train, x_test
	x_train, x_test = shuffle(x_train.T, x_test.T)
	return x_train.T[:, :many], x_test.T[:,:many]

def convert_y(y):
	n,c = y.shape
	# print(y.shape)
	result = []
	for ar in y:
		for j in range(len(ar)):
			if ar[j]:
				result.append(j)
	# print(len(result))
	return np.array(result)

# returns performance rate
def evaluate_acc(y, y_hat):
	success = 0
	if np.shape(y) != np.shape(y_hat):
		print("error: y != y_h")
		print(y.shape)
		print(y_hat.shape)
		# raise SizeError('Size y != size yh')
	if y.shape[1] != 1:
		y = convert_y(y)
		y_hat = convert_y(y_hat)
	for i in range(np.shape(y)[0]):
		if y[i] == y_hat[i]:
			success += 1
	return success/(np.shape(y)[0])

def evaluate_acc_NB(y, y_hat):
	N = y.shape[0]
	C = y.shape[1]
	if C == 1:
		y = convertY_NB(y)
	success = 0

	if np.shape(y) != np.shape(y_hat):
		print("error: y != y_h")
	for n in range(N):
		y_row = y[n, :]
		y_idx = np.nonzero(y_row)
		y_hat_row = y_hat[n, :]
		y_hat_idx = np.nonzero(y_hat_row)
		if y_idx == y_hat_idx:
			success += 1
	return success / N

def convertY_NB(y):
	N = y.shape[0]
	y = np.append(np.zeros((N, 1)), y, axis=1)
	for n in range(N):
		if y[n][1] == 0:
			y[n][0] = 1
	return y

# receoves X and Y appended to the end
def k_fold(x, y, k):
	split_list = []
	size = int(x.shape[0] / k)

	for i in range(k-1):
		x_v = x[i*size : (i+1)*size]
		y_v = y[i*size : (i+1)*size]

		x_t = np.delete(x, np.s_[i*size : (i+1)*size], 0)
		y_t = np.delete(y, np.s_[i*size : (i+1)*size], 0)

		t = (x_t,y_t)
		v = (x_v,y_v)


		split_list.append((t,v))

	x_v = x[(k-1)*size :]
	y_v = y[(k-1)*size :]

	x_t = np.delete(x, np.s_[(k-1)*size :], 0)
	y_t = np.delete(y, np.s_[(k-1)*size :], 0)

	t = (x_t,y_t)
	v = (x_v,y_v)


	split_list.append((t,v))

	return split_list

# for naive bayes, having x_con and x_cat split up
def k_fold_split(x1, x2, y, k):
	split_list = []
	if x1 is None:
		size = int(x2.shape[0] / k)
	else:
		size = int(x1.shape[0] / k)

	for i in range(k-1):
		# initialise to None in case we have no con or cat features in a dataset
		x1_v, x1_t, x2_v, x2_t = None, None, None, None
		if x1 is not None:
			x1_v = x1[i * size: (i + 1) * size]
			x1_t = np.delete(x1, np.s_[i * size: (i + 1) * size], 0)

		if x2 is not None:
			x2_v = x2[i*size : (i+1)*size]
			x2_t = np.delete(x2, np.s_[i * size: (i + 1) * size], 0)

		y_v = y[i * size: (i + 1) * size]
		y_t = np.delete(y, np.s_[i * size: (i + 1) * size], 0)

		t = (x1_t, x2_t, y_t)
		v = (x1_v, x2_v, y_v)

		split_list.append((t,v))

	x1_v, x1_t, x2_v, x2_t = None, None, None, None
	if x1 is not None:
		x1_v = x1[(k-1)*size :]
		x1_t = np.delete(x1, np.s_[(k - 1) * size:], 0)

	if x2 is not None:
		x2_v = x2[(k-1)*size :]
		x2_t = np.delete(x2, np.s_[(k-1)*size :], 0)

	y_v = y[(k - 1) * size:]
	y_t = np.delete(y, np.s_[(k-1)*size :], 0)

	t = (x1_t, x2_t, y_t)
	v = (x1_v, x2_v, y_v)

	split_list.append((t,v))

	return split_list

# for logistic regression
# list of iteration lengths
def test_lr_vs_its(dataset, lr_list):
	x, y, x_t, y_t = testImport.read_data(dataset, 1)
	result = []
	for lr in lr_list:
		model = LogRegression.Log_Regression(lr, 0.005, 25000)
		model.fit(x,y)
		result.append(model.compute_avg_its())
	return result

def lr_vs_its(lr_list):
	for i in range(1,5):
		avg_its = test_lr_vs_its(i, lr_list)
		plt.plot(lr_list, avg_its)
		plt.xlabel('learning rate')
		plt.ylabel('average iterations')
	plt.legend(['ionosphere', 'census', 'poker hands', 'credit rating'])
	plt.savefig('log_r_testing/lr_vs_its')

def test_lr_vs_perf(dataset, lr_list):
	x, y, x_t, y_t = testImport.read_data(dataset, 1)
	folds = k_fold(x,y,5)
	result = []
	for lr in lr_list:
		perf = run_k_folds((lr, 0.005, 25000), folds)
		result.append(perf)
	return result

def lr_vs_perf(lr_list):
	for i in range(1,5):
		performances = test_lr_vs_perf(i, lr_list)
		plt.plot(lr_list, performances)
		plt.xlabel('learning rate')
		plt.ylabel('performance')
	plt.legend(['ionosphere', 'census', 'poker hands', 'credit rating'])
	plt.savefig('log_r_testing/lr_vs_perf')


def test_n_vs_perf(dataset, n_list):
	x, y, x_t, y_t = testImport.read_data(dataset, 1)
	perf_list = []
	for n in n_list:
		xs, ys = less_cases_together(x, y, n)
		model = LogRegression.Log_Regression(1,0.005,10000)
		model.fit(xs, ys)
		perf = evaluate_acc(y_t, model.predict(x_t))
		perf_list.append(perf)
	return perf_list

# test dataset size vs performance on test data
# for logistic regression
def n_vs_perf(n_list):
	for i in range(1,5):
		perf_list = test_n_vs_perf(i, n_list)
		plt.plot(n_list, perf_list)
		plt.xlabel('dataset size')
		plt.ylabel('performance')
	plt.legend(['ionosphere', 'census', 'poker hands', 'credit rating'])
	plt.savefig('log_r_testing/n_vs_perf2')

def test_d_vs_perf(dataset, d_list):
	x, y, x_t, y_t = testImport.read_data(dataset, 1)
	print(x.shape[1])
	perf_list = []
	for d in d_list:
		xs, xs_t = less_features(x, x_t, d)
		model = LogRegression.Log_Regression(1,0.005,10000)
		model.fit(xs, y)
		perf = evaluate_acc(y_t, model.predict(xs_t))
		perf_list.append(perf)
	return perf_list
# test dataset size vs performance on test data
# for logistic regression
def d_vs_perf(d_list):
	for i in range(1,5):
		perf_list = test_d_vs_perf(i, d_list)
		plt.plot(d_list, perf_list)
		plt.xlabel('features size')
		plt.ylabel('performance')
	plt.legend(['ionosphere', 'census', 'poker hands', 'credit rating'])
	plt.savefig('log_r_testing/d_vs_perf') 

def test():
	lr_list = [2,1.75,1.5,1.25,1,0.75,0.5,0.25,0.1]
	avg_its = test_lr_vs_its(1, lr_list)
	print(lr_list)
	print(avg_its)
	plt.plot(lr_list, avg_its)
	plt.xlabel('learning rate')
	plt.ylabel('average iterations')
	plt.show()

def val_vs_perf():
	full_p = []
	k_p = []
	k_on_t_p = []
	for i in range(1,5):
		x,y,x_t, y_t = testImport.read_data(i, 1)
		full_model = LogRegression.Log_Regression(1, 0.005, 25000)
		full_model.fit(x,y)
		perf = evaluate_acc(y_t, full_model.predict(x_t))
		full_p.append(perf)
		perf = run_k_folds((1, 0.005, 25000), k_fold(x,y,5))
		k_model = run_k_folds_best((1, 0.005, 25000), k_fold(x,y,5))
		k_p.append(perf)
		perf = evaluate_acc(y_t, k_model.predict(x_t))
		k_on_t_p.append(perf)
	print(full_p)
	print(k_p)
	print(k_on_t_p)

def test_both():
	log_res = []
	nb_res = []
	for i in range(1,5):
		x, y, x_test, y_test = testImport.read_data(i,1)
		x_con, x_cat, y_, xt_con, xt_cat, yt = testImport.read_data(i,0)

		log = LogRegression.Log_Regression(1, 0.005, 25000)
		nb = NaiveBayes.NaiveBayes()

		log.fit(x,y)
		nb.fit(x_con, x_cat, y)

		log_per = evaluate_acc(y_test, log.predict(x_test))
		nb_per = evaluate_acc_NB(yt, nb.predict(xt_con, xt_cat))
		log_res.append(log_per)
		nb_res.append(nb_per)
	print(log_res)
	print(nb_res)

if __name__ == "__main__":
	test_both()