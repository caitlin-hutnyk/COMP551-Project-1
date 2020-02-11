import numpy as np
import pandas as pd

import testImport
import LogRegression
import eval_model
from sklearn.utils import shuffle

def run_k_folds(model, fold_list):
	perf = 0
	for train_validate_set in fold_list:

		train, validate = train_validate_set
		train_X, train_y = train
		validate_X, validate_y = validate

		# add first col of ones
		train_X = np.append(np.ones((train_X.shape[0], 1)), train_X, axis=1)
		validate_X = np.append(np.ones((validate_X.shape[0], 1)), validate_X, axis=1)

		# print("train shape {} type {}" .format(train_X.shape, train_X.dtype))
		# print("v shape {} type {}".format(validate_X.shape, validate_X.dtype))

		model.fit(train_X, train_y)
		y_h = model.predict(validate_X)

		# print("training set {} ".format(training_set_count))
		# print("y label: {}" .format(train_y.dtype))
		# print("y predicted: {}" .format(y_h.dtype))

		percent = eval_model.evaluate_acc(validate_y, y_h)
		perf += (percent)
	perf /= len(fold_list)
	return perf

# runs tests for log_r
# takes lots of parameters, to get a reasonable amount of data keep most lists size 1
# dataset is just 1-4
# lr_list, eps_list, max_list, reg_list all lists of logreg parameters to try
# n_sizes, d_sizes is list of sizes for cases/features you want to try
# folds is how many folds to run in k-fold
def test_model_log(dataset, lr_list, eps_list, max_list, n_sizes, d_sizes, folds, reg_list = None):
	x, y, x_t, y_t = testImport.read_data(1, dataset)

	# currently don't combine removing features and test cases
	# tests each n_size with all the other stuff
	n_performances = []
	for size in n_sizes:
		x_s, y_s = less_cases_together(x,y,size)
		print(x_s.shape)
		print(y_s.shape)
		print(x_t.shape)
		print(y_t.shape)
		fold_list = eval_model.k_fold(x_s, y_s, folds)
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
							real_perf = eval_model.evaluate_acc(y_t, model.predict(np.append(np.ones((x_t.shape[0], 1)), x_t, axis=1)))
							performances.append((lr, eps, m, r, k_perf, real_perf))
					else: 
						model = LogRegression.Log_Regression(lr, eps, m)
						k_perf = run_k_folds(model, fold_list)
						real_perf = eval_model.evaluate_acc(y_t, model.predict(np.append(np.ones((x_t.shape[0], 1)), x_t, axis=1)))
						performances.append((lr, eps, m, k_perf, real_perf))
		n_performances.append(performances)
					
	#tests each d_size with all the other stuff
	d_performances = []
	for size in d_sizes:
		x_s, x_t_s = less_features(x,x_t,size)
		fold_list = eval_model.k_fold(x_s, y, folds)
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
							real_perf = eval_model.evaluate_acc(y_t, model.predict(np.append(np.ones((x_t_s.shape[0], 1)), x_t_s, axis=1)))
							performances.append((lr, eps, m, r, k_perf, real_perf))
					else: 
						model = LogRegression.Log_Regression(lr, eps, m)
						k_perf = run_k_folds(model, fold_list)
						real_perf = eval_model.evaluate_acc(y_t, model.predict(np.append(np.ones((x_t_s.shape[0], 1)), x_t_s, axis=1)))
						performances.append((lr, eps, m, k_perf, real_perf))
		d_performances.append(performances)

	return n_performances, d_performances

def test_model_nb(dataset):
	pass

# runs test_model on both model types, all 4 datasets
# returns a tuple of tuples
# ((4 results in order for log_r), (4 results in order for nb))
# so result[1][3] would be n_performances, d_performances for log_r, poker hands
def iterate(lr_list, eps_list, max_list, n_sizes, d_sizes, folds, reg_list = None):
	log_r = []
	nb = []

	for i in range(4):
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

def test():
	result = test_model_log(1, [1, 0.5, 1], [0.005], [20000], [500, 250, 100], [100], 5)
	print(result)

if __name__ == "__main__":
	test()