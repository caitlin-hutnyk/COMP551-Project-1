import numpy as np
import pandas as pd
import constant
from sklearn.utils import shuffle

np.set_printoptions(threshold=np.inf)
pd.set_option('display.max_columns', 500)

def read_data(which, type_):

	if which == constant.IONOSPHERE:
		trainfile = 'data/ionosphere.data'
		columns = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
		         21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]

		# import txt file, keep original layout and format, and display top five instances as sample
		data = pd.read_csv(trainfile, sep=",", header=None, na_values=[" ?", "?", "99999"," 99999"], names=columns)
		data.head()
		data.info()

		data = data.dropna(axis='index', how="any")

		# convert categorical variables into dummy variables
		data_new = pd.get_dummies(data)
		data_new.head()

		# convert from panda data frame to numpy arrays
		data_new_arr = pd.DataFrame(data_new).to_numpy()

		split = int(np.shape(data_new_arr)[0] * 0.8)
		# print (data_new_arr.shape)
		# split original matrix into train and test, and x and y
		train_y = np.array(data_new_arr[:split,-2:-1])
		train_x_con = np.array(data_new_arr[:split, :-2])
		train_x_cat = -1

		test_y = np.array(data_new_arr[split:,-2:-1])
		test_x_con = np.array(data_new_arr[split:, :-2])
		test_x_cat = -1

		if type_:
			return train_x_con, train_y, test_x_con, test_y
		return train_x_con, train_x_cat, train_y, test_x_con, test_x_cat, test_y

	elif which == constant.CENSUS:
		trainfile = 'data/adult.data'
		columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
		            'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
		           'salary']

		# import txt file, keep original layout and format, and display top five instances as sample
		data = pd.read_csv(trainfile, sep=",", header=None, na_values=[" ?", "?", "99999"," 99999"], names=columns)
		data.head()
		data.info()

		data = data.dropna(axis='index', how="any")

		# convert categorical variables into dummy variables
		data_new = pd.get_dummies(data)
		data_new.head()

		# convert from panda data frame to numpy arrays
		data_new_arr = pd.DataFrame(data_new).to_numpy()
		# print (data_new_arr.shape)
		split = int(np.shape(data_new_arr)[0] * 0.8)

		# split original matrix into x and y
		train_y = np.array(data_new_arr[:split,-2:-1])
		train_x_con = np.array(data_new_arr[:split, :6])
		train_x_cat = np.array(data_new_arr[:split, 6:-2])

		test_y = np.array(data_new_arr[split:,-2:-1])
		test_x_con = np.array(data_new_arr[split:, :6])
		test_x_cat = np.array(data_new_arr[split:, 6:-2])

		if type_:
			return np.append(train_x_con, train_x_cat, axis=1), train_y, np.append(test_x_con, test_x_cat, axis=1), test_y
		return train_x_con, train_x_cat, train_y, test_x_con, test_x_cat, test_y

	elif which == constant.POKER:
		trainfile = 'data/poker-hand-training-true.data'
		testfile = 'data/poker-hand-testing.data'
		columns = ['S1', 'C1', 'S2', 'C2', 'S3', 'C3', 'S4', 'C4', 'S5', 'C5', 'CLASS']

		# import txt file, keep original layout and format, and display top five instances as sample
		data = pd.read_csv(trainfile, sep=",", header=None, na_values=[" ?", "?", "99999", " 99999"], names=columns)

		data.head()
		data.info()

		data = data.dropna(axis='index', how="any")

		# convert categorical variables into dummy variables
		data_new = pd.get_dummies(data.astype(str))
		data_new.head()

		# convert from panda data frame to numpy arrays
		data_new_arr = pd.DataFrame(data_new).to_numpy()
		# print (data_new_arr.shape)
		split = int(np.shape(data_new_arr)[0] * 0.8)

		# split original matrix into x and y
		train_y = np.array(data_new_arr[:split, -10:])
		train_x_con = -1
		train_x_cat = np.array(data_new_arr[:split,:-10])

		test_y = np.array(data_new_arr[split:, -10:])
		test_x_con = -1
		test_x_cat = np.array(data_new_arr[split:,:-10])

		if type_:
			return train_x_cat, train_y, test_x_cat, test_y
		return train_x_con, train_x_cat, train_y, test_x_con, test_x_cat, test_y

	elif which == constant.CREDIT:

		trainfile = 'data/crx.data'
		# import txt file, keep original layout and format, and display top five instances as sample
		data = pd.read_csv(trainfile, sep=",", header=None, na_values=[" ?", "?", "99999", " 99999"])
		data.head()
		data.info()

		data = data.dropna(axis='index', how="any")

		# convert categorical variables into dummy variables
		data_new = pd.get_dummies(data)
		data_new.head()

		# convert from panda data frame to numpy arrays
		data_new_arr = pd.DataFrame(data_new).to_numpy()
		# print(data_new_arr.shape)

		split = int(np.shape(data_new_arr)[0] * 0.8)

		# split original matrix into x and y
		train_y = np.array(data_new_arr[:split, -2:-1])
		train_x_con = np.array(data_new_arr[:split, :6])
		train_x_cat = np.array(data_new_arr[:split, 6:-2])

		test_y = np.array(data_new_arr[split:, -2:-1])
		test_x_con = np.array(data_new_arr[split:, :6])
		test_x_cat = np.array(data_new_arr[split:, 6:-2])

		if type_:
			return np.append(train_x_con, train_x_cat, axis=1), train_y, np.append(test_x_con, test_x_cat, axis=1), test_y
		return train_x_con, train_x_cat, train_y, test_x_con, test_x_cat, test_y

# for log_reg
# shuffles order of test cases and returns 'many' many of them
# so that we can standardize how many test cases we have for each example
def less_cases_together(x_t, y_t, many):
	if x_t.shape[0] <= many:
		return (x_t, y_t)
	x_shuf, y_shuf = shuffle(x_t, y_t)
	return x_shuf[:many,:], y_shuf[:many,:]

# same but for x split up for nb
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

def testing():
	x, y, x_test, y_test = read_data(1,1)
	print('initial shapes:')
	print(x.shape)
	print(y.shape)
	print(x_test.shape)
	print(y_test.shape)

	a,b = less_cases_together(x,y,100)

	print('shapes after:')
	print(a.shape)
	print(b.shape)

if __name__ == "__main__":
	testing()